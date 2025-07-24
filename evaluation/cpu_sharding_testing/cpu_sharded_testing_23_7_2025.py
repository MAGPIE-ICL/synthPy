import numpy as np

import sys
import os

sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')
#sys.path.insert(0, 'C:/Users/samma/programming/synthPy/src/simulator')

import importlib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--core", type = int)
args = parser.parse_args()

n_cells = 128
if args.domain is not None:
    n_cells = args.domain

Np = 10000
if args.rays is not None:
    Np = args.rays

print("\nStarting cpu sharding test run with", n_cells, "cells and", Np, "rays.")

if args.core is not None:
    core_limit = args.core
else:
    from multiprocessing import cpu_count
    core_limit = cpu_count()

assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(core_limit)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax

with jax.checking_leaks():
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_traceback_filtering', 'off')

    extent_x = 5e-3
    extent_y = 5e-3
    extent_z = 10e-3

    probing_extent = extent_z

    lengths = 2 * np.array([extent_x, extent_y, extent_z])

    import jax.numpy as jnp

    from scipy.constants import c
    from scipy.constants import e

    class ScalarDomain():
        def __init__(self, lengths, dim):
            self.x_length, self.y_length, self.z_length = lengths[0], lengths[1], lengths[2]
            self.x_n, self.y_n, self.z_n = dim, dim, dim

            self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
            self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
            self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))

            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)
            self.ZZ = None

            self.XX = self.XX.at[:, :].set(self.XX / 2e-3)
            self.XX = self.XX.at[:, :].set(10 ** self.XX)

            self.YY = self.YY.at[:, :].set(self.YY / 1e-3)
            self.YY = self.YY.at[:, :].set(jnp.pi * self.YY)
            self.YY = self.YY.at[:, :].set(2 * self.YY)
            self.YY = self.YY.at[:, :].set(jnp.cos(self.YY))
            self.YY = self.YY.at[:, :].set(1 + self.YY)

            self.ne = self.XX * self.YY

            self.ne = self.ne.at[:, :].set(1e24 * self.ne)

    domain = ScalarDomain(lengths, n_cells)

    lwl = 1064e-9

    Np = 10000
    divergence = 5e-5
    beam_size = extent_x
    ne_extent = probing_extent
    beam_type = 'circular'

    def init_beam(Np, beam_size, divergence, ne_extent):
        s0 = jnp.zeros((9, Np))

        t  = 2 * jnp.pi * np.random.randn(Np)

        u  = np.random.randn(Np)

        ϕ = jnp.pi * np.random.randn(Np)
        χ = divergence * np.random.randn(Np)

        s0 = s0.at[0, :].set(beam_size * u * jnp.cos(t))
        s0 = s0.at[1, :].set(beam_size * u * jnp.sin(t))
        s0 = s0.at[2, :].set(-ne_extent)

        s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
        s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
        s0 = s0.at[5, :].set(c * jnp.cos(χ))

        s0 = s0.at[6, :].set(1.0)
        s0 = s0.at[8, :].set(0.0)
        s0 = s0.at[7, :].set(0.0)

        return s0

    beam_definition = init_beam(Np, beam_size, divergence, ne_extent)

    from scipy.integrate import odeint, solve_ivp
    from time import time
    from sys import getsizeof as getsizeof_default

    from utils import getsizeof
    #from utils import trilinearInterpolator
    from utils import mem_conversion

    def trilinearInterpolator(coordinates, length, dim, values, query_points, *, fill_value = jnp.nan):
        idr = jnp.array(len(points), 3)
        wr = jnp.array(len(points), 3)

        for i in range(len(points)):
            idr = idr.at[i, :].set(jnp.floor(points[i] * dim / length))
            wr = wr.at[i, :].set((points[i] - coordinates[idr[i]]) / (coordinates[idr[i] + 1] - coordinates[idr[i]]))

        def get_val(dx, dy, dz):
            return values[idr[0, :] + dx, idr[1, :] + dy, idr[2, :] + dz]

        return (
            get_val(0, 0, 0) * (1 - wr[0]) * (1 - wr[1]) * (1 - wr[2]) +
            get_val(1, 0, 0) *      wr[0]  * (1 - wr[1]) * (1 - wr[2]) +
            get_val(0, 1, 0) * (1 - wr[0]) *      wr[1]  * (1 - wr[2]) +
            get_val(0, 0, 1) * (1 - wr[0]) * (1 - wr[1]) *      wr[2]  +
            get_val(1, 1, 0) *      wr[0]  *      wr[1]  * (1 - wr[2]) +
            get_val(1, 0, 1) *      wr[0]  * (1 - wr[1]) *      wr[2]  +
            get_val(0, 1, 1) * (1 - wr[0]) *      wr[1]  *      wr[2]  +
            get_val(1, 1, 1) *      wr[0]  *      wr[1]  *      wr[2]
        )

    def calc_dndr(ScalarDomain, lwl = 1064e-9):
        omega = 2 * jnp.pi * c / lwl
        nc = 3.14207787e-4 * omega ** 2

        return (jnp.array(ScalarDomain.ne / nc, dtype = jnp.float32), omega)

    def dndr(r, ne, omega, x, y, z):
        grad = jnp.zeros_like(r)

        dndx = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
        grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), 1, 1, dndx, r.T, fill_value = 0.0))
        del dndx

        dndy = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
        grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), 1, 1, dndy, r.T, fill_value = 0.0))
        del dndy

        dndz = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), z, axis = 2)
        grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), 1, 1, dndz, r.T, fill_value = 0.0))
        del dndz

        return grad

    def dsdt(t, s, ne, x, y, z, omega):
        s = jnp.reshape(s, (9, 1))
        sprime = jnp.zeros_like(s)

        r = s[:3, :]
        v = s[3:6, :]

        amp = s[6, :]

        del s

        sprime = sprime.at[3:6, :].set(dndr(r, ne, omega, x, y, z))
        sprime = sprime.at[:3, :].set(v)

        del r
        del v
        del amp

        return sprime.flatten()

    def solve(s0_import, coordinates, dim, probing_depth, ne, omega, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False):
        Np = s0_import.shape[1]

        t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
        norm_factor = jnp.max(t)

        args = (ne, *coordinates, omega)

        available_devices = jax.devices()

        running_device = jax.lib.xla_bridge.get_backend().platform
        print("\nRunning device:", running_device, end='')

        s0_transformed = s0_import.T
        del s0_import

        core_count = int(os.environ['XLA_FLAGS'].replace("--xla_force_host_platform_device_count=", ''))
        print(", with:", core_count, "cores.")

        from jax.sharding import PartitionSpec as P, NamedSharding

        mesh = jax.make_mesh((core_count,), ('rows',))

        Np = ((Np // core_count) * core_count)
        assert Np > 0, "Not enough rays to parallelise over cores, increase to at least " + str(core_count)

        s0 = jax.device_put(s0_transformed[0:Np, :], NamedSharding(mesh, P('rows', None)))  # 'None' means don't shard axis 0

        print(s0.sharding)

        del s0_transformed

        def dsdt_ODE(t, y, args):
            return dsdt(t, y, *args) * norm_factor

        from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

        def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
            term = ODETerm(dydt)
            solver = Tsit5()
            saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
            stepsize_controller = PIDController(rtol = rtol, atol = atol)

            return lambda s0, args : diffeqsolve(
                term,
                solver,
                y0 = jnp.array(s0),
                args = args,
                t0 = t0,
                t1 = t1,
                dt0 = (t1 - t0) * norm_factor / Nt,
                saveat = saveat,
                stepsize_controller = stepsize_controller,
                max_steps = dim[0] * dim[1] * dim[2] * 100
            )

        ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, save_steps)

        if jitted:
            start_comp = time()

            from equinox import filter_jit
            ODE_solve = filter_jit(ODE_solve)
            print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

        start = time()
        sol = jax.block_until_ready(
            jax.vmap(ODE_solve, in_axes = (0, None))(s0, args)
        )

        duration = time() - start

        del s0

        return sol.ys[:, -1, :].T

    rf = solve(
        beam_definition,
        (domain.x, domain.y, domain.z),
        (domain.x_n, domain.y_n, domain.z_n),   # domain.dim - this causes a TracerBoolConversionError, check why later, could be interesting and useful to know
        ne_extent,
        *calc_dndr(domain, lwl)
    )

    print("\nRun complete!")