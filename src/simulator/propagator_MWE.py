import jax
import jax.numpy as jnp
import os

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

from scipy.constants import c
from scipy.constants import e

from utils import getsizeof
from utils import mem_conversion
from utils import colour
from utils import add_integer_postfix

@jax.jit
def trilinearInterpolator(x, y, z, lengths, dims, values, query_points, *, fill_value = jnp.nan):
    idr = jnp.clip(
        jnp.floor(
            ((query_points / jnp.asarray(lengths)) + 0.5) * (jnp.asarray(dims) - 1)
        ).astype(jnp.int32),
        0, jnp.asarray(dims) - 2
    )

    wx = (query_points[:, 0] - x[idr[:, 0]]) / (x[idr[:, 0] + 1] - x[idr[:, 0]])
    wy = (query_points[:, 1] - y[idr[:, 1]]) / (y[idr[:, 1] + 1] - y[idr[:, 1]])
    wz = (query_points[:, 2] - z[idr[:, 2]]) / (z[idr[:, 2] + 1] - z[idr[:, 2]])

    return (
        values[idr[:, 0], idr[:, 1], idr[:, 2]] * (1 - wx) * (1 - wy) * (1 - wz) +
        values[idr[:, 0], idr[:, 1], idr[:, 2] + 1] * (1 - wx) * (1 - wy) * wz       +
        values[idr[:, 0], idr[:, 1] + 1, idr[:, 2]] * (1 - wx) * wy       * (1 - wz) +
        values[idr[:, 0], idr[:, 1] + 1, idr[:, 2] + 1] * (1 - wx) * wy       * wz       +
        values[idr[:, 0] + 1, idr[:, 1], idr[:, 2]] * wx       * (1 - wy) * (1 - wz) +
        values[idr[:, 0] + 1, idr[:, 1], idr[:, 2] + 1] * wx       * (1 - wy) * wz       +
        values[idr[:, 0] + 1, idr[:, 1] + 1, idr[:, 2]] * wx       * wy       * (1 - wz) +
        values[idr[:, 0] + 1, idr[:, 1] + 1, idr[:, 2] + 1] * wx       * wy       * wz
    )

def dndr(r, ne, omega, x, y, z, lengths, dims):
    grad = jnp.zeros_like(r.T)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndx, r, fill_value = 0.0))
    del dndx

    dndy = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndy, r, fill_value = 0.0))
    del dndy

    dndz = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, inv_brems, phaseshift, B_on, ne, B, Te, Z, x, y, z, omega, VerdetConst, lengths, dims):
    s = jnp.reshape(s, (9, 1))

    sprime = jnp.zeros_like(s)

    r = s[:3, :].T
    v = s[3:6, :]

    amp = s[6, :]

    del s

    sprime = sprime.at[3:6, :].set(dndr(r, ne, omega, x, y, z, lengths, dims))
    sprime = sprime.at[:3, :].set(v)

    del r
    del v
    del amp

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, ne_extent, *, probing_direction = 'z', keep_current_plane = False, return_E = False):
    Np = rays.shape[1] # number of photons

    ray_p = jnp.zeros((4, Np))

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    t_bp = (z - ne_extent) / vz

    ray_p = ray_p.at[0].set(x - vx * t_bp)
    ray_p = ray_p.at[2].set(y - vy * t_bp)

    ray_p = ray_p.at[1].set(jnp.arctan(vx / vz))
    ray_p = ray_p.at[3].set(jnp.arctan(vy / vz))
    
    del x
    del y
    del z
    del vx
    del vy
    del vz

    del Np

    return ray_p

def solve(s0_import, ScalarDomain, dims, probing_depth, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False):
    Np = s0_import.shape[1]

    print("\nSize in memory of initial rays:", mem_conversion(getsizeof_default(s0_import) * Np))

    print(" --> tracing to depth of", probing_depth, "mm's")

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    args = (parallelise, ScalarDomain.inv_brems, ScalarDomain.phaseshift, ScalarDomain.B_on, ScalarDomain.ne, ScalarDomain.B, ScalarDomain.Te, ScalarDomain.Z, ScalarDomain.x, ScalarDomain.y, ScalarDomain.z, 2 * jnp.pi * c / lwl, 0.0, ScalarDomain.lengths, ScalarDomain.dims)

    available_devices = jax.devices()

    running_device = jax.lib.xla_bridge.get_backend().platform
    print("\nRunning device:", running_device, end='')

    core_count = int(os.environ['XLA_FLAGS'].replace("--xla_force_host_platform_device_count=", ''))
    print(", with:", core_count, "cores.")

    from jax.sharding import PartitionSpec as P, NamedSharding

    mesh = jax.make_mesh((core_count,), ('rows',))

    Np = ((Np // core_count) * core_count)
    assert Np > 0, "Not enough rays to parallelise over cores, increase to at least " + str(core_count)

    s0 = jax.device_put(s0_import.T[0:Np, :], NamedSharding(mesh, P('rows', None)))
    del s0_import

    print(s0.sharding)

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, *args) * norm_factor

    from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

    def diffrax_solve(dydt, t0, t1, Nt, *, rtol = 1e-7, atol = 1e-9):
        term = ODETerm(dydt)
        solver = Tsit5()
        saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
        stepsize_controller = PIDController(rtol, atol)

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
            max_steps = 10000
        )

    ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, save_steps)

    start_comp = time()

    from equinox import filter_jit
    ODE_solve = filter_jit(ODE_solve)

    print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

    start = time()

    sol = jax.block_until_ready(
        jax.vmap(ODE_solve, in_axes = (0, None))(s0, args)
    )

    duration = time() - start
    print("\nCompleted ray trace in", jnp.round(duration, 3), "seconds.")

    del s0

    return ray_to_Jonesvector(sol.ys[:, -1, :].T, probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E), duration