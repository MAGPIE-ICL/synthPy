import numpy as np
import matplotlib.pyplot as plt

import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
args = parser.parse_args()

domain = 128
if args.domain is not None:
    domain = args.domain

rays = 10000
if args.rays is not None:
    domain = args.rays

from multiprocessing import cpu_count

assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(cpu_count())
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

n_cells = 128

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

import jax.numpy as jnp
import equinox as eqx

from scipy.constants import c
from scipy.constants import e

class ScalarDomain(eqx.Module):
    """
    A class to hold and generate scalar domains.
    This contains also the method to propagate rays through the scalar domain
    """

    inv_brems: bool
    phaseshift: bool
    B_on: bool

    probing_direction: str

    x_length: jnp.int64
    y_length: jnp.int64
    z_length: jnp.int64

    lengths: jax.Array

    x_n: jnp.int64
    y_n: jnp.int64
    z_n: jnp.int64

    dim: jax.Array

    x: jax.Array
    y: jax.Array
    z: jax.Array

    XX: jax.Array
    YY: jax.Array
    ZZ: jax.Array

    ne: jax.Array

    B: jax.Array
    Te: jax.Array
    Z: jax.Array

    def __init__(self, lengths, dim, *, ne_type = None, inv_brems = False, phaseshift = False, B_on = False, probing_direction = 'z'):
        self.ne = None
        self.B = None
        self.Te = None
        self.Z = None

        # Logical switches
        self.inv_brems = inv_brems
        self.phaseshift = phaseshift
        self.B_on = B_on

        self.probing_direction = probing_direction

        valid_types = (int, float, jnp.int64)

        if isinstance(lengths, valid_types):
            self.x_length, self.y_length, self.z_length = lengths, lengths, lengths
            self.lengths = jnp.array([lengths, lengths, lengths])
        else:
            if len(lengths) != 3:
                raise Exception('lengths must have len = 3: (x,y,z)')

            self.x_length, self.y_length, self.z_length = lengths[0], lengths[1], lengths[2]
            self.lengths = jnp.array(lengths)

        if isinstance(dim, valid_types):
            self.x_n, self.y_n, self.z_n = dim, dim, dim
            self.dim = jnp.array([dim, dim, dim])
        else:
            if len(dim) != 3:
                raise Exception('n must have len = 3: (x_n, y_n, z_n)')

            self.x_n, self.y_n, self.z_n = dim[0], dim[1], dim[2]
            self.dim = jnp.array(dim)

        self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
        self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
        self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))

        self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)#False) - has to be true for jnp
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

domain = ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos")

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

from jax.scipy.interpolate import RegularGridInterpolator

def calc_dndr(ne, lwl = 1064e-9):
    omega = 2 * jnp.pi * c / lwl
    nc = 3.14207787e-4 * omega ** 2

    return jnp.array(ne / nc, dtype = jnp.float32)

def dndr(r, ne_nc, x, y, z):
    grad = jnp.zeros_like(r)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne_nc, x, axis = 0)
    dndx_interp = RegularGridInterpolator((x, y, z), dndx, bounds_error = False, fill_value = 0.0)
    del dndx

    grad = grad.at[0, :].set(dndx_interp(r.T))
    del dndx_interp

    dndy = -0.5 * c ** 2 * jnp.gradient(ne_nc, y, axis = 1)
    dndy_interp = RegularGridInterpolator((x, y, z), dndy, bounds_error = False, fill_value = 0.0)
    del dndy

    grad = grad.at[1, :].set(dndy_interp(r.T))
    del dndy_interp

    dndz = -0.5 * c ** 2 * jnp.gradient(ne_nc, z, axis = 2)
    dndz_interp = RegularGridInterpolator((x, y, z), dndz, bounds_error = False, fill_value = 0.0)
    del dndz

    grad = grad.at[2, :].set(dndz_interp(r.T))
    del dndz_interp

    return grad

def dsdt(t, s, ne_nc, x, y, z):
    s = jnp.reshape(s, (9, 1))
    sprime = jnp.zeros_like(s)

    r = s[:3, :]
    v = s[3:6, :]

    a = s[6, :]

    sprime = sprime.at[3:6, :].set(dndr(r, ne_nc, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

def solve(s0_import, ne_nc, x, y, z, x_n, y_n, z_n, extent):
    Np = s0_import.shape[1]
    s0 = s0_import.T
    del s0_import

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * extent / c, 2)
    norm_factor = jnp.max(t)

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, *args) * norm_factor

    import diffrax

    def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
        term = diffrax.ODETerm(dydt)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
        stepsize_controller = diffrax.PIDController(rtol = rtol, atol = atol)

        return lambda s0, args : diffrax.diffeqsolve(
            term,
            solver,
            y0 = jnp.array(s0),
            args = args,
            t0 = t0,
            t1 = t1,
            dt0 = (t1 - t0) * norm_factor / Nt,
            saveat = saveat,
            stepsize_controller = stepsize_controller,
            # set max steps to no. of cells x100
            max_steps = x_n * y_n * z_n * 100 #10000 - default for solve_ivp?????
        )

    from equinox import filter_jit
    ODE_solve = filter_jit(diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, 2))

    args = (ne_nc, x, y, z)
    sol = jax.block_until_ready(jax.vmap(lambda s: ODE_solve(s, args))(s0))

    del ne_nc

    return sol.ys[:, -1, :].T

ne_nc = calc_dndr(domain.ne, lwl)

jax.print_environment_info()

rf = solve(beam_definition, ne_nc, domain.x, domain.y, domain.z, domain.x_n, domain.y_n, domain.z_n, ne_extent)