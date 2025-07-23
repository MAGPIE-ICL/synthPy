import numpy as np

import sys
import os

sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')

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

import jax

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
        self.B_on = False
        self.inv_brems = False
        self.phaseshift = False

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

        self.B = None
        self.Te = None
        self.Z = None

        self.probing_direction = 'z'

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

def trilinearInterpolator(x, y, z, values, query_points, *, fill_value = jnp.nan):
    """
    Trilinear interpolation on a 3D regular grid.

    Assumes:
        - coordinates = (x, y, z) where each is 1D
        - values.shape == (len(x), len(y), len(z))
        - query_points.shape == (N, 3)
    """

    values = jnp.asarray(values)
    query_points = jnp.asarray(query_points.T)

    def get_indices_and_weights(coord_grid, points):
        idx = jnp.searchsorted(coord_grid, points, side = 'right') - 1
        idx = jnp.clip(idx, 0, len(coord_grid) - 2)

        x0 = coord_grid[idx]
        return idx, (points - x0) / (coord_grid[idx + 1] - x0)

    ix, wx = get_indices_and_weights(x, query_points[:, 0])
    iy, wy = get_indices_and_weights(y, query_points[:, 1])
    iz, wz = get_indices_and_weights(z, query_points[:, 2])

    def get_val(dx, dy, dz):
        return values[ix + dx, iy + dy, iz + dz]

    results = (
        get_val(0, 0, 0) * (1 - wx) * (1 - wy) * (1 - wz) +
        get_val(1, 0, 0) * wx       * (1 - wy) * (1 - wz) +
        get_val(0, 1, 0) * (1 - wx) * wy       * (1 - wz) +
        get_val(0, 0, 1) * (1 - wx) * (1 - wy) * wz +
        get_val(1, 1, 0) * wx       * wy       * (1 - wz) +
        get_val(1, 0, 1) * wx       * (1 - wy) * wz +
        get_val(0, 1, 1) * (1 - wx) * wy       * wz +
        get_val(1, 1, 1) * wx       * wy       * wz
    )

    # Check out-of-bounds
    oob = (
        (query_points[:, 0] < x[0]) | (query_points[:, 0] > x[-1]) |
        (query_points[:, 1] < y[0]) | (query_points[:, 1] > y[-1]) |
        (query_points[:, 2] < z[0]) | (query_points[:, 2] > z[-1])
    )

    return jnp.where(oob, fill_value, results)

##
## Helper functions for calculations
##

def omega_pe(ne):
    """Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28"""

    return 5.64e4 * jnp.sqrt(ne)

# NRL formulary inverse brems - cheers Jack Halliday for coding in Python
# Converted to rate coefficient by multiplying by group velocity in plasma
def kappa(ne, Te, Z, omega):
    # Useful subroutines
    def v_the(Te):
        """Calculate electron thermal speed. Provide Te in eV. Retrurns result in m/s"""

        return 4.19e5 * jnp.sqrt(Te)

    def V(ne, Te, Z, omega):
        o_pe = omega_pe(ne)
        #o_max = jnp.copy(o_pe)
        #o_max[o_pe < omega] = omega
        o_pe = o_pe.at[:, :].set(jnp.where(o_pe < omega, omega, o_pe))
        L_classical = Z * e / Te
        L_quantum = 2.760428269727312e-10 / jnp.sqrt(Te) # hbar / jnp.sqrt(m_e * e * Te)
        L_max = jnp.maximum(L_classical, L_quantum)

        #return o_max * L_max
        return o_pe * L_max

    def coloumbLog(ne, Te, Z, omega):
        return jnp.maximum(2.0, jnp.log(v_the(Te) / V(ne, Te, Z, omega)))

    ne_cc = ne * 1e-6
    # don't think this is actually used?
    #o_pe = omega_pe(ne_cc)
    CL = coloumbLog(ne_cc, Te, Z, omega)

    result = 3.1e-5 * Z * c * jnp.power(ne_cc / omega, 2) * CL * jnp.power(Te, -1.5) # 1/s
    del ne_cc

    return result

# Plasma refractive index
def n_refrac(ne, omega):
    return jnp.sqrt(1.0 - (omega_pe(ne * 1e-6) / omega) ** 2)

def calc_dndr(ScalarDomain, lwl = 1064e-9, *, keep_domain = False):
    VerdetConst = 0.0
    if (ScalarDomain.B_on):
        VerdetConst = 2.62e-13 * lwl ** 2 # radians per Tesla per m^2

    omega = 2 * jnp.pi * c / lwl

    return (
        ScalarDomain.ne,
        ScalarDomain.B,
        ScalarDomain.Te,
        ScalarDomain.Z,
        omega,
        VerdetConst,
        ScalarDomain.inv_brems,
        ScalarDomain.phaseshift,
        ScalarDomain.B_on,
        ScalarDomain.probing_direction
    )

def dndr(r, ne, omega, x, y, z):
    grad = jnp.zeros_like(r)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator(x, y, z, dndx, r, fill_value = 0.0))
    del dndx

    dndy = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator(x, y, z, dndy, r, fill_value = 0.0))
    del dndy

    dndz = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator(x, y, z, dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, inv_brems, phaseshift, B_on, ne, B, Te, Z, x, y, z, omega, VerdetConst):
    if not parallelise:
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (9, s.size // 9))
    else:
        # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
        s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :]
    v = s[3:6, :]

    # Amplitude, phase and polarisation
    amp = s[6, :]
    #phase = s[7,:]
    #pol = s[8,:]

    # was deleting before it needed using before by accident - obviously caused issues (AbstractTerm error)
    # - fine to delete after used, only one slice of s0 rather than deleting s0
    # although probably really unnecessary?
    del s

    # must unpack coordinates tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, ne, omega, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    # Attenuation due to inverse bremsstrahlung
    if inv_brems:
        sprime = sprime.at[6, :].set(trilinearInterpolator(x, y, z, kappa(ne, Te, Z, omega), r) * amp)
    if phaseshift:
        sprime = sprime.at[7, :].set(omega * (trilinearInterpolator(x, y, z, n_refrac(ne, omega), r) - 1.0))
    if B_on:
        """
        Returns the VerdetConst ne B.v

        Args:
            x (3xN float): N [x,y,z] locations
            v (3xN float): N [vx,vy,vz] velocities

        Returns:
            N float: N values of ne B.v
        """

        ne_N = trilinearInterpolator(x, y, z, ne, r)

        Bv_N = jnp.sum(
            jnp.array(
                [
                    trilinearInterpolator(x, y, z, B[:, :, :, 0], r),
                    trilinearInterpolator(x, y, z, B[:, :, :, 1], r),
                    trilinearInterpolator(x, y, z, B[:, :, :, 2], r)
                ]
            ) * v, axis = 0
        )

        sprime = sprime.at[8, :].set(VerdetConst * ne_N * Bv_N)

    del r
    del v
    del amp

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, ne_extent, *, probing_direction = 'z', keep_current_plane = False, return_E = False):
    Np = rays.shape[1] # number of photons

    ray_p = jnp.zeros((4, Np))
    ray_J = jnp.zeros((2, Np), dtype = complex)

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x - ne_extent) / vx

        # Positions on plane
        if not keep_current_plane:
            ray_p = ray_p.at[0].set(y - vy * t_bp)
            ray_p = ray_p.at[2].set(z - vz * t_bp)
        else:
            ray_p = ray_p.at[0].set(y)
            ray_p = ray_p.at[2].set(z)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vy / vx))
        ray_p = ray_p.at[3].set(jnp.arctan(vz / vx))
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y - ne_extent) / vy

        if not keep_current_plane:
            ray_p = ray_p.at[0].set(z - vz * t_bp)
            ray_p = ray_p.at[2].set(x - vx * t_bp)
        else:
            ray_p = ray_p.at[0].set(z)
            ray_p = ray_p.at[2].set(x)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vz / vy))
        ray_p = ray_p.at[3].set(jnp.arctan(vx / vy))
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z - ne_extent) / vz

        # Positions on plane
        if not keep_current_plane:
            ray_p = ray_p.at[0].set(x - vx * t_bp)
            ray_p = ray_p.at[2].set(y - vy * t_bp)
        else:
            ray_p = ray_p.at[0].set(x)
            ray_p = ray_p.at[2].set(y)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vx / vz))
        ray_p = ray_p.at[3].set(jnp.arctan(vy / vz))
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")
    
    del x
    del y
    del z
    del vx
    del vy
    del vz

    if return_E:
        # Resolve Jones vectors
        amp, phase, pol = rays[6], rays[7], rays[8]

        # Assume initially polarised along y
        E_x_init = jnp.zeros(Np)
        E_y_init = jnp.ones(Np)

        # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
        ray_J = ray_J.at[0].set(amp * (jnp.cos(phase) + 1.0j * jnp.sin(phase)) * (jnp.cos(pol) * E_x_init - jnp.sin(pol) * E_y_init))
        ray_J = ray_J.at[1].set(amp * (jnp.cos(phase) + 1.0j * jnp.sin(phase)) * (jnp.sin(pol) * E_x_init + jnp.cos(pol) * E_y_init))

        del amp
        del phase
        del pol

        del E_x_init
        del E_y_init

    del Np

    # ray_p [x, phi, y, theta], ray_J [E_x, E_y]
    if return_E:
        return ray_p, ray_J

    return ray_p, None

def solve(s0_import, coordinates, dim, probing_depth, ne, B, Te, Z, omega, VerdetConst, inv_brems, phaseshift, B_on, probing_direction, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False):
    Np = s0_import.shape[1]

    print("\nSize in memory of initial rays:", mem_conversion(getsizeof_default(s0_import) * Np))

    # if batched: or if auto_batching: etc.
    # proing_depth /= some integer with some corrections I expect
    # make logic too loop it and pick up from previous solution

    # Need to make sure all rays have left volume
    # Conservative estimate of diagonal across volume
    # Then can backproject to surface of volume

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
    # think we should change this???

    ##
    ## currently NOT passing interps
    ## - get AbstractTerm either when interps are passed, both as dictionary or as an equinox class
    ## - try to fix later, pass individually perhaps?
    ## NOTES SAY DICT IS NOT HASHABLE - try as a tuple? - tuple did not work either :(
    ##

    # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not
    args = (parallelise, inv_brems, phaseshift, B_on, ne, B, Te, Z, *coordinates, omega, VerdetConst)

    if not parallelise:
        from numpy import array
        s0 = array(jnp.ravel(s0_import))
        #s0 = s0.flatten() #odeint insists

        start = time()
        # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
        sol = solve_ivp(lambda t, y: dsdt(t, y, *args), [0, t[-1]], s0, t_eval = t)
    else:
        available_devices = jax.devices()

        '''
        if force_device is not None:
            try:
                #jax.default_device = jax.devices(force_device)[0]
                jax.config.update('jax_platform_name', force_device)
            except:
                print("\njax cannot detect that device if it does exist - try not passing a force_device param and seeing if it runs.")
        '''

        running_device = jax.lib.xla_bridge.get_backend().platform # - deprecated, using still as needed for HPC
        #running_device = jax.extend.backend.get_backend().platform
        print("\nRunning device:", running_device, end='')

        # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
        s0_transformed = s0_import.T
        del s0_import

        if running_device == 'cpu':
            #from multiprocessing import cpu_count
            #core_count = cpu_count()

            core_count = int(os.environ['XLA_FLAGS'].replace("--xla_force_host_platform_device_count=", ''))
            print(", with:", core_count, "cores.")

            from jax.sharding import PartitionSpec as P, NamedSharding

            # Create a Sharding object to distribute a value across devices:
            # Assume self.core_count is the no. of core devices available
            mesh = jax.make_mesh((core_count,), ('rows',))  # 1D mesh for columns

            # Specify sharding: don't split axis 0 (rows), split axis 1 (columns) across devices
            # then apply sharding to rewrite s0 as a sharded array from it's original matrix
            # and use jax.device_put to distribute it across devices:
            Np = ((Np // core_count) * core_count)
            assert Np > 0, "Not enough rays to parallelise over cores, increase to at least " + str(core_count)

            # if you don't wish to transpose before operation you need to use the old call
            # s0 = jax.device_put(s0_transformed[:, 0:Np], NamedSharding(mesh, P(None, 'cols')))
            s0 = jax.device_put(s0_transformed[0:Np, :], NamedSharding(mesh, P('rows', None)))  # 'None' means don't shard axis 0

            print(s0.sharding)            # See the sharding spec
            #print(s0.addressable_shards)  # Check each device's shard
            #jax.debug.visualize_array_sharding(s0)
        elif running_device == 'gpu':
            gpu_devices = jax.devices('gpu')
            print("\nThere are", len(gpu_devices), "available GPU devices:", gpu_devices)
            assert len(gpu_devices) > 0, "Running on GPU yet none detected?"

            s0 = jax.device_put(s0_transformed, gpu_devices[0])
        elif running_device == 'tpu':
            s0 = s0_transformed
            pass
        else:
            assert "No suitable device detected!"

        del s0_transformed
        # optional for aggressive cleanup?
        #jax.clear_caches()

        # wrapper for same reason, diffrax.ODETerm instantiaties this and passes args
        # I have no idea why, but this has to be defined in solve rather than as a global function - else there is an abstract variable error
        def dsdt_ODE(t, y, args):
            return dsdt(t, y, *args) * norm_factor

        from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve
        #import optax - diffrax uses as a dependency, don't need to import directly

        def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
            """
            Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
            """

            # We convert our python function to a diffrax ODETerm
            # should use the function passed into the wrapper - not the local definition
            term = ODETerm(dydt)
            # We chose a solver (time-stepping) method from within diffrax library
            solver = Tsit5() # (RK45 - closest I could find to solve_ivp's default method)

            # At what time points you want to save the solution
            saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
            # Diffrax uses adaptive time stepping to gain accuracy within certain tolerances
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
                # set max steps to no. of cells x100
                max_steps = dim[0] * dim[1] * dim[2] * 100 #10000 - default for solve_ivp?????
            )

        # hardcode to normalise to 1 due to diffrax bug
        ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, save_steps)

        if jitted:
            start_comp = time()

            from equinox import filter_jit
            # equinox.filter_jit() (imported as filter_jit()) provides debugging info unlike jax.jit() - it does not like static args though so sticking with jit for now
            #ODE_solve = jax.jit(ODE_solve)#, static_argnums = 1)#, device = available_devices[0])
            ODE_solve = filter_jit(ODE_solve)#, device = available_devices[0])
            # not sure about the performance of non-static specified arguments with filter_jit() - only use for debugging not in 'production'

            print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

        from functools import partial

        # pass s0[:, i] for each ray via a jax.vmap for parallelisation
        start = time()
        sol = jax.block_until_ready(
            # in_axes version ensures that vmap doesn't map args parameters, just s0
            #jax.vmap(lambda rays, args: ODE_solve, in_axes = (0, None))(s0, args)
            # default vmap_method argument is sequential, this is deprecated though and will cause a warning (if debugging) past jax 0.6.0
            # look into different options for this parameter at a later date
            #jax.vmap(partial(lambda s: ODE_solve(s, args), vmap_method = "sequential"))(s0)
            #jax.vmap(partial(ODE_solve, in_axes = (0, None), vmap_method = "sequential"))(s0, args)
            jax.vmap(ODE_solve, in_axes = (0, None))(s0, args)
        )

        #sol = jax.block_until_ready(jax.vmap(ODE_solve, in_axes = (0, None))(s0, args))

    duration = time() - start

    #del ne_nc

    if memory_debug:
        if parallelise:
            # Visualises sharding, looks cool, but pretty useless - and a pain with higher core counts
            jax.debug.visualize_array_sharding(sol.ys[:, -1, :])

        from utils import domain_estimate

        print(colour.BOLD + "\nMemory summary - total estimate:", mem_conversion(domain_estimate(dim) + (getsizeof_default(s0) + getsizeof_default(sol)) * Np))
        print("\nSize in memory of initial rays:", mem_conversion(getsizeof_default(s0) * Np))
        print("Size in memory of solution class / single ray (?):", getsizeof(sol))
        print("Size in memory of solution:", mem_conversion(getsizeof_default(sol) * Np))

    del s0

    if memory_debug:
        folder_name = "memory_benchmarks/"
        rel_path_to_folder = "../../evaluation/"

        path = rel_path_to_folder + folder_name

        if os.path.isdir(os.getcwd() + "/" + folder_name):
            path = folder_name
        elif os.path.isdir(os.getcwd() + "/" + path):
            pass
            '''
            elif not os.path.isdir(os.getcwd() + "/" + path):
                import errno

                try:
                    os.mkdir(path)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            '''
        else:
            path = os.getcwd() + "/" + folder_name

            try:
                os.mkdir(path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        from datetime import datetime
        path += "memory-domain" + str(dim[0]) + "_rays"+ str(s0.shape[1]) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
        jax.profiler.save_device_memory_profile(path)

        print("\n", end = '')
        if os.path.isfile(os.path.expanduser("~") + "/go/bin/pprof"):
            #import sys
            from os import system as os_system

            #os_system(f"~/go/bin/pprof -top {sys.executable} memory_{N}.prof")
            os_system(f"~/go/bin/pprof -top /bin/ls " + path)
            #os_system(f"~/go/bin/pprof --web " + path)
        else:
            print("No pprof install detected. Please download (using go) to visualise memory usage.")

    if not parallelise:
        rf = sol.y[:,-1].reshape(9, Np)
    else:
        """
        #for i in enumerate(sol.result):
        #    print(i)
        for idx, result in enumerate(sol.result):
            # Check if each result is successful
            if result.success:
                print(f"Solution at index {idx} succeeded.")
            else:
                print(f"Solution at index {idx} failed.")
    
        #print(next(sol.result))
        #print(next(sol.result))
        #print(type(sol.result[0]))  # Check the type of results
        """

        #if sol.result == RESULTS.successful:
        #rf = sol.ys[:, -1, :].reshape(9, Np)# / scalar
        rf = sol.ys[:, -1, :].T

        print("\n\nParallelised output has resulting 3D matrix of form: [batch_count, 2, 9]:", sol.ys.shape)
        print("\t2 to account for the start and end results")
        print("\t9 containing the 3 position and velocity components, amplitude, phase and polarisation")
        print("\tIf batch_count is lower than expected, this is likely due to jax's forced integer batch sharding when parallelising over cpu cores.")
        print("\nWe slice the end result and transpose into the form:", rf.shape, "to work with later code.")
        #else:
        #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

    return *ray_to_Jonesvector(rf, probing_depth, probing_direction = probing_direction, return_E = return_E), duration

rf, Jf, duration = solve(
    beam_definition,
    (domain.x, domain.y, domain.z),
    (domain.x_n, domain.y_n, domain.z_n),   # domain.dim - this causes a TracerBoolConversionError, check why later, could be interesting and useful to know
    ne_extent,
    *p.calc_dndr(domain, lwl, keep_domain = True)
)

print("\nRun complete!")