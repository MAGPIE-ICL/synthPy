import matplotlib.pyplot as plt
import os
import sys
import jax
import jax.numpy as jnp

from scipy.integrate import odeint, solve_ivp
from time import time
from datetime import datetime
from os import system as os_system
from jax.scipy.interpolate import RegularGridInterpolator

from scipy.constants import c
from scipy.constants import e
#from scipy.constants import hbar
#from scipy.constants import m_e

import interpolations

from utils import getsizeof
#from utils import trilinearInterpolator

# Attenuation due to inverse bremsstrahlung
def atten(kappa_interp, inv_brems, x):
    if(inv_brems):
        return kappa_interp(x.T)
    else:
        return 0.0

# Phase shift introduced by refractive index
def phase(refractive_index_interp, phaseshift, x, omega):
    if(phaseshift):
        return omega * (refractive_index_interp(x.T) - 1.0)
    else:
        return 0.0

def neB(ne_interp, Bx_interp, By_interp, Bz_interp, B_on, x, v, VerdetConst):
    """
    Returns the VerdetConst ne B.v

    Args:
        x (3xN float): N [x,y,z] locations
        v (3xN float): N [vx,vy,vz] velocities

    Returns:
        N float: N values of ne B.v
    """

    def get_ne(ne_interp, x):
        return ne_interp(x.T)

    def get_B(Bx_interp, By_interp, Bz_interp, x):
        return jnp.array([Bx_interp(x.T), By_interp(x.T), Bz_interp(x.T)])

    if (B_on):
        ne_N = get_ne(ne_interp, x)
        Bv_N = jnp.sum(get_B(Bx_interp, By_interp, Bz_interp, x) * v, axis = 0)
        pol = VerdetConst * ne_N * Bv_N
    else:
        pol = 0.0

    return pol

def calc_dndr(ScalarDomain, lwl = 1064e-9, *, keep_domain = False):
    """
    Generate interpolators for derivatives.

    Args:
        lwl (float, optional): laser wavelength. Defaults to 1064e-9 m.
    """

    # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
    VerdetConst = 0.0
    if (ScalarDomain.B_on):
        VerdetConst = 2.62e-13 * lwl ** 2 # radians per Tesla per m^2

    omega = 2 * jnp.pi * c / lwl
    nc = 3.14207787e-4 * omega ** 2

    ne_nc = jnp.array(ScalarDomain.ne / nc, dtype = jnp.float32)

    interps = interpolations.set_up_interps(ScalarDomain, omega)

    if not keep_domain:
        try:
            del ScalarDomain.ne
        except:
            ScalarDomain.ne = None

    if not keep_domain:
        try:
            del ScalarDomain.B
        except:
            ScalarDomain.B = None

    return (
        interps,
        ne_nc,
        omega,
        VerdetConst,
        ScalarDomain.inv_brems,
        ScalarDomain.phaseshift,
        ScalarDomain.B_on,
        ScalarDomain.probing_direction
    )

def dndr(r, ne_nc, coordinates):
    """
    Returns the gradient at the locations r

    Args:
        r (3xN float): N [x, y, z] locations

    Returns:
        3 x N float: N [dx, dy, dz] electron density gradients
    """

    grad = jnp.zeros_like(r)

    #More compact notation is possible here, but we are explicit
    dndx = -0.5 * c ** 2 * jnp.gradient(ne_nc, x, axis = 0)
    dndx_interp = RegularGridInterpolator(coordinates, dndx, bounds_error = False, fill_value = 0.0)
    del dndx

    grad = grad.at[0, :].set(dndx_interp(r.T))
    del dndx_interp

    dndy = -0.5 * c ** 2 * jnp.gradient(ne_nc, y, axis = 1)
    dndy_interp = RegularGridInterpolator(coordinates, dndy, bounds_error = False, fill_value = 0.0)
    del dndy

    grad = grad.at[1, :].set(dndy_interp(r.T))
    del dndy_interp

    dndz = -0.5 * c ** 2 * jnp.gradient(ne_nc, z, axis = 1)
    dndz_interp = RegularGridInterpolator(coordinates, dndz, bounds_error = False, fill_value = 0.0)
    del dndz

    grad = grad.at[2, :].set(dndz_interp(r.T))
    del dndz_interp

    # reassinging the same variable (so in theory same memory addresses) instead of using x, y, z seperate variables and deleting
    # is less memory efficient according to benchmarking - would it decrease the likelihood of memory leaks though?

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, interps, parallelise, inv_brems, phaseshift, B_on, ne_nc, coordinates, omega, VerdetConst):
    """
    Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (9N float array): flattened 9xN array of rays used by ode_int
        ScalarDomain (ScalarDomain): an ScalarDomain object which can calculate gradients

    Returns:
        9N float array: flattened array for ode_int
    """

    if not parallelise:
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (9, s.size // 9))
    else:
        # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
        s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised
    
    # unsure as jax array - also passed not created, should be a copy anyway no?
    #del s

    #sprime = jnp.zeros_like(s.reshape(9, s.size // 9))
    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :]
    v = s[3:6, :]

    # Amplitude, phase and polarisation
    amp = s[6, :]
    #phase = s[7,:]
    #pol = s[8,:]

    sprime = sprime.at[3:6, :].set(dndr(r, ne_nc, *coordinates))
    sprime = sprime.at[:3, :].set(v)

    sprime = sprime.at[6, :].set(atten(interps['kappa_interp'], inv_brems, r) * amp)
    sprime = sprime.at[7, :].set(phase(interps['refractive_index_interp'], phaseshift, r, omega))
    sprime = sprime.at[8, :].set(
        neB(
            interps['ne_interp'],
            interps['Bx_interp'],
            interps['By_interp'],
            interps['Bz_interp'],
            B_on, r, v, VerdetConst
        )
    )

    del r
    del v
    del a

    return sprime#.flatten()

# wrapper for same reason, diffrax.ODETerm instantiaties this and passes args (this will contain self)
# diffrax/jax prefers top level functions for tracing purposes
def dsdt_ODE(t, y, args):
    return dsdt(t, y, *args) * norm_factor

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, ne_extent, *, probing_direction = 'z', keep_current_plane = False, return_E = False):
    # * forces keep_current_plane and return_E to be keyword-only arguments
    # meaning .. return_E = True (missing out keep_current_plane) will work as it will not rely on position
    """
    Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

    Gives position (and angles) in other axes at point where ray is in end plane of its extent in the probing axis
    (if keep_current_plane is set to True, it does not return the rays to the end of volume - just returns current 2D slice position)

    Args:
        rays (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of shape (cuboid) in probing direction, m
        probing_direction (str): x, y or z.
        keep_current_plane (boolean): flag to enable compatability (via True) with use in diagnostics.py, defaults to False

    Returns:
        [type]: [description]
    """

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

        #
        # I have switched x & z for the sake of consistent ordering of the axes
        # Standardised in keeping with positive 'forward' notation, etc. x * y = z but don't do y * x = -z
        # If memory is not a concern then will instead create a class to cover directions
        # This would entail both the array and a self.dir parameter of type char - containing 'x', 'y' or 'z'
        #

        # Positions on plane
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
        return jnp.array(ray_p), jnp.array(ray_J)

    return jnp.array(ray_p), None

def solve(s0_import, extent, r_n, coordinates, interps, ne_nc, omega, VerdetConst, inv_brems, phaseshift, B_on, probing_direction, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False):
    Np = s0_import.shape[1]

    print("\nSize in memory of initial rays:", getsizeof(s0_import))

    # Need to make sure all rays have left volume
    # Conservative estimate of diagonal across volume
    # Then can backproject to surface of volume

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * extent / c, 2)

    # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
    # think we should change this???

    if not parallelise:
        from numpy import array
        s0 = array(jnp.ravel(s0_import))
        #s0 = s0.flatten() #odeint insists

        start = time()
        # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
        sol = solve_ivp(lambda t, y: dsdt(t, y, interps, parallelise), [0, t[-1]], s0, t_eval = t)
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

        from jax.lib import xla_bridge
        running_device = xla_bridge.get_backend().platform
        print("\nRunning device:", running_device, end='')

        # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
        s0_transformed = s0_import.T
        del s0_import

        if running_device == 'cpu':
            from multiprocessing import cpu_count
            core_count = cpu_count()
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

        norm_factor = jnp.max(t)

        from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve
        #import optax - diffrax uses as a dependency, don't need to import directly

        def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
            """
            Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
            """

            # We convert our python function to a diffrax ODETerm
            term = ODETerm(dsdt_ODE)
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
                max_steps = 10000 #r_n[0] * r_n[1] * r_n[2] * 100 #10000 - default for solve_ivp?????
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

            print("\njax compilation of solver took:", time() - start_comp, "seconds")

        # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not
        args = (interps, parallelise, inv_brems, phaseshift, B_on, ne_nc, coordinates, omega, VerdetConst)

        # pass s0[:, i] for each ray via a jax.vmap for parallelisation
        start = time()
        sol = jax.block_until_ready(
            jax.vmap(lambda rays: ODE_solve(rays, args))(s0)
        )

        #sol = jax.block_until_ready(jax.vmap(ODE_solve, in_axes = (0, None))(s0, args))
        #sol = jax.block_until_ready(jax.vmap(lambda s, args: ODE_solve(s, args), in_axes = (0, None))(s0, args))

    duration = time() - start

    del interps
    del ne_nc

    if memory_debug and parallelise:
        # Visualises sharding, looks cool, but pretty useless - and a pain with higher core counts
        #jax.debug.visualize_array_sharding(sol.ys[:, -1, :])

        print("\nSize in memory of initial rays:", getsizeof(s0))
        print("Size in memory of solution:", getsizeof(sol))
        print("Size in memory of propagator class:", getsizeof(sol))

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

        path += "memory-domain" + str(ScalarDomain.dim[0]) + "_rays"+ str(s0.shape[1]) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
        jax.profiler.save_device_memory_profile(path)

        print("\n", end = '')
        if os.path.isfile(os.path.expanduser("~") + "/go/bin/pprof"):
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

        print("\nParallelised output has resulting 3D matrix of form: [batch_count, 2, 9]:", sol.ys.shape)
        print("\t2 to account for the start and end results")
        print("\t9 containing the 3 position and velocity components, amplitude, phase and polarisation")
        print("\tIf batch_count is lower than expected, this is likely due to jax's forced integer batch sharding when parallelising over cpu cores.")
        print("\nWe slice the end result and transpose into the form:", rf.shape, "to work with later code.")
        #else:
        #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

    return ray_to_Jonesvector(rf, extent, probing_direction = probing_direction, return_E = return_E), duration

# need to remove this, replacing main solve function with this option as a flag for part solves in reduced domains
# (or just if someone wanted that for some reason - doesn't have to be the intended one of course, that's the point of generalisation)
def solve_at_depth(self, s0_import, z):
    """
    Solve intial rays up until a given depth, z
    """

    # Need to make sure all rays have left volume
    # Conservative estimate of diagonal across volume
    # Then can backproject to surface of volume

    length = self.extent + z
    t = jnp.linspace(0.0, length / c, 2)

    s0 = s0_import.flatten() #odeint insists
    del s0_import

    print("\nStarting ray trace.")

    start = time()

    parallelise = False
    dsdt_ODE = lambda t, y: dsdt(t, y, self, parallelise)
    sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t)

    finish = time()
    self.duration = finish - start

    print("\nRay trace completed in:\t", self.duration, "s")

    self.rf, _ = ray_to_Jonesvector(sol.y[:,-1].reshape(9, s0.size // 9), self.extent, probing_direction = self.probing_direction)