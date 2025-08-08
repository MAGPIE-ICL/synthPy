import jax
import jax.numpy as jnp
import numpy as np

import os

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

from scipy.constants import c
from scipy.constants import e

from shared.utils import getsizeof
from shared.utils import mem_conversion
from shared.printing import colour
from shared.utils import add_integer_postfix

# change name when it actualy is a trilinear interpolator - if it's still a regular grid, change it.
from simulator.interpolator import RegularGridInterpolator as trilinearInterpolator

from shared.propagation import ray_to_Jonesvector
from shared.propagation import back_propogate

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

def dndr(r, ne, omega, x, y, z):
    """
    Returns the gradient at the locations r

    Args:
        r (3xN float): N [x, y, z] locations

    Returns:
        3 x N float: N [dx, dy, dz] electron density gradients
    """

    grad = jnp.zeros_like(r.T)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), dndx, r, fill_value = 0.0))
    del dndx

    dndy = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), dndy, r, fill_value = 0.0))
    del dndy

    dndz = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, inv_brems, phaseshift, B_on, ne, B, Te, Z, x, y, z, omega, VerdetConst, lengths, dims):
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

    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :].T  # transposed so it is of the correct shape for interpolators
    v = s[3:6, :]

    # Amplitude, phase and polarisation
    amp = s[6, :]
    #phase = s[7,:]
    #pol = s[8,:]

    # was deleting before it needed using before by accident - obviously caused issues (AbstractTerm error)
    # - fine to delete after used, only one slice of s0 rather than deleting s0
    # although probably really unnecessary?
    del s

    # must unpack x, y, z tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, ne, omega, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    # Attenuation due to inverse bremsstrahlung
    if inv_brems:
        sprime = sprime.at[6, :].set(trilinearInterpolator((x, y, z), kappa(ne, Te, Z, omega), r) * amp)
    if phaseshift:
        sprime = sprime.at[7, :].set(omega * (trilinearInterpolator((x, y, z), n_refrac(ne, omega), r) - 1.0))

    if B_on:
        """
        Returns the VerdetConst ne B.v

        Args:
            x (3xN float): N [x,y,z] locations
            v (3xN float): N [vx,vy,vz] velocities

        Returns:
            N float: N values of ne B.v
        """

        ne_N = trilinearInterpolator((x, y, z), ne, r)

        Bv_N = jnp.sum(
            jnp.array(
                [
                    trilinearInterpolator((x, y, z), B[:, :, :, 0], r),
                    trilinearInterpolator((x, y, z), B[:, :, :, 1], r),
                    trilinearInterpolator((x, y, z), B[:, :, :, 2], r)
                ]
            ) * v, axis = 0
        )

        sprime = sprime.at[8, :].set(VerdetConst * ne_N * Bv_N)

    del r
    del v
    del amp

    # flattening is not changing its shape, it is a flattened array as its parallelised
    # solve_ivp expects it flattened anyway so either way this is the correct return format
    # only issue would be if it is flattened differently this time to the first and to how it was unflattened
    # - as then data would be in the wrong place
    return sprime.flatten()

def process_results(solutions, depth_traced, trace_depth, probing_direction, return_E, duration, save_points_per_region, ray_batch_count):
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

    if ray_batch_count > 1:
        from diffrax import Solution

        # Concatenate time and state arrays
        ts = jnp.concatenate([sol.ts for sol in solutions], axis = 0)
        ys = jnp.concatenate([sol.ys for sol in solutions], axis = 0)

        # Combine stats
        stats_keys = solutions[0].stats.keys()
        stats = {
            key: jnp.concatenate([sol.stats[key] for sol in solutions], axis = 0)
            for key in stats_keys
        }

        # Combine other fields
        t0 = solutions[0].t0
        t1 = solutions[-1].t1
        result = solutions[-1].result  # Use the last result

        del solutions

        # if info is missing that you need, this is why - implement it !
        solutions = Solution(
            t0=t0,
            t1=t1,
            ts=ts,
            ys=ys,
            interpolation=None,  # Optional: you can implement logic to keep interpolations
            stats=stats,
            result=result,
            solver_state=None,
            controller_state=None,
            made_jump=None,
            event_mask=None
        )

        solutions = np.asarray([solutions], dtype = Solution)

    #if sol.result == RESULTS.successful:
    #rf = sol.ys[:, -1, :].reshape(9, Np)# / scalar

    if save_points_per_region == 2 or save_points_per_region == 1:
        rf = solutions[0].ys[:, -1, :].T

        # depth_traced + trace_depth or just trace_depth
        return *ray_to_Jonesvector(rf, depth_traced + trace_depth, probing_direction = probing_direction, return_E = return_E), duration
    elif save_points_per_region > 2:
        slice_rf_list = []
        slice_Jf_list = []

        for i in range(len(solutions)):
            #save_point_depth = depth_traced
            for j in range(save_points_per_region):
                '''
                if j == save_points_per_region - 1:
                    save_point_depth = depth_traced + trace_depth
                else:
                    save_point_depth += trace_depth // save_points_per_region
                '''

                if j < save_points_per_region - 1 or (j == save_points_per_region - 1 and i == len(solutions) - 1):
                    # sol.ts having shape of (Np, save_points_per_region) per region is very inefficent given there are N - 1 duplications
                    # - issue with diffrax though I can't fix this
                    rf_slice, Jf_slice = ray_to_Jonesvector(solutions[i].ys[:, j, :].T, depth_traced + trace_depth * solutions[i].ts[0, j], probing_direction = probing_direction, return_E = return_E, keep_current_plane = True)

                    slice_rf_list.append(rf_slice)
                    if Jf_slice is not None:
                        slice_Jf_list.append(Jf_slice)

        rf = jnp.stack(slice_rf_list, axis = 0)
        del slice_rf_list

        if len(slice_Jf_list) > 0:
            Jf = jnp.stack(slice_Jf_list, axis = 0)
            del slice_Jf_list
        else:
            Jf = None

        return rf, Jf, duration
    else:
        assert "\nWhat."

def solve(beam, ScalarDomain, probing_depth, *, return_E = False, parallelise = True, jitted = True, save_points_per_region = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False, return_raw_results = False, verbose = True):
    # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
    VerdetConst = 0.0
    if (ScalarDomain.B_on):
        VerdetConst = 2.62e-13 * lwl ** 2 # radians per Tesla per m^2

    omega = 2 * jnp.pi * c / lwl

    Np_total = ScalarDomain.Np_total
    ray_batch_count = ScalarDomain.ray_batch_count

    from simulator.beam import Beam
    if ray_batch_count == 1:
        if not isinstance(beam, Beam):
            beam_instance = False

            s0_import = beam
            del beam
        elif isinstance(beam, Beam):
            beam_instance = True

            temp_beam = Beam(Np, beam_size = beam[0], divergence = beam[1], ne_extent = beam[2], probing_direction = beam[3], beam_type = beam[4], seeded = beam[5])
            s0_import = temp_beam.s0
            del temp_beam

        Np = s0_import.shape[1]
        rays_per_batch = Np # not necessary, just so there is something to print if someone tries

        rays = np.array([Np], dtype = np.int64)
    else:
        #Np = Np_total // ray_batch_count
        rays_per_batch = Np_total // ray_batch_count
        rays = np.array([rays_per_batch] * (ray_batch_count - 1) + [Np_total - rays_per_batch * (ray_batch_count - 1)], dtype = np.int64)

    # s0_import[:, 0] and s0_import input to getsizeof_default(...) produce the same result
    # I think this estimation is correct, if jax reports failing to allocate a lower amount, check the amount reported isn't just the max memory available
    # if it is, estimation is likely correct and this is just an issue with reporting
    # if it is lower, you likely have a memory leak
    # this is relevant generally not just for ray memory - just cropped up as an issue here first

    # if batched: or if auto_batching: etc.
    # proing_depth /= some integer with some corrections I expect
    # make logic too loop it and pick up from previous solution

    duration = 0
    solutions = np.empty(ray_batch_count, dtype=object)
    print(rays)
    print(rays_per_batch)
    print(ray_batch_count)
    print(Np_total)

    for Np in rays:
        depth_traced = 0.0

        if ray_batch_count > 1:
            print("c")
            #def __init__(self, Np, beam_size, divergence, ne_extent, *, probing_direction = 'z', wavelength = 1064e-9, beam_type = 'circular', seeded = False):

            temp_beam = Beam(Np, beam_size = beam[0], divergence = beam[1], ne_extent = beam[2], probing_direction = beam[3], beam_type = beam[4], seeded = beam[5])
            s0_import = temp_beam.s0
            del temp_beam
        print("d")

        print("\nEst. size in memory of rays:", mem_conversion(getsizeof_default(s0_import[:, 0]) * Np))
        if beam_instance:
            print("Est. potential size in memory of total rays:", mem_conversion(getsizeof_default(s0_import[:, 0]) * Np_total))
        if ray_batch_count > 1:
            print(" --> Np = {} ({} batches)".format(Np_total, ray_batch_count))
        else:
            print(" --> Np = {}".format(Np))

        print("ScalarDomain.region_count", ScalarDomain.region_count)
        for i in range(1, ScalarDomain.region_count + 1):
            print("hello")
            if ScalarDomain.region_count == 1:
                print("\nNo need to generate any sections of the domain, batching not utilised.")

                trace_depth = probing_depth
            else:
                if i == 1:
                    print("\nUsing pre-generated 1st section of domain.")
                else:
                    print("\nGenerating", add_integer_postfix(i), "section of the domain...")

                    lengths = ScalarDomain.lengths
                    dims = ScalarDomain.dims

                    ne_type = ScalarDomain.ne_type

                    inv_brems = ScalarDomain.inv_brems
                    phaseshift = ScalarDomain.phaseshift
                    B_on = ScalarDomain.B_on

                    probing_direction = ScalarDomain.probing_direction

                    region_count = ScalarDomain.region_count

                    leeway_factor = ScalarDomain.leeway_factor

                    coord_backup = ScalarDomain.coord_backup
                    future_dims = ScalarDomain.future_dims

                    try:
                        del ScalarDomain
                    except:
                        ScalarDomain = None

                    import simulator.domain as d
                    ScalarDomain = d.ScalarDomain(
                        lengths, dims,
                        ne_type = ne_type,
                        inv_brems = inv_brems,
                        phaseshift = phaseshift,
                        B_on = B_on,
                        probing_direction = probing_direction,
                        auto_batching = True,
                        iteration = i,
                        region_count = region_count,
                        leeway_factor = leeway_factor,
                        coord_backup = coord_backup,
                        future_dims = future_dims
                    )

                    del lengths
                    del dims

                    del ne_type

                    del inv_brems
                    del phaseshift
                    del B_on

                    del probing_direction

                    del region_count

                    del leeway_factor

                    del coord_backup
                    del future_dims

                # Need to make sure all rays have left volume
                # Conservative estimate of diagonal across volume
                # Then can backproject to surface of volume

                depth_remaining = probing_depth - depth_traced

                trace_depth = ScalarDomain.lengths[['x', 'y', 'z'].index(ScalarDomain.probing_direction)]
                if trace_depth > depth_remaining:
                    trace_depth = depth_remaining

                del depth_remaining

            target_depth = trace_depth + depth_traced

            # it isn't tracing up till this depth, it is tracing this amount further
            # at end positions are r(vector) + trace_depth (ish) NOT trace_depth(vector)
            print(" --> tracing a depth of", trace_depth, "mm's to the target depth of", target_depth, "mm's")

            t = jnp.linspace(0.0, jnp.sqrt(8.0) * trace_depth / c, 2)
            norm_factor = jnp.max(t)

            # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
            # think we should change this???

            # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not
            args = (parallelise, ScalarDomain.inv_brems, ScalarDomain.phaseshift, ScalarDomain.B_on, ScalarDomain.ne, ScalarDomain.B, ScalarDomain.Te, ScalarDomain.Z, ScalarDomain.x, ScalarDomain.y, ScalarDomain.z, omega, VerdetConst, ScalarDomain.lengths, ScalarDomain.dims)

            if not parallelise:
                from numpy import array
                if i == 1:
                    s0 = array(jnp.ravel(s0_import))
                    #s0 = s0.flatten() #odeint insists
                else:
                    # need a backpropogation algorithm that works for this too
                    s0 = array(jnp.ravel(sol))
                    del sol

                start = time()
                # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
                sol = solve_ivp(lambda t, y: dsdt(t, y, *args), [0, t[-1]], s0, t_eval = t)
            else:
                # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
                available_devices = jax.devices()

                running_device = jax.lib.xla_bridge.get_backend().platform # - deprecated, using still as needed for HPC
                #running_device = jax.extend.backend.get_backend().platform
                print("\nRunning device:", running_device, end='')

                if i == 1:
                    s0_transformed = s0_import.T
                    del s0_import
                else:
                    # change target_depth back to trace_depth and check the difference
                    s0_transformed = back_propogate(sol.ys[:, -1, :].T, target_depth, ScalarDomain.probing_direction).T
                    del sol

                if running_device == 'cpu':
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
                    pass

                    s0 = s0_transformed
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

                # using lengths and/or dims to set parameters of diffeqsolve(...) results in BooleanConversionError due to tracing variable resolution
                def diffrax_solve(dydt, t0, t1, Nt, lengths, dims, *, rtol = 1e-7, atol = 1e-9):
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
                    #dtmax = 0.5 * ((lengths[0] * lengths[1] * lengths[2]) / (dims[0] * dims[1] * dims[2])) ** (1 / 3) / (c * norm_factor)
                    stepsize_controller = PIDController(rtol = 1, atol = 1e-5)#, dtmax = dtmax)

                    return lambda s0, args : diffeqsolve(
                        term,
                        solver,
                        y0 = jnp.array(s0),
                        args = args,
                        t0 = t0,
                        t1 = t1,
                        dt0 = (t1 - t0) * norm_factor / Nt, # can set = 0 if dtmax is set apparently?
                        saveat = saveat,
                        stepsize_controller = stepsize_controller,
                        # set max steps to no. of cells x100
                        # cannot be passed as dims --> causes boolean conversion error, has to be passed directly
                        # need to pass this correctly so that it remains consistent with class when batching
                        max_steps = 10000#dims[0] * dims[1] * dims[2] * 100 #10000 - default for solve_ivp?????
                    )

                # hardcode to normalise to 1 due to diffrax bug
                ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, save_points_per_region, ScalarDomain.lengths, ScalarDomain.dims)

                if jitted:
                    start_comp = time()

                    from equinox import filter_jit
                    # equinox.filter_jit() (imported as filter_jit()) provides debugging info unlike jax.jit() - it does not like static args though so sticking with jit for now
                    #ODE_solve = jax.jit(ODE_solve)#, static_argnums = 1)#, device = available_devices[0])
                    ODE_solve = filter_jit(ODE_solve)#, device = available_devices[0])
                    # not sure about the performance of non-static specified arguments with filter_jit() - only use for debugging not in 'production'

                    print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

                # pass s0[:, i] for each ray via a jax.vmap for parallelisation
                start = time()
                sol = jax.block_until_ready(
                    # in_axes version ensures that vmap doesn't map args parameters, just s0
                    #jax.vmap(lambda rays, args: ODE_solve, in_axes = (0, None))(s0, args)

                    # default vmap_method argument is sequential, this is deprecated though and will cause a warning (if debugging) past jax 0.6.0
                    # look into different options for this parameter at a later date

                    jax.vmap(ODE_solve, in_axes = (0, None))(s0, args)
                )


            duration += time() - start

            if memory_debug:
                if parallelise:
                    # Visualises sharding, looks cool, but pretty useless - and a pain with higher core counts
                    jax.debug.visualize_array_sharding(sol.ys[:, -1, :])

                from utils import domain_estimate

                print(colour.BOLD + "\nMemory summary - total estimate:", mem_conversion(domain_estimate(ScalarDomain.dims) + (getsizeof_default(s0) + getsizeof_default(sol)) * Np) + colour.END)
                print("\nEst. size of domain:", mem_conversion(getsizeof_default(s0) * Np))
                print("Est. size of initial rays:", mem_conversion(getsizeof_default(s0) * Np))
                print("Est. size of solution class / single ray (?):", getsizeof(sol))
                print("Est. size of solution (bef. JV):", mem_conversion(getsizeof_default(sol) * Np))

                folder_name = "memory"
                postfix = "_benchmarks/"

                path = "evaluation/benchmarks/" + folder_name + "/"

                if os.path.isdir(os.getcwd() + "/" + path):
                    pass
                else:
                    path = os.getcwd() + "/../" + folder_name + postfix

                    if os.path.isdir(path):
                        pass
                    else:
                        try:
                            os.mkdir(path)
                        except OSError as e:
                            import errno

                            print("\nFailed to create folder above current working directory, attempting in cwd:")

                            path = os.getcwd() + "/" + folder_name + postfix

                            if os.path.isdir(path):
                                path = folder_name + postfix
                            else:
                                try:
                                    os.mkdir(path)
                                except OSError as e:
                                    print("\nFailed in cwd too! No folder created.")
                                    if e.errno != errno.EEXIST:
                                        raise

                                #if e.errno != errno.EEXIST:
                                #    raise

                from datetime import datetime
                path += "memory-domain" + str(ScalarDomain.dims[0]) + "_rays"+ str(s0.shape[1]) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
                jax.profiler.save_device_memory_profile(path)

                print("\n", end = '')
                if os.path.isfile(os.path.expanduser("~") + "/go/bin/pprof"):
                    #import sys
                    from os import system

                    #system(f"~/go/bin/pprof -top {sys.executable} memory_{N}.prof")
                    system(f"~/go/bin/pprof -top /bin/ls " + path)
                    #system(f"~/go/bin/pprof --web " + path)
                else:
                    print("No pprof install detected. Please download to visualise memory usage - requires Golang to run.")

            del s0

            if i == ScalarDomain.region_count:
                solutions.append(sol)
                del sol

            depth_traced += trace_depth

    print("\nCompleted ray trace in", colour.BOLD + str(jnp.round(duration, 3)) + colour.END, "seconds.")

    print("solutions.shape", solutions.shape)
    print(solutions[0])

    if return_raw_results:
        return solutions, None, duration
    else:
        if not parallelise:
            return *ray_to_Jonesvector(solutions.y[:,-1].reshape(9, Np), probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E), duration
        else:
            # need to confirm there is no mismatch between total depth_traced and the target probing_depth
            rf, Jf, duration = process_results(solutions, depth_traced, trace_depth, ScalarDomain.probing_direction, return_E, duration, save_points_per_region, ray_batch_count)

            print("rf.shape", rf.shape)
            print(rf)

            if verbose:
                print("\nParallelised output has resulting 3D matrix of form: [batch_count, (save_points_per_region - 1) * ScalarDomain.region_count, 9]:", sol.ys.shape)
                print(" - 2 to account for the start and end results (typical, can be greater if set)")
                print(" - 9 containing the 3 position and velocity components, amplitude, phase and polarisation")
                print(" - If batch_count is lower than expected, this is likely due to jax's forced integer batch sharding requirement over cpu cores.")

                print("\nWe slice the", end = " ")
                if len(rf.shape) == 3:
                    print("results", end = " ")
                else:
                    print("end result", end = " ")
                print("and transpose into the form:", rf.shape, "to work with later code.")

            #else:
            #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

            return rf, Jf, duration