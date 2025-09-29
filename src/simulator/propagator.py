import jax
import jax.numpy as jnp
import numpy as np

import os

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

# object type of diffrax output
from diffrax import Solution

from scipy.constants import c
from scipy.constants import e
from jax.scipy.interpolate import RegularGridInterpolator
from shared.utils import getsizeof
from shared.utils import mem_conversion
from shared.printing import colour
from shared.utils import add_integer_postfix
from shared.SpK_reader import open_emi_files
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

def opacity_grid_generation(domain, energy):
    opa_max = domain.z_n/domain.z_length

    if domain.num_materials == 1:
        # pulls data from files
        grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{domain.opacity_files}")

        # creates an energy grid?
        energy_grid = jnp.full_like(domain.densities, energy)
        # maps data and energies to grid via interpolation for later calculation, returns opacity_mapped_grid
        return trilinearInterpolator((grp_centres, rho, Te), jnp.minimum(opa_max, opa_data), (energy_grid, domain.densities, domain.Te))

        ### opacity_mapped_grid is what needs to be passed as a new domain sized object to be interpolated
        ### trilinearInterpolator will need generalising to take in a 3D grid of interpolation points

        # interpolates at points r
        #opacity_spatial_grid = trilinearInterpolator((domain.x, domain.y, domain.z), opacity_mapped_grid, r)
    else:
        if (len(domain.opacity_files) != len(domain.densities)) & (len(domain.densities) != domain.num_materials):
            raise ValueError("densities and opacity_files must have length equal to num_materials")

        opacity_grids = [0] * domain.num_materials
        for i in range (domain.num_materials):
            grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{domain.opacity_files[i]}")

            # rhos = domain.densities[i].ravel()
            # r = jnp.c_[domain.energy*jnp.ones_like(rhos),rhos,domain.Te.ravel()]
            opacity_grids[i] = trilinearInterpolator((grp_centres, rho, Te), jnp.minimum(opa_max, opa_data), (energy, domain.densities[i], domain.Te))

        # returns opacity_grids_tot
        return np.sum(opacity_grids, axis = 0)

        # interpolates at points r
        #opacity_spatial_grid_tot = trilinearInterpolator((domain.x, domain.y, domain.z), opacity_grids_tot, r)

def attenuation(domain, energy):
    opa_max = domain.z_n/domain.z_length
    
    if domain.num_materials == 1:
        grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{domain.opacity_files}")

        opa_data_capped = jnp.minimum(opa_max, opa_data)
        opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped, bounds_error = False, fill_value = 0.0)
        energy_grid = jnp.full_like(domain.densities, energy)
        opacity_grid = opacity_interp((energy_grid, domain.densities, domain.Te))
        opacity_spatial_interp = RegularGridInterpolator((domain.x, domain.y, domain.z), opacity_grid, bounds_error = False, fill_value = 0.0)

        return opacity_spatial_interp
    else:
        if (len(domain.opacity_files) != len(domain.densities)) & (len(domain.densities) != domain.num_materials):
            raise ValueError("densities and opacity_files must have length equal to num_materials")

        opacity_grids = [0] * domain.num_materials
        for i in range (domain.num_materials):
            grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{domain.opacity_files[i]}")

            opa_data_capped = jnp.minimum(opa_max, opa_data)
            # rhos = domain.densities[i].ravel()
            # r = jnp.c_[domain.energy*jnp.ones_like(rhos),rhos,domain.Te.ravel()]
            opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped,bounds_error = False, fill_value = 0.0)
            opacity_grids[i] = opacity_interp((energy, domain.densities[i], domain.Te))

        opacity_grids_tot = np.sum(opacity_grids, axis = 0)
        opacity_spatial_interp_tot = RegularGridInterpolator((domain.x, domain.y, domain.z), opacity_grids_tot, bounds_error = False, fill_value = 0.0)

        return opacity_spatial_interp_tot

def dndr(r, gradient_term, omega, x, y, z):
    """
    Returns the gradient at the locations r

    Args:
        r (3xN float): N [x, y, z] locations

    Returns:
        3 x N float: N [dx, dy, dz] electron density gradients
    """

    grad = jnp.zeros_like(r.T)

    dndx = jnp.gradient(gradient_term, x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), dndx, r, fill_value = 0.0))
    del dndx

    dndy = jnp.gradient(gradient_term, y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), dndy, r, fill_value = 0.0))
    del dndy

    dndz = jnp.gradient(gradient_term, z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, parallelise, inv_brems, phaseshift, B_on, ne, B, Te, Z, x, y, z, omega, VerdetConst, lengths, dims, opacity, edensity, refrac_field, opacity_interp):
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
        print("False")
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (9, s.size // 9))
    else:
        print("True")
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

    if edensity is True:
        print("True")
        gradient_term = -0.5 * c ** 2 * ne / (3.14207787e-4 * omega ** 2)
    else:
        print("False")
        gradient_term = 0.5 * c ** 2 * refrac_field ** 2

    # must unpack x, y, z tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, gradient_term, omega, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    ###
    ### Sort out passed functions and objects
    ###

    # Attenuation due to x-ray opacity, this takes into account inverse brehmmstrauhlung effects - hence opacity = True overrides inv_brems = True
    if opacity:
        print("opacity")
        sprime = sprime.at[6, :].set(-opacity_interp(r) * c * amp)
    # Attenuation due to inverse bremsstrahlung
    if inv_brems:
        print("inv_brems")
        sprime = sprime.at[6, :].set(trilinearInterpolator((x, y, z), kappa(ne, Te, Z, omega), r) * amp)

    ##
    ## Commented out code is previous version - this was apparently causing floating point errors (Alan did not specify what/how)
    ## Second form of this is the expansion of the n_refrac() function directly into calculation
    ##
    ## Current version is an expansion of the expression to avoid these floating point errors
    ## It has been changes to use trilinearInterpolator in similar fashion to original instead of passing a phase() function as this change was made to do originally
    ##

    '''
    if phaseshift:
        print("phaseshift")
        sprime = sprime.at[7, :].set(omega * (trilinearInterpolator((x, y, z), n_refrac(ne, omega), r) - 1.0))
        #sprime = sprime.at[7, :].set(omega * (trilinearInterpolator((x, y, z), jnp.sqrt(1.0 - (5.64 * jnp.sqrt(ne) / omega) ** 2), r) - 1.0))
    '''
    if phaseshift:
        print("phaseshift")
        sprime = sprime.at[7, :].set(jnp.array(-0.5 * trilinearInterpolator((x, y, z), ne, r) / (3.14207787e-4 * omega), dtype = jnp.float64))

    if B_on:
        print("B_on")
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

def process_results(solutions, depth_traced, trace_depth, probing_direction, return_E, duration, save_points_per_region, ray_batch_count, verbose, amp_phase_return):
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

    #else:
    #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

    #if sol.result == RESULTS.successful:
    #rf = sol.ys[:, -1, :].reshape(9, Np)# / scalar

    if ray_batch_count > 1:
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
            t0 = t0,
            t1 = t1,
            ts = ts,
            ys = ys,
            interpolation = None,  # Optional: you can implement logic to keep interpolations
            stats = stats,
            result = result,
            solver_state = None,
            controller_state = None,
            made_jump = None,
            event_mask = None
        )

        solutions = np.asarray([solutions], dtype = Solution)

    if verbose:
        print("\nParallelised output has resulting 3D matrix of form: [batch_count, (save_points_per_region - 1) * ScalarDomain.region_count, 9]:", solutions[0].ys.shape)
        print(" - 2 to account for the start and end results (typical, can be greater if set)")
        print(" - 9 containing the 3 position and velocity components, amplitude, phase and polarisation")
        print(" - If batch_count is lower than expected, this is likely due to jax's forced integer batch sharding requirement over cpu cores.")

        print("\nWe slice the", end = " ")
        if len(solutions[0].ys.shape) == 3:
            print("results", end = " ")
        else:
            print("end result", end = " ")
        print("and transpose into the form:", solutions[0].ys.shape, "to work with later code.")

    if save_points_per_region == 2 or save_points_per_region == 1:
        rf = solutions[0].ys[:, -1, :].T

        # depth_traced + trace_depth or just trace_depth
        return *ray_to_Jonesvector(rf, ne_extent = depth_traced + trace_depth, probing_direction = probing_direction, return_E = return_E, amp_phase_return = amp_phase_return), duration
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
                    rf_slice, Jf_slice = ray_to_Jonesvector(solutions[i].ys[:, j, :].T, ne_extent = depth_traced + trace_depth * solutions[i].ts[0, j], probing_direction = probing_direction, return_E = return_E, keep_current_plane = True, amp_phase_return = amp_phase_return)

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
    # General assertions
    if return_E == False and ScalarDomain.B_on == True:
        print(colour.BOLD + "Warning:" + colour.END + "return_E == False and ScalarDomain.B_on == True leads to pointless calculations as the output will not be used.")
        print("\n --> Proceeding with run as per request, however please note this suggestion")
    assert return_E == ScalarDomain.B_on or (return_E == False and ScalarDomain.B_on == True), colour.BOLD + "Setting ScalarDomain.B_on == False and return_E == True leads to incorrect results as B field calculations will not occur\n --> Stopping process, correct settings and re-run" + colour.END

    if ScalarDomain.opacity is True or ScalarDomain.inv_brems is True or ScalarDomain.phaseshift is True:
        amp_phase_return = True
    else:
        amp_phase_return = False

    # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
    VerdetConst = 0.0
    if (ScalarDomain.B_on):
        VerdetConst = 2.62e-13 * lwl ** 2 # radians per Tesla per m^2

    if (ScalarDomain.opacity):
        opacity_domain_grid = opacity_grid_generation(domain = ScalarDomain, energy = 6.63e-34 * c / (lwl * 1.6e-19))

        energy = 6.63e-34 * c / (lwl * 1.6e-19)
        opacity_interp = attenuation(domain = ScalarDomain, energy = energy)
        def atten(x):
            return opacity_interp(x)
    else:
        opacity_domain_grid = 0.0

        def atten():
            return 0.0

    omega = 2 * jnp.pi * c / lwl

    region_count = ScalarDomain.region_count
    ray_batch_count = ScalarDomain.ray_batch_count

    print("\nNumber of domain batches:", region_count)
    print("Number of ray batches:", ray_batch_count)

    from simulator.beam import Beam
    assert not isinstance(beam, Beam), "\nThis function does not take in the direct output of the Beam object, pass either Beam.s0 rays, or the parameters passed to be Beam here as a tuple if batching rays."

    unbatched_beam = False
    if ray_batch_count == 1:
        import array
        if isinstance(beam, array.array) or isinstance(beam, np.ndarray) or isinstance(beam, jax.Array):
            assert len(beam.shape) == 2, "\nExpected a matrix of pre-created rays."

            s0_import = beam
            del beam

            Np = s0_import.shape[1]

            Np_total = Np
            rays_per_batch = Np # not necessary, just so there is something to print if someone tries

            rays = np.array([Np], dtype = np.int64)
        elif isinstance(beam, tuple):
            unbatched_beam = True

            print("\nUsing tuple values to create the unbatched beam, domain must be used in the same fashion.")

            Np_total = ScalarDomain.Np_total
            rays_per_batch = Np_total

            rays = np.array([Np_total], dtype = np.int64)
    else:
        assert isinstance(beam, tuple), "\nExpect a tuple of Beam properties if you wish to batch rays."

        Np_total = ScalarDomain.Np_total

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

    duration = np.float64(0.0)
    solutions = np.empty(ray_batch_count, dtype = Solution)

    for ray_index, Np in enumerate(rays):
        depth_traced = 0.0

        if ray_batch_count > 1 or unbatched_beam:
            s0_import = Beam(Np, beam_size = beam[0], divergence = beam[1], ne_extent = beam[2], probing_direction = beam[3], beam_type = beam[4], seeded = beam[5]).s0

        single_ray_size = getsizeof_default(s0_import[:, 0])
        print("\nEst. size in memory of rays (1 = {}): {}".format(mem_conversion(single_ray_size), mem_conversion(single_ray_size * Np)))
        total_ray_size_estimate_raw = getsizeof_default(s0_import[:, 0]) * Np_total
        if ray_batch_count > 1:
            print("Est. potential size in memory of total rays:", mem_conversion(total_ray_size_estimate_raw))
            print(" --> Np (total) = {} (in {} batches) - {} for this batch".format(Np_total, ray_batch_count, Np))
        else:
            print(" --> Np = {}".format(Np))

        for i in range(1, ScalarDomain.region_count + 1):
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

                    refrac_field = ScalarDomain.refrac_field
                    opacity_files = ScalarDomain.opacity_files
                    densities = ScalarDomain.densities
                    num_materials = ScalarDomain.num_materials

                    inv_brems = ScalarDomain.inv_brems
                    opacity = ScalarDomain.opacity
                    phaseshift = ScalarDomain.phaseshift
                    B_on = ScalarDomain.B_on
                    edensity = ScalarDomain.edensity

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
                        refrac_field = refrac_field,
                        opacity_files = opacity_files,
                        densities = densities,
                        num_materials = num_materials,
                        inv_brems = inv_brems,
                        opacity = opacity,
                        phaseshift = phaseshift,
                        B_on = B_on,
                        edensity = edensity,
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

                    del refrac_field
                    del opacity_files
                    del densities
                    del num_materials

                    del inv_brems
                    del opacity
                    del phaseshift
                    del B_on
                    del edensity

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
            args = (
                parallelise, ScalarDomain.inv_brems, ScalarDomain.phaseshift, ScalarDomain.B_on, 
                ScalarDomain.ne, ScalarDomain.B, ScalarDomain.Te, ScalarDomain.Z,
                ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
                omega, VerdetConst,
                ScalarDomain.lengths, ScalarDomain.dims,
                ScalarDomain.opacity, ScalarDomain.edensity, ScalarDomain.refrac_field,
                opacity_domain_grid
            )

            ###
            ### Check the original algorithm still works for the sake of testing
            ###

            if not parallelise:
                from numpy import array

                assert i == 1, "\nDomain batching is not set up to work with the legacy solver yet."

                s0 = array(jnp.ravel(s0_import))
                #s0 = s0.flatten() #odeint insists

                '''
                # need a backpropogation algorithm that works for this too
                s0 = array(jnp.ravel(sol))
                del sol
                '''

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

                    if Np >= core_count:
                        from jax.sharding import PartitionSpec as P, NamedSharding

                        # Create a Sharding object to distribute a value across devices:
                        # Assume self.core_count is the no. of core devices available
                        mesh = jax.make_mesh((core_count,), ('rows',))  # 1D mesh for columns

                        # Specify sharding: don't split axis 0 (rows), split axis 1 (columns) across devices
                        # then apply sharding to rewrite s0 as a sharded array from it's original matrix
                        # and use jax.device_put to distribute it across devices:
                        Np = ((Np // core_count) * core_count)
                        #assert Np > 0, "Not enough rays to parallelise over cores, increase to at least " + str(core_count)

                        # if you don't wish to transpose before operation you need to use the old call
                        # s0 = jax.device_put(s0_transformed[:, 0:Np], NamedSharding(mesh, P(None, 'cols')))
                        s0 = jax.device_put(s0_transformed[0:Np, :], NamedSharding(mesh, P('rows', None)))  # 'None' means don't shard axis 0

                        print(s0.sharding)            # See the sharding spec
                        #print(s0.addressable_shards)  # Check each device's shard
                        #jax.debug.visualize_array_sharding(s0)
                    else:
                        s0 = jax.device_put(s0_transformed)

                        print(colour.BOLD + "Not enough rays to parallelise over cores" + colour.END + ": increase to at least " + str(core_count) + " to utilise parallelisation")
                        print(" --> Running CPU processes sequentially")
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
                def diffrax_solve(dydt, t0, t1, Nt, lengths, dims, *, rtol = 1e-2, atol = 1e-5):
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
                    dtmax = 0.5 * ((lengths[0]/dims[0])**2 + (lengths[1]/dims[1])**2 + (lengths[2]/dims[2])**2) ** (1 / 2) / (c * norm_factor)
                    stepsize_controller = PIDController(rtol = rtol, atol = atol, dtmax = dtmax)

                    return lambda s0, args : diffeqsolve(
                        term,
                        solver,
                        y0 = jnp.array(s0),
                        args = args,# + (atten, ),
                        t0 = t0,
                        t1 = t1,
                        dt0 = None, # can set = 0 if dtmax is set apparently?
                        saveat = saveat,
                        stepsize_controller = stepsize_controller,
                        # set max steps to no. of cells x100
                        # cannot be passed as dims --> causes boolean conversion error, has to be passed directly
                        # need to pass this correctly so that it remains consistent with class when batching
                        max_steps = int(2e8) #dims[0] * dims[1] * dims[2] * 100 #10000 - default for solve_ivp?????
                    )

                # hardcode to normalise to 1 due to diffrax bug
                ODE_solve = diffrax_solve(dsdt_ODE, 0, 1, save_points_per_region, ScalarDomain.lengths, ScalarDomain.dims)

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

            duration += np.float64(time() - start)

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

            ###
            ### Test if streaming is still the source of memory issues by using, del s0 test again
            ###

            #del s0

            #del sol - # this (and commenting out below section) prevents memory issues, so clearly solutions[...] needs to be
            # forced written to storage if over a certain memory limit

            # if est. solutions < ram but > vram, write to ram
            # if > both, write to storage
            # if < vram, keep on gpu - but then it wouldn't be batched anyway so sort of irrelevant

            if i == ScalarDomain.region_count:
                from shared.utils import memory_report

                if total_ray_size_estimate_raw >= memory_report("cpu")['free_raw']:
                    target_folder = os.getcwd() + "/saves"
                    if not os.path.isdir(target_folder):
                        try:
                            os.mkdir(target_folder)
                        except OSError as e:
                            print("\nFailed to create folder at " + target_folder)
                            if e.errno != errno.EEXIST:
                                raise

                    tar_gz_path = target_folder + "/ray_output_total_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5.tar.gz"

                    '''
                    from utils.handle_filetypes import save_jax_matrix_to_hdf5 as compressed_solution_export
                    filepath, filename = compressed_solution_export(
                        ray_to_Jonesvector(sol.ys[:,-1].reshape(9, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E, amp_phase_return = amp_phase_return)[0],
                        file_path = target_folder
                        #filename = None, file_path = ".", dataset_name = 'data', compression = 'gzip', compression_level = 4
                    )

                    from utils.handle_filetypes import move_file_to_tar_gz
                    move_file_to_tar_gz(tar_gz_path, filepath)
                    '''

                    from utils.handle_filetypes import compress_matrix_to_hdf5_BytesIO
                    from utils.handle_filetypes import stream_data_to_tar_gz

                    filename = "run_" + str(ray_index)
                    stream_data_to_tar_gz(tar_gz_path, filename,
                        compress_matrix_to_hdf5_BytesIO(
                            ray_to_Jonesvector(sol.ys[:,-1].reshape(9, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E, amp_phase_return = amp_phase_return)[0]
                        )
                    )
                else:
                    solutions[ray_index] = sol
                    del sol

            depth_traced += trace_depth

    print("\nCompleted ray trace in", colour.BOLD + str(np.round(duration, 3).astype(np.float64)) + colour.END, "seconds.")

    if total_ray_size_estimate_raw < memory_report("cpu")['free_raw']:
        if return_raw_results:
            return solutions, None, duration
        else:
            if not parallelise:
                return *ray_to_Jonesvector(solutions.ys[:,-1].reshape(9, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E, amp_phase_return = amp_phase_return), duration
            else:
                # need to confirm there is no mismatch between total depth_traced and the target probing_depth
                return process_results(solutions, depth_traced, trace_depth, ScalarDomain.probing_direction, return_E, duration, save_points_per_region, ray_batch_count, verbose, amp_phase_return)
    else:
        print("\nData output as a hdf4.tar.gz file due to limitations of vram/ram space.")
        print("Graphs can be iteratively plotted by cycling through the 'run_n' entries after extraction from .tar.gz format.")