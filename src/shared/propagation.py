import jax.numpy as jnp

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
        ne_extent (float): edge lengths of shape (cuboid) in probing direction, m
        probing_direction (str): x, y or z.
        keep_current_plane (boolean): flag to enable compatability (via True) with use in diagnostics.py, defaults to False

    Returns:
        [type]: [description]
    """

    Np = rays.shape[1] # number of photons

    ray_p = jnp.zeros((4, Np))
    if return_E:
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

    # test returning these values and using the fresnel integral
    #return ray_p, rays[6], rays[7]

    # ray_p [x, phi, y, theta], ray_J [E_x, E_y]

    if return_E:
        return ray_p, ray_J

    return ray_p, None

def back_propogate(rays, ne_extent, probing_direction):
    Np = rays.shape[1] # number of photons

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x - ne_extent) / vx

        # Positions on plane
        rays = rays.at[0].set(ne_extent)
        rays = rays.at[1].set(y - vy * t_bp)
        rays = rays.at[2].set(z - vz * t_bp)
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
        rays = rays.at[0].set(z - vz * t_bp)
        rays = rays.at[1].set(ne_extent)
        rays = rays.at[2].set(x - vx * t_bp)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z - ne_extent) / vz

        # Positions on plane
        rays = rays.at[0].set(x - vx * t_bp)
        rays = rays.at[1].set(y - vy * t_bp)
        rays = rays.at[2].set(ne_extent)
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")

    del x
    del vx

    del y
    del vy

    del z
    del vz

    return rays