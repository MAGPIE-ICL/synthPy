import jax
import jax.numpy as jnp
import os

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

from scipy.constants import c
from scipy.constants import e
#from scipy.constants import hbar
#from scipy.constants import m_e

from utils import getsizeof
#from utils import trilinearInterpolator
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

def dndr(r, x, y, z, ne, omega, lengths, dims):
    grad = jnp.zeros_like(r)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndx, r.T, fill_value = 0.0))
    del dndx

    dndy = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndy, r.T, fill_value = 0.0))
    del dndy

    dndz = -0.5 * c ** 2 * jnp.gradient(ne / (3.14207787e-4 * omega ** 2), z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator(x, y, z, lengths, dims, dndz, r.T, fill_value = 0.0))
    del dndz

    return grad

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

    #return_E_test = True
    #if return_E_test:
    #    return ray_p, rays[6], rays[7]

    # ray_p [x, phi, y, theta], ray_J [E_x, E_y]
    if return_E:
        return ray_p, ray_J

    return ray_p, None

def dsdt(t, s, x, y, z, ne, omega, lengths, dims):
    # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
    s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    r = s[:3, :]
    v = s[3:6, :]

    a = s[6, :]

    sprime = sprime.at[3:6, :].set(dndr(r, x, y, z, ne, omega, lengths, dims))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

def solve(s0, ScalarDomain, lwl, probing_depth, return_E = False, parallelise = True, jitted = True):
    t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    start = time()

    def dsdt_ODE(t, y, args):
        # does *args versus args[0], args[1], ... make a performance difference?
        return dsdt(t, y, args[0], args[1], args[2], args[3], args[4], args[5], args[6]) * norm_factor

    import diffrax

    def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
        term = diffrax.ODETerm(dsdt_ODE)
        solver = diffrax.Tsit5()

        saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
        stepsize_controller = diffrax.PIDController(rtol = 1, atol = 1e-5)

        return lambda s0, args : diffrax.diffeqsolve(
            term,
            solver,
            y0 = jnp.array(s0),
            args = args,
            t0 = t0,
            t1 = t1,
            dt0 = (t1 - t0) * norm_factor / Nt,
            saveat = saveat,
            stepsize_controller = stepsize_controller
        )

    ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, len(t))

    from equinox import filter_jit

    if jitted:
        start_comp = time()

        ODE_solve = filter_jit(ODE_solve)

        finish_comp = time()
        print("jax compilation of solver took:", finish_comp - start_comp)

    x = jnp.float32(jnp.linspace(-ScalarDomain.x_length/2, ScalarDomain.x_length/2, ScalarDomain.x_n))
    y = jnp.float32(jnp.linspace(-ScalarDomain.y_length/2, ScalarDomain.y_length/2, ScalarDomain.y_n))
    z = jnp.float32(jnp.linspace(-ScalarDomain.z_length/2, ScalarDomain.z_length/2, ScalarDomain.z_n))

    omega = 2 * jnp.pi * c / lwl

    args = (x, y, z, ScalarDomain.ne, omega, ScalarDomain.lengths, ScalarDomain.dims)
    sol = jax.vmap(lambda s: ODE_solve(s, args))(s0.T)

    finish = time()
    duration = finish - start

    rf = sol.ys[:, -1, :].T
    rf, _ = ray_to_Jonesvector(rf, probing_depth, probing_direction = ScalarDomain.probing_direction)

    return rf