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

from itertools import product

import numpy as np

from jax._src import dtypes
from jax._src.numpy import (asarray, broadcast_arrays,
                            empty, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

def RegularGridInterpolator(points, values, xi, method="linear", bounds_error=False, fill_value=np.nan):
    """
    Interpolate coordinates on a regular rectangular grid.

    JAX implementation of a custom trilinear interpolator to decrease memory overhead in our use case

    Args:
        coordinates: length-N sequence of arrays specifying the grid coordinates.
        values: N-dimensional array specifying the grid values.
        fill_value: value returned for coordinates outside the grid, defaults to NaN.

    Returns:
        results: interpolated value(s) instead of object to test.

    Examples:
        >>> coordinates = (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        >>> values = jnp.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> query_points = jnp.array([[1.5, 4.5], [2.2, 5.8]])
        >>> interpolated_values = trilinearInterpolator(coordinates, values, query_points)

        Array([30., 64.], dtype=float32)
    """

    if method != "linear":
        raise NotImplementedError("`method` has no effect, defaults to `linear` with no other options available")

    if bounds_error:
        raise NotImplementedError("`bounds_error` takes no effect under JIT")

    check_arraylike("RegularGridInterpolator", values)
    if len(points) > values.ndim:
        ve = f"there are {len(points)} point arrays, but values has {values.ndim} dimensions"
        raise ValueError(ve)

    values, = promote_dtypes_inexact(values)

    if fill_value is not None:
        check_arraylike("RegularGridInterpolator", fill_value)
        fill_value = asarray(fill_value)
        if not dtypes.can_cast(fill_value.dtype, values.dtype, casting='same_kind'):
            ve = "fill_value must be either 'None' or of a type compatible with values"
            raise ValueError(ve)

    # TODO: assert sanity of `points` similar to SciPy but in a JIT-able way
    check_arraylike("RegularGridInterpolator", *points)
    grid = tuple(asarray(p) for p in points)

    ndim = len(grid)

    """Convert a tuple of coordinate arrays to a (..., ndim)-shaped array."""
    if isinstance(xi, tuple) and len(xi) == 1:
        # handle argument tuple
        xi = xi[0]
    if isinstance(xi, tuple):
        p = broadcast_arrays(*xi)
        for p_other in p[1:]:
            if p_other.shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        xi = empty(p[0].shape + (len(xi),), dtype=float)
        for j, item in enumerate(p):
            xi = xi.at[..., j].set(item)
    else:
        check_arraylike("_ndim_coords_from_arrays", xi)
        xi = asarray(xi)  # SciPy: asanyarray(xi)
        if xi.ndim == 1:
            if ndim is None:
                xi = xi.reshape(-1, 1)
            else:
                xi = xi.reshape(-1, ndim)

    if xi.shape[-1] != len(grid):
        raise ValueError("the requested sample points xi have dimension"
                        f" {xi.shape[1]}, but this RegularGridInterpolator has"
                        f" dimension {ndim}")

    xi_shape = xi.shape
    xi = xi.reshape(-1, xi_shape[-1])

    # find relevant edges between which xi are situated
    indices = []
    # compute distance to lower edge in unity units
    norm_distances = []
    # check for out of bounds xi
    out_of_bounds = zeros((xi.T.shape[1],), dtype=bool)
    # iterate through dimensions
    for x, g in zip(xi.T, grid):
        i = searchsorted(g, x) - 1
        i = where(i < 0, 0, i)
        i = where(i > g.size - 2, g.size - 2, i)
        indices.append(i)
        norm_distances.append((x - g[i]) / (g[i + 1] - g[i]))
        if not bounds_error:
            out_of_bounds += x < g[0]
            out_of_bounds += x > g[-1]

    # slice for broadcasting over trailing dimensions in self.values
    vslice = (slice(None),) + (None,) * (values.ndim - len(indices))

    # find relevant values
    # each i and i+1 represents a edge
    edges = product(*[[i, i + 1] for i in indices])
    result = asarray(0.)
    for edge_indices in edges:
        weight = asarray(1.)
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= where(ei == i, 1 - yi, yi)
        result += values[edge_indices] * weight[vslice]

    if not bounds_error and fill_value is not None:
        bc_shp = result.shape[:1] + (1,) * (result.ndim - 1)
        result = where(out_of_bounds.reshape(bc_shp), fill_value, result)

    return result.reshape(xi_shape[:-1] + values.shape[ndim:])

def dndr(r, x, y, z, dndx, dndy, dndz):
    grad = jnp.zeros_like(r.T)
    coords = jnp.array([x, y, z])

    grad = grad.at[0, :].set(RegularGridInterpolator(coords, dndx, r, fill_value = 0.0))
    grad = grad.at[0, :].set(RegularGridInterpolator(coords, dndy, r, fill_value = 0.0))
    grad = grad.at[0, :].set(RegularGridInterpolator(coords, dndz, r, fill_value = 0.0))

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, x, y, z, dndx, dndy, dndz):
    # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
    s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    r = s[:3, :].T
    v = s[3:6, :]

    amp = s[6, :]

    sprime = sprime.at[3:6, :].set(dndr(r, x, y, z, dndx, dndy, dndz))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, ne_extent, *, probing_direction = 'z', keep_current_plane = False, return_E = False):
    Np = rays.shape[1] # number of photons

    ray_p = jnp.zeros((4, Np))

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    t_bp = (z - ne_extent) / vz

    ray_p = ray_p.at[0].set(x - vx * t_bp)
    ray_p = ray_p.at[2].set(y - vy * t_bp)

    # Angles to plane
    ray_p = ray_p.at[1].set(jnp.arctan(vx / vz))
    ray_p = ray_p.at[3].set(jnp.arctan(vy / vz))

    return ray_p

def solve_alt(s0_import, ScalarDomain, dims, probing_depth, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False):
    import os
    os.environ["EQX_ON_ERROR"] = "breakpoint"

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, args[0], args[1], args[2], args[3], args[4], args[5]) * norm_factor

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

    if jitted:
        start_comp = time()

        from equinox import filter_jit

        ODE_solve = filter_jit(ODE_solve)

        print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

    x = jnp.float32(jnp.linspace(-ScalarDomain.x_length/2, ScalarDomain.x_length/2, ScalarDomain.x_n))
    y = jnp.float32(jnp.linspace(-ScalarDomain.y_length/2, ScalarDomain.y_length/2, ScalarDomain.y_n))
    z = jnp.float32(jnp.linspace(-ScalarDomain.z_length/2, ScalarDomain.z_length/2, ScalarDomain.z_n))

    omega = 2 * jnp.pi * c / lwl

    dndx = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    dndy = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    dndz = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), z, axis = 2)

    args = (x, y, z, dndx, dndy, dndz)
    sol = jax.vmap(lambda s: ODE_solve(s, args))(s0_import.T)

    return ray_to_Jonesvector(sol.ys[:, -1, :].T, probing_depth)

def solve(s0_import, ScalarDomain, dims, probing_depth, *, return_E = False, parallelise = True, jitted = True, save_steps = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False):
    t = jnp.linspace(0.0, jnp.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, *args) * norm_factor

    from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve

    def diffrax_solve(dydt, t0, t1, Nt, *, rtol = 1e-7, atol = 1e-9):
        term = ODETerm(dydt)
        solver = Tsit5()
        saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
        stepsize_controller = PIDController(rtol = 1, atol = 1e-5)

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

    if jitted:
        start_comp = time()

        from equinox import filter_jit

        ODE_solve = filter_jit(ODE_solve)

        print("\njax compilation of solver took:", time() - start_comp, "seconds", end='')

    x = jnp.float32(jnp.linspace(-ScalarDomain.x_length/2, ScalarDomain.x_length/2, ScalarDomain.x_n))
    y = jnp.float32(jnp.linspace(-ScalarDomain.y_length/2, ScalarDomain.y_length/2, ScalarDomain.y_n))
    z = jnp.float32(jnp.linspace(-ScalarDomain.z_length/2, ScalarDomain.z_length/2, ScalarDomain.z_n))

    omega = 2 * jnp.pi * c / lwl

    dndx = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    dndy = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    dndz = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), z, axis = 2)

    args = (x, y, z, dndx, dndy, dndz)

    start = time()
    sol = jax.block_until_ready(
        jax.vmap(ODE_solve, in_axes = (0, None))(s0_import.T, args)
    )

    duration = time() - start
    print("\nCompleted ray trace in", jnp.round(duration, 3), "seconds.")

    return ray_to_Jonesvector(sol.ys[:, -1, :].T, probing_depth, probing_direction = ScalarDomain.probing_direction, return_E = return_E)#, duration