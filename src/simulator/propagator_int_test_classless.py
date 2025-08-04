import numpy as np
import diffrax
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from scipy.integrate import odeint, solve_ivp
from time import time
from equinox import filter_jit

from itertools import product

from scipy.constants import c
from jax._src import dtypes
from jax._src.numpy import (asarray, broadcast_arrays,
                            empty, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

def trilinearInterpolator(points, values, xi, method="linear", bounds_error=False, fill_value=np.nan):
    if method != "linear":
        raise NotImplementedError("`method` has no effect, defaults to `linear` with no other options available")

    if bounds_error:
        raise NotImplementedError("`bounds_error` takes no effect under JIT")

    #check_arraylike("RegularGridInterpolator", values)
    if len(points) > values.ndim:
        ve = f"there are {len(points)} point arrays, but values has {values.ndim} dimensions"
        raise ValueError(ve)

    values, = promote_dtypes_inexact(values)

    if fill_value is not None:
        #check_arraylike("RegularGridInterpolator", fill_value)
        fill_value = asarray(fill_value)
        if not dtypes.can_cast(fill_value.dtype, values.dtype, casting='same_kind'):
            ve = "fill_value must be either 'None' or of a type compatible with values"
            raise ValueError(ve)

    # TODO: assert sanity of `points` similar to SciPy but in a JIT-able way
    #check_arraylike("RegularGridInterpolator", *points)
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
        #check_arraylike("_ndim_coords_from_arrays", xi)
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

def dndr_scipy_alt(r, x, y, z, dndx, dndy, dndz):
    grad = jnp.zeros_like(r)

    coords = jnp.array([x, y, z])

    grad = grad.at[0, :].set(
        trilinearInterpolator(
            coords,
            dndx,
            r.T,
            fill_value = 0.0
        )
    )

    grad = grad.at[1, :].set(
        trilinearInterpolator(
            coords,
            dndy,
            r.T,
            fill_value = 0.0
        )
    )

    grad = grad.at[2, :].set(
        trilinearInterpolator(
            coords,
            dndz,
            r.T,
            fill_value = 0.0
        )
    )

    return grad

def solve(s0_import, probing_depth, ScalarDomain, lwl, return_E = False):
    import os
    os.environ["EQX_ON_ERROR"] = "breakpoint"

    s0 = s0_import.T

    t = np.linspace(0.0, np.sqrt(8.0) * probing_depth / c, 2)
    norm_factor = jnp.max(t)

    start = time()

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, args[0], args[1], args[2], args[3], args[4], args[5]) * norm_factor

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
    start_comp = time()

    ODE_solve = filter_jit(ODE_solve)

    finish_comp = time()
    print("jax compilation of solver took:", finish_comp - start_comp)

    x = jnp.float32(jnp.linspace(-ScalarDomain.x_length/2, ScalarDomain.x_length/2, ScalarDomain.x_n))
    y = jnp.float32(jnp.linspace(-ScalarDomain.y_length/2, ScalarDomain.y_length/2, ScalarDomain.y_n))
    z = jnp.float32(jnp.linspace(-ScalarDomain.z_length/2, ScalarDomain.z_length/2, ScalarDomain.z_n))

    omega = 2 * jnp.pi * c / lwl 

    dndx = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), x, axis = 0)
    dndy = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), y, axis = 1)
    dndz = -0.5 * c ** 2 * jnp.gradient(ScalarDomain.ne / (3.14207787e-4 * omega ** 2), z, axis = 2)

    args = (x, y, z, dndx, dndy, dndz)
    sol = jax.vmap(lambda s: ODE_solve(s, args))(s0)

    duration = time() - start
    print("Time to ray trace:", duration)

    return ray_to_Jonesvector(sol.ys[:, -1, :].T, probing_depth), duration

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, x, y, z, dndx, dndy, dndz):
    # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
    s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    r = s[:3, :]
    v = s[3:6, :]

    a = s[6, :]

    sprime = sprime.at[3:6, :].set(dndr_scipy_alt(r, x, y, z, dndx, dndy, dndz))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

def ray_to_Jonesvector(ode_sol, ne_extent):
    Np = ode_sol.shape[1] # number of photons

    ray_p = np.zeros((4, Np))

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    t_bp = (z - ne_extent) / vz

    # Positions on plane
    ray_p[0] = x - vx * t_bp
    ray_p[2] = y - vy * t_bp

    # Angles to plane
    ray_p[1] = np.arctan(vx / vz)
    ray_p[3] = np.arctan(vy / vz)

    return ray_p