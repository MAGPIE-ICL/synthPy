# from jax import numpy as jnp - would that work too? what's the difference if so?
import jax.numpy as jnp

from itertools import product

from jax import dtypes
from jax.numpy import (asarray, broadcast_arrays,
                            empty, searchsorted, where, zeros)
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

def trilinearInterpolator_old(coordinates, values, query_points, *, fill_value = jnp.nan):
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

    # Initially derived and customised from jax's implementation (our's was causing solver errors)
    # This was based on SciPy's implementation which in turn is originally based on an
    # implementation by Johannes Buchner

    check_arraylike("RegularGridInterpolator", values)
    if len(coordinates) > values.ndim:
        raise ValueError(f"there are {len(coordinates)} point arrays, but values has {values.ndim} dimensions")

    values, = promote_dtypes_inexact(values)

    '''
    if fill_value is not None:
        check_arraylike("RegularGridInterpolator", fill_value)
        fill_value = asarray(fill_value)
        if not dtypes.can_cast(fill_value.dtype, values.dtype, casting='same_kind'):
            ve = "fill_value must be either 'None' or of a type compatible with values"
            raise ValueError(ve)
    '''

    # TODO: assert sanity of `coordinates` similar to SciPy but in a JIT-able way
    check_arraylike("RegularGridInterpolator", *coordinates)

    """Convert a tuple of coordinate arrays to a (..., ndim)-shaped array."""
    ndim = len(coordinates[-1])

    if isinstance(query_points, tuple):
        if len(query_points) == 1:
            # handle argument tuple
            query_points = query_points[0]
        else:
            p = broadcast_arrays(*query_points)
            for p_other in p[1:]:
                if p_other.shape != p[0].shape:
                    raise ValueError("coordinate arrays do not have the same shape")

            query_points = empty(p[0].shape + (len(query_points),), dtype=float)
            for j, item in enumerate(p):
                query_points = query_points.at[..., j].set(item)
    else:
        check_arraylike("_ndim_coords_from_arrays", query_points)
        query_points = asarray(query_points)  # SciPy: asanyarray(query_points)

        if query_points.ndim == 1:
            query_points = query_points.reshape(-1, ndim)

    '''
    if len(query_points[-1]) != len(coordinates[-1]):
        raise ValueError("the requested sample coordinates query_points have dimension"
                    f" {len(query_points[1])}, but this trilinearInterpolator has"
                    f" dimension {ndim}")
    '''

    # find relevant edges between which query_points are situated
    indices = []
    # compute distance to lower edge in unity units
    norm_distances = []
    # check for out of bounds query_points
    out_of_bounds = zeros((len(query_points[-1]),), dtype = bool)
    # iterate through dimensions
    for x, g in zip(query_points, tuple(asarray(p) for p in coordinates)):
        i = searchsorted(g, x) - 1
        i = where(i < 0, 0, i)
        i = where(i > g.size - 2, g.size - 2, i)
        indices.append(i)

        norm_distances.append((x - g[i]) / (g[i + 1] - g[i]))

        out_of_bounds += x < g[0]
        out_of_bounds += x > g[-1]

    # slice for broadcasting over trailing dimensions in values
    vslice = (slice(None),) + (None,) * (values.ndim - len(indices))

    # find relevant values
    # each i and i+1 represents a edge
    edges = product(*[[i, i + 1] for i in indices])
    results = zeros((len(query_points[-1]),))
    for edge_indices in edges:
        weight = asarray(1.)
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= where(ei == i, 1 - yi, yi)
        #results = results.at[:].set(results + values[edge_indices] * weight[vslice])
        results += values[edge_indices] * weight[vslice]

    '''
    if fill_value is not None:
        bc_shp = results.shape[:1] + (1,) * (results.ndim - 1)
        results = where(out_of_bounds.reshape(bc_shp), fill_value, results)
    '''

    return 0.0#results.reshape(query_points.shape[:-1] + results.shape[ndim:])

'''
def trilinearInterpolator(values, location):
    grid = np.stack(
        (
            np.meshgrid(indexing = 'ij', copy = True,
                [np.lower(location[0]), np.upper(location[0])],
                [np.lower(location[1]), np.upper(location[1])],
                [np.lower(location[2]), np.upper(location[2])])
        ),

        axis = -1
    )

    weights = np.array(
        (
            (location[0] - grid[0, 0, 0, 0]) / (grid[1, 0, 0, 0] - grid[0, 0, 0, 0]),
            (location[1] - grid[0, 0, 0, 1]) / (grid[0, 1, 0, 1] - grid[0, 0, 0, 1]),
            (location[2] - grid[0, 0, 0, 2]) / (grid[0, 0, 1, 2] - grid[0, 0, 0, 2])
        )
    )

    weights = jnp.array(
        [
            (location[0, :] - np.lower(location[0, :])) / (np.upper(location[0, :]) - np.lower(location[0, :])),
            (location[1, :] - np.lower(location[1, :])) / (np.upper(location[1, :]) - np.lower(location[1, :])),
            (location[2, :] - np.lower(location[2, :])) / (np.upper(location[2, :]) - np.lower(location[2, :]))
        ]
    )

    # vectorize instead of for loop!!!!!
    results = jnp.zeros(len(location[0]))
    for i in range(len(location[0])):
        for j in range(8):
            buffer = 0

            indices = list(binary(j)[7:])
            print(indices)

            if indices[0] == 1:
                x = weights[0]
            else:
                x = 1 - weights[0]

            if indices[1] == 1:
                y = weights[1]
            else:
                y = 1 - weights[1]

            if indices[2] == 1:
                z = weights[2]
            else:
                z = 1 - weights[2]

            buffer += values[0, 0, 0] * x * y * z

        results = results.at[i].set(buffer)

    return 1.0
    #else:
    #    print("Expected 3D array, defaulting to jax.scipy.interpolate.RegularGridInterpolator incase this is an issue with our custom trilinear interpolator.")
    #    # default to scipy code
'''