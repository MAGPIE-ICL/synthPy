import numpy as np
import jax.numpy as jnp

from sys import getsizeof as getsizeof_default

def random_array(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.rand(length)

def random_array_n(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.randn(length)

def count_nans(matrix, axes = [0, 2]):
    for i in axes:
        x = r2[0, :]
        y = r2[2, :]

        print("\nrf size expected: (", len(x), ", ", len(y), ")", sep='')
        mask = ~np.isnan(x) & ~np.isnan(y)

        x = x[mask]
        y = y[mask]

def getsizeof(object):
    memory_attributed = getsizeof_default(object)

    count = 0
    while memory_attributed > 1024:
        memory_attributed /= 1024
        count += 1

    if count == 0:
        unit = 'B'
    elif count == 1:
        unit = 'KB'
    elif count == 2:
        unit = 'MB'
    elif count == 3:
        unit = 'GB'
    else:
        unit = "-> didn't resolve unit, this value is way too big something is very wrong."

    return str(memory_attributed) + unit

class trilinearInterpolator:
    """
    Interpolate points on a regular rectangular grid.

    JAX implementation of a custom interpolator to decrease memory overhead in our use case

    Args:
        points: length-N sequence of arrays specifying the grid coordinates.
        values: N-dimensional array specifying the grid values.
        method: interpolation method, either ``"linear"`` or ``"nearest"``.
        bounds_error: not implemented by JAX
        fill_value: value returned for points outside the grid, defaults to NaN.

    Returns:
        interpolator: callable interpolation object.

    Examples:
        >>> points = (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
        >>> values = jnp.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> interpolate = RegularGridInterpolator(points, values, method='linear')

        >>> query_points = jnp.array([[1.5, 4.5], [2.2, 5.8]])
        >>> interpolate(query_points)
        Array([30., 64.], dtype=float32)
    """
    # Initially derived and customised from jax's implementation (our's was causing solver errors)
    # This was based on SciPy's implementation which in turn is originally based on an
    # implementation by Johannes Buchner

    def __init__(self, points, values, method="linear", bounds_error=False,fill_value=np.nan):
        if method not in ("linear", "nearest"):
            raise ValueError(f"method {method!r} is not defined")
        self.method = method
        self.bounds_error = bounds_error
        if self.bounds_error:
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
        self.fill_value = fill_value

        # TODO: assert sanity of `points` similar to SciPy but in a JIT-able way
        check_arraylike("RegularGridInterpolator", *points)
        self.grid = tuple(asarray(p) for p in points)
        self.values = values

    def __call__(self, xi, method=None):
        method = self.method if method is None else method
        if method not in ("linear", "nearest"):
            raise ValueError(f"method {method!r} is not defined")

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("the requested sample points xi have dimension"
                       f" {xi.shape[1]}, but this RegularGridInterpolator has"
                       f" dimension {ndim}")

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances)
        else:
            raise AssertionError("method must be bound")
        if not self.bounds_error and self.fill_value is not None:
            bc_shp = result.shape[:1] + (1,) * (result.ndim - 1)
            result = where(out_of_bounds.reshape(bc_shp), self.fill_value, result)

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = asarray(0.)
        for edge_indices in edges:
            weight = asarray(1.)
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= where(ei == i, 1 - yi, yi)
            values += self.values[edge_indices] * weight[vslice]
        return values

    def _evaluate_nearest(self, indices, norm_distances):
        idx_res = [
            where(yi <= .5, i, i + 1) for i, yi in zip(indices, norm_distances)
        ]

        return self.values[tuple(idx_res)]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = zeros((xi.shape[1],), dtype=bool)
        # iterate through dimensions
        for x, g in zip(xi, self.grid):
            i = searchsorted(g, x) - 1
            i = where(i < 0, 0, i)
            i = where(i > g.size - 2, g.size - 2, i)
            indices.append(i)
            norm_distances.append((x - g[i]) / (g[i + 1] - g[i]))
            if not self.bounds_error:
                out_of_bounds += x < g[0]
                out_of_bounds += x > g[-1]
        return indices, norm_distances, out_of_bounds

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