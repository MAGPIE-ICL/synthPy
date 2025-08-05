import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sys import getsizeof as getsizeof_default

def random_array(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.rand(length)

def random_array_n(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.randn(length)

def random_inv_pow_array(power, length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.power(power, length)

def count_nans(matrix, axes = [0, 2]):
    for i in axes:
        x = r2[0, :]
        y = r2[2, :]

        print("\nrf size expected: (", len(x), ", ", len(y), ")", sep='')
        mask = ~jnp.isnan(x) & ~jnp.isnan(y)

        x = x[mask]
        y = y[mask]

        print("rf after clearing nan's: (", len(x), ", ", len(y), ")", sep='')

def getsizeof(object):
    return mem_conversion(getsizeof_default(object))

def mem_conversion(mem_size):
    count = 0
    while mem_size > 1024:
        mem_size /= 1024
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

    return str(mem_size) + " " + unit

# stored here for later in case needed - check first, this was just copied from stackoverflow I have no idea if it works yet
def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
      a = num[:-2-(not dec)]       # integer part
      b = int(num[-2-(not dec)])+1 # decimal part
      return float(a)+b**(-dec+1) if a and b == 10 else float(a+str(b))
    return float(num[:-1])

def dalloc(var):
    try:
        del var
        #print(f'del {var}')
    except:
        var = None
        #print(f'set {var = }')

def domain_estimate(dim):
    return dim[0] * dim[1] * dim[2] * 4

def add_integer_postfix(int):
    if int // 10 == 1:
        postfix = "th"
    else:
        digit = int % 10

        if digit == 1:
            postfix = "st"
        elif digit == 2:
            postfix = "nd"
        elif digit == 3:
            postfix = "rd"
        else:
            postfix = "th"

    return str(int) + postfix

def find_sig_n(x, n):
    '''
    ValueError: Non-hashable static arguments are not supported. An error occurred while trying to hash an object of type <class 'jaxlib.xla_extension.ArrayImpl'>, 5. The error was:
    TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'

    using jnp.int32 instead of regular int conversion causes issues - why does jnp.round not support standard jax data types?
    '''

    return n - int(jnp.floor(jnp.log10(abs(x)))) - 1

def round_to_n(x, n):
    return jnp.round(x, find_sig_n(x, n))

    ##
    ## package-wide code for interpolations
    ##

from itertools import product

from jax._src import dtypes
from jax._src.numpy import (asarray, broadcast_arrays, empty, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

# is overhead better using jnp.clip or a vectorised(?) if statement?
# if we can sort out my original solution to this - clip would not be necessary at all
# can we speed this up even further?

def RegularGridInterpolator(points, values, xi, method = "linear", bounds_error = False, fill_value = 0.0):
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

def baseRayPlot(x, y, *, scaling = 1, bin_scale = 1, pix_x = 3448, pix_y = 2574, Lx = 18, Ly = 13.5):
    print("\nrf size expected: (", len(x), ", ", len(y), ")", sep='')

    # means that jnp.isnan(a) returns True when a is not Nan
    # ensures that x & y are the same length, if output of either is Nan then will not try to render ray in histogram
    mask = ~jnp.isnan(x) & ~jnp.isnan(y)

    x = x[mask]
    y = y[mask]

    print("rf after clearing nan's: (", len(x), ", ", len(y), ")", sep='')

    H, xedges, yedges = jnp.histogram2d(x, y, bins=[pix_x // bin_scale, pix_y // bin_scale], range=[[-Lx / 2, Lx / 2],[-Ly / 2, Ly / 2]])
    H = H.T

    plt.imshow(H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

def heat_plot(x, y, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, Lx = 18, Ly = 13.5):
    #fig, axis = plt.subplots(1, figsize = (20,5))

    H,_,_,im1 = plt.hist2d(x, y, bins = (pix_x, pix_y), cmap = "turbo")

    #plt.imshow(H, cmap = 'turbo', interpolation = 'nearest', clim = (0, 10))
    #im1.set_clim(0, 10)

    plt.colorbar(im1)
    plt.grid(False)

    #axis.set_xlabel("x (mm)")
    #axis.set_ylabel("z (mm)")
    #axis.set_xlim([-9, 9])
    #axis.set_ylim([-6.75, 6.75])