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

class colour:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

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

def trilinearInterpolator(x, y, z, values, query_points, *, fill_value = jnp.nan):
    """
    Trilinear interpolation on a 3D regular grid.

    Assumes:
        - coordinates = (x, y, z) where each is 1D
        - values.shape == (len(x), len(y), len(z))
        - query_points.shape == (N, 3)
    """

    values = jnp.asarray(values)
    query_points = jnp.asarray(query_points.T)

    def get_indices_and_weights(coord_grid, points):
        idx = jnp.searchsorted(coord_grid, points, side = 'right') - 1
        idx = jnp.clip(idx, 0, len(coord_grid) - 2)

        x0 = coord_grid[idx]
        return idx, (points - x0) / (coord_grid[idx + 1] - x0)

    ix, wx = get_indices_and_weights(x, query_points[:, 0])
    iy, wy = get_indices_and_weights(y, query_points[:, 1])
    iz, wz = get_indices_and_weights(z, query_points[:, 2])

    def get_val(dx, dy, dz):
        return values[ix + dx, iy + dy, iz + dz]

    results = (
        get_val(0, 0, 0) * (1 - wx) * (1 - wy) * (1 - wz) +
        get_val(1, 0, 0) * wx       * (1 - wy) * (1 - wz) +
        get_val(0, 1, 0) * (1 - wx) * wy       * (1 - wz) +
        get_val(0, 0, 1) * (1 - wx) * (1 - wy) * wz +
        get_val(1, 1, 0) * wx       * wy       * (1 - wz) +
        get_val(1, 0, 1) * wx       * (1 - wy) * wz +
        get_val(0, 1, 1) * (1 - wx) * wy       * wz +
        get_val(1, 1, 1) * wx       * wy       * wz
    )

    # Check out-of-bounds
    oob = (
        (query_points[:, 0] < x[0]) | (query_points[:, 0] > x[-1]) |
        (query_points[:, 1] < y[0]) | (query_points[:, 1] > y[-1]) |
        (query_points[:, 2] < z[0]) | (query_points[:, 2] > z[-1])
    )

    return jnp.where(oob, fill_value, results)