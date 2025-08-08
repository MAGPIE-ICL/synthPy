import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sys import getsizeof as getsizeof_default

def random_array(length, seed = None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(length)

def random_array_n(length, seed = None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(length)

def random_inv_pow_array(power, length, seed = None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.power(power, length)

def count_nans(matrix, *, axes = [0, 2], ret = False):
    matrix = jnp.asarray(matrix)

    dim = len(axes)
    stats = np.zeros((dim, 2))

    mask = True
    for i in range(dim):
        arr = matrix[i]
        mask = mask & ~jnp.isnan(arr)

    for i in range(dim):
        stats[i, 0] = len(matrix[i])
        stats[i, 1] = len(matrix[i][mask])

    print("\nrf size expected:", stats[:, 0])
    print("rf after clearing nan's:", stats[:, 1])

    '''
    if ret:
        matrix = matrix.at[:, :].set(matrix[:, mask])

        # jnp.split turns the matrix into M rows of shape (1, N)
        # jnp.squeeze forces each row to shape (N,)
        return tuple(jnp.squeeze(r, axis=0) for r in jnp.split(matrix, matrix.shape[0], axis = 0))
    '''

    if ret:
        #matrix = matrix.at[:, :].set(matrix[:, mask]) - can't do this, jax arrays are immutable with respect to shape
        return matrix[0, mask], matrix[2, mask]

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

def domain_estimate(x_n, y_n, z_n, *, enable_x64 = False):
    if enable_x64:
        conv = 8
    else:
        conv = 4

    return np.int64(x_n * y_n * z_n * conv)

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

def memory_report():
    from jax.lib import xla_bridge
    running_device = xla_bridge.get_backend().platform

    if running_device == 'cpu':
        from psutil import virtual_memory

        info = virtual_memory()

        free = info.available
    elif running_device == 'gpu':
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        nvmlInit()

        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)

        free = info.free
    elif running_device == 'tpu':
        free_mem = None
    else:
        assert "\nNo suitable device detected when checking ram/vram available."

    total = info.total
    used = info.used

    return {
        'device': running_device,
        'total_raw': total,
        'total': mem_conversion(total),
        'free_raw': free,
        'free': mem_conversion(free),
        'used_raw': used,
        'used': mem_conversion(used)
    }