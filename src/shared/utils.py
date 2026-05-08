import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

from sys import getsizeof as getsizeof_default

def random_array(length, seed = None):
    """
    Generates an array of random floats in [0.0, 1.0).
    
    :param length: Size of the array
    :type length: int
    
    :param seed: Random seed
    :type seed: int, default: None
    
    :return: Random array
    :rtype: np.array
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.rand(length)

def random_array_n(length, seed = None):
    """
    Generates an array of random floats from the standard normal distribution.
    
    :param length: Size of the array
    :type length: int
    
    :param seed: Random seed
    :type seed: int, default: None
    
    :return: Random array
    :rtype: np.array
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(length)

def random_inv_pow_array(power, length, seed = None):
    """
    Generates an array of random floats from a power law distribution.
    
    :param power: Power law parameter
    :type power: float
    
    :param length: Size of the array
    :type length: int
    
    :param seed: Random seed
    :type seed: int, default: None
    
    :return: Random array
    :rtype: np.array
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.power(power, length)

def count_nans(matrix, *, axes = [0, 2], ret = False):
    """
    Counts the number of non-NaN entries in the specified axes of a matrix.
    
    :param matrix: Input matrix
    :type matrix: array-like
    
    :param axes: Axes to check for NaNs
    :type axes: list, default: [0, 2]
    
    :param ret: Whether to return the filtered array
    :type ret: bool, default: False
    
    :return: Filtered sub-arrays if ret is True, else None
    :rtype: tuple or None
    """
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
    """
    Gets the memory footprint of an object in a human-readable format.
    
    :param object: Object to check
    :type object: any
    
    :return: Formatted memory string
    :rtype: str
    """
    return mem_conversion(getsizeof_default(object))

def mem_conversion(mem_size):
    """
    Converts a byte count into a human-readable string (B, KB, MB, GB).
    
    :param mem_size: Size in bytes
    :type mem_size: float
    
    :return: Formatted size string
    :rtype: str
    """
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
    """
    Rounds a number properly (half round up) to a given decimal place.
    
    :param num: Number to round
    :type num: float
    
    :param dec: Decimal places
    :type dec: int, default: 0
    
    :return: Rounded number
    :rtype: float
    """
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
      a = num[:-2-(not dec)]       # integer part
      b = int(num[-2-(not dec)])+1 # decimal part
      return float(a)+b**(-dec+1) if a and b == 10 else float(a+str(b))
    return float(num[:-1])

def dalloc(var):
    """
    Safely deallocates a variable from memory.
    
    :param var: Variable to delete
    :type var: any
    
    :return: No return
    :rtype: None
    """
    try:
        del var
        #print(f'del {var}')
    except:
        var = None
        #print(f'set {var = }')

def domain_estimate(x_n, y_n, z_n, *, enable_x64 = False):
    """
    Estimates the memory usage for a 3D grid with the given dimensions.
    
    :param x_n: Number of elements in x
    :type x_n: int
    
    :param y_n: Number of elements in y
    :type y_n: int
    
    :param z_n: Number of elements in z
    :type z_n: int
    
    :param enable_x64: Whether to use 64-bit precision (8 bytes) instead of 32-bit
    :type enable_x64: bool, default: False
    
    :return: Memory size in bytes
    :rtype: np.int64
    """
    if enable_x64:
        conv = 8
    else:
        conv = 4

    return np.int64(x_n * y_n * z_n * conv)
    # np.int64 not jnp.int64 as jnp arrays are limited to 32 bits by default
    #return np.int64(dims[0] * dims[1] * dims[2] * conv)

def add_integer_postfix(int):
    """
    Adds a proper ordinal suffix (st, nd, rd, th) to an integer string.
    
    :param int: Integer value
    :type int: int
    
    :return: Integer string with suffix
    :rtype: str
    """
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
    """
    Finds the number of decimal places needed to round `x` to `n` significant figures.
    
    :param x: Number to evaluate
    :type x: float
    
    :param n: Target number of significant figures
    :type n: int
    
    :return: Decimal places to round to
    :rtype: int
    """
    '''
    ValueError: Non-hashable static arguments are not supported. An error occurred while trying to hash an object of type <class 'jaxlib.xla_extension.ArrayImpl'>, 5. The error was:
    TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'

    using jnp.int32 instead of regular int conversion causes issues - why does jnp.round not support standard jax data types?
    '''

    return n - int(jnp.floor(jnp.log10(abs(x)))) - 1

def round_to_n(x, n):
    """
    Rounds a number to a specified number of significant figures.
    
    :param x: Number to round
    :type x: float
    
    :param n: Number of significant figures
    :type n: int
    
    :return: Rounded number
    :rtype: float
    """
    return jnp.round(x, find_sig_n(x, n))

    ##
    ## package-wide code for interpolations
    ##

def baseRayPlot(x, y, *, scaling = 1, bin_scale = 1, pix_x = 3448, pix_y = 2574, Lx = 18, Ly = 13.5):
    """
    Generates a 2D histogram plot for ray locations.
    
    :param x: Ray x-coordinates
    :type x: array-like
    
    :param y: Ray y-coordinates
    :type y: array-like
    
    :param scaling: Scaling factor
    :type scaling: int, default: 1
    
    :param bin_scale: Bin scale divisor
    :type bin_scale: int, default: 1
    
    :param pix_x: Number of x pixels
    :type pix_x: int, default: 3448
    
    :param pix_y: Number of y pixels
    :type pix_y: int, default: 2574
    
    :param Lx: Physical width
    :type Lx: float, default: 18
    
    :param Ly: Physical height
    :type Ly: float, default: 13.5
    
    :return: No return
    :rtype: None
    """
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
    """
    Generates a heatmap/2D histogram plot for ray locations.
    
    :param x: Ray x-coordinates
    :type x: array-like
    
    :param y: Ray y-coordinates
    :type y: array-like
    
    :param bin_scale: Bin scale divisor
    :type bin_scale: int, default: 1
    
    :param pix_x: Number of x pixels
    :type pix_x: int, default: 3448
    
    :param pix_y: Number of y pixels
    :type pix_y: int, default: 2574
    
    :param Lx: Physical width
    :type Lx: float, default: 18
    
    :param Ly: Physical height
    :type Ly: float, default: 13.5
    
    :return: No return
    :rtype: None
    """
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

def memory_report(running_device = None, memory_limit = None):
    """
    Generates a dictionary with memory usage statistics for the running device.
    
    :param running_device: The device platform ('cpu', 'gpu', etc.)
    :type running_device: str, default: None
    
    :param memory_limit: Optional memory limit override
    :type memory_limit: float, default: None
    
    :return: Memory statistics dictionary
    :rtype: dict
    """
    if running_device is None:
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

    results = {
        'device': running_device,
        'total_raw': total,
        'total': mem_conversion(total),
        'free_raw': free,
        'free': mem_conversion(free),
        'used_raw': used,
        'used': mem_conversion(used)
    }

    if memory_limit is not None:
        memory_limit *= 1024
        if memory_limit < results['total_raw']:
            results['total_raw'] = memory_limit
            results['total'] = mem_conversion(results['total_raw'])

            # could be referencing memory_stats in shared_utils for results['total_raw'] instead -> memory_stats()['total_raw']
            # turns out memory_stats in shared_utils was renamed to memory_report, this is memory_report() function so referencing internally
            # naming lead to confusion...
            results['free_raw'] = results['total_raw'] - results['used_raw']
            results['free'] = mem_conversion(results['free_raw'])

    return results

generic_valid_types = (int, np.int32, np.int64, jnp.int32, jnp.int64, float, np.float32, np.float64, jnp.float32, jnp.float64)