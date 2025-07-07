import numpy as np

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