import numpy as np

def random_array(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.rand(length)

def random_array_n(length, seed = False):
    if seed:
        np.random.seed(0)

    return np.random.randn(length)
