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

#dndx = -0.5 * c ** 2 * np.gradient(self.ne_nc, self.ScalarDomain.x, axis = 0)
#dndx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), dndx, bounds_error = False, fill_value = 0.0)
#del dndx

def trilinearInterpolator(coords, values, location):
    #print(len(coords[0, :]))
    #print(len(coords[:, 0]))
    print(values)
    print(location)

    if len(coords[:, 0]) == 3:
        '''
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
        '''

        weights = np.array(
            (
                (location[0, :] - np.lower(location[0, :])) / (np.upper(location[0, :]) - np.lower(location[0, :])),
                (location[1, :] - np.lower(location[1, :])) / (np.upper(location[1, :]) - np.lower(location[1, :])),
                (location[2, :] - np.lower(location[2, :])) / (np.upper(location[2, :]) - np.lower(location[2, :]))
            )
        )

        # vectorize instead of for loop!!!!!
        results = np.zeros(len(location[0]))
        for i in range(len(location[0])):
            for j in range(8):
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

                results = results.at[i].set(results[i] + values[indices] * x * y * z)

        return results
    else:
        print("Expected 3D array, defaulting to np.scipy.interpolate.RegularGridInterpolator incase this is an issue with our custom trilinear interpolator.")
        # default to scipy code