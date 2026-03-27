import sys
import os

import jax

class colour:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

import numpy as np
import jax.numpy as jnp

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

def dalloc(var):
    try:
        del var
        #print(f'del {var}')
    except:
        var = None
        #print(f'set {var = }')

def round_to_n(x, n):
    return jnp.round(x, n - int(jnp.floor(jnp.log10(abs(x)))) - 1)

generic_valid_types = (int, np.int32, np.int64, jnp.int32, jnp.int64, float, np.float32, np.float64, jnp.float32, jnp.float64)

class Beam:
    def __init__(self, Np, beam_size, divergence, ne_extent, *, probing_direction = 'z', beam_type = 'circular', seeded = False, offset = None):
        self.Np = np.int64(Np)
        self.beam_size = beam_size
        self.divergence = divergence
        self.ne_extent = ne_extent
        self.probing_direction = probing_direction
        self.beam_type = beam_type

        if seeded:
            self.seed = 42
        else:
            self.seed = None

        self.offset = jnp.asarray(offset)

        # calls actual initialisation of beam automatically, first function just initialises variables
        # forces ne_extent to negative when passed to init_beam(... ne_extent < 0 ...)
        Beam.init_beam(self, -self.ne_extent, self.seed, self.offset) # [x if x < 0 else -x for x in jnp.array(ne_extent)]

    def init_beam(self, ne_extent, seed, offset):
        from scipy.constants import c

        s0 = jnp.zeros((9, self.Np))
        if(self.beam_type == 'circular'):
            assert isinstance(self.beam_size, generic_valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly within a circle
            t  = 2 * jnp.pi * random_array(self.Np, seed) #polar angle of position

            # inversely weights probability with radius so that positions are uniformly distributed
            u = random_inv_pow_array(2, self.Np, seed) # radial coordinate of position

            # angle
            ϕ = jnp.pi * random_array(self.Np) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[2, :].set(self.beam_size * u * jnp.sin(t))
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(self.beam_size * u * jnp.sin(t))
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                #s0 = s0.at[3, :].set(c * jnp.sin(1.373))
                #s0 = s0.at[4, :].set(c * jnp.cos(1.373))
                #s0 = s0.at[5, :].set(0)

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(self.beam_size * u * jnp.sin(t))
        elif(self.beam_type == 'square'):
            assert isinstance(self.beam_size, generic_valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(self.beam_size * u)
                s0 = s0.at[2, :].set(self.beam_size * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u)
                s0 = s0.at[1, :].set(self.beam_size * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(self.beam_size * t)
        elif(self.beam_type == 'rectangular'):
            size_dim = len(self.beam_size)
            assert size_dim == 2, colour.BOLD + "\nERROR: " + colour.END + "Must pass a list of length 2 to initialise a rectangular beam," + str(size_dim) + "item was passed."

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            beam_size_1 = self.beam_size[0] #m
            beam_size_2 = self.beam_size[1] #m

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        elif(self.beam_type == 'linear'):
            assert isinstance(self.beam_size, generic_valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly along a line - probing direction is defaulted z, solved in x,z plane
            t  = 2 * random_array(self.Np, seed) - 1.0
            # angle
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            # Initial velocity
            s0 = s0.at[3, :].set(c * jnp.sin(χ))
            s0 = s0.at[4, :].set(0.0)
            s0 = s0.at[5, :].set(c * jnp.cos(χ))
            # Initial position
            s0 = s0.at[0, :].set(self.beam_size * t)
            s0 = s0.at[1, :].set(0.0)
            s0 = s0.at[2, :].set(ne_extent)
        elif(self.beam_type == 'even'): # evenly distributed circular ray using concentric discs
            # number of concentric discs and points
            num_of_circles = (-1 + jnp.sqrt(1 + 8 * (self.Np // 6))) / 2 
            self.Np = 3 * (num_of_circles + 1) * num_of_circles + 1 

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            # position, uniformly within a circle
            t = [0]
            u = [0]

            # vectorise?
            for i in range(1, num_of_circles + 1): # for every disc
                for j in range(0, i * 6): # for every point in the disc
                    u.append(i / num_of_circles)
                    t.append(j * 2 * jnp.pi / (i * 6))  
        elif(self.beam_type == 'rect_trackers'):
            size_dim = len(self.beam_size)
            assert size_dim == 2, colour.BOLD + "\nERROR: " + colour.END + "Must pass a list of length 2 to initialise a rectangular beam," + str(size_dim) + "item was passed."

            # Randomly choose N_trackers indices to mark as tracking particles
            # tracker_indices = jnp.random.choice(self.Np, N_trackers, replace=False)

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            beam_size_1 = self.beam_size[0] #m
            beam_size_2 = self.beam_size[1] #m

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'y'):
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        else:
            print("\nself.beam_type unrecognised! Accepted args: circular, square, rectangular, linear, even, rect_trackers.")

        if offset is not None:
            s0 = s0.at[0, :].set(s0[0, :] + offset[0, None])
            s0 = s0.at[1, :].set(s0[1, :] + offset[1, None])
            s0 = s0.at[2, :].set(s0[2, :] + offset[2, None])

        del t
        del u
        del ϕ
        del χ

        # Initialise amplitude, phase and polarisation
        s0 = s0.at[6, :].set(1.0)

        self.s0 = s0
        #self.rf = s0

        del s0

def memory_report(running_device = None, memory_limit = None):
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

            results['free_raw'] = memory_stats['total_raw'] - results['used_raw']
            results['free'] = mem_conversion(results['free_raw'])

    return results

import equinox as eqx

from math import ceil
from math import floor

class ScalarDomain(eqx.Module):
    s: jnp.float32
    s1: jnp.float32
    s2: jnp.float32

    Ly: jnp.float32

    ne_0: jnp.float32

    probing_direction: str

    ne_type: str

    leeway_factor: jnp.float32

    x_length: jnp.int32
    y_length: jnp.int32
    z_length: jnp.int32

    lengths: jax.Array

    x_n: jnp.int32
    y_n: jnp.int32
    z_n: jnp.int32

    dims: jax.Array

    x: jax.Array
    y: jax.Array
    z: jax.Array

    coordinates: jax.Array

    XX: jax.Array
    YY: jax.Array
    ZZ: jax.Array

    ne: jax.Array
  
    region_count: jnp.int32

    coord_backup: jax.Array
    future_dims: jax.Array

    memory_limit: np.int64

    Np_total: np.int64
    ray_batch_count: np.int64

    x_offset: jnp.int32
    y_offset: jnp.int32
    z_offset: jnp.int32

    zeropoint: jax.Array

    def __init__(self, lengths, dims, *, ne_type = None, probing_direction = 'z', auto_batching = True, iteration = 1, region_count = 1, leeway_factor = None, coord_backup = None, future_dims = None, Np = None,
        s = None, s1 = None, s2 = None, Ly = None, ne_0 = None, ne = None, memory_limit = None, zeropoint = None):

        # initalise
        self.s = s
        del s

        self.s1 = s1
        del s1

        self.s2 = s2
        del s2

        self.Ly = Ly
        del Ly

        self.ne_0 = ne_0
        del ne_0

        self.ne = ne
        del ne

        self.probing_direction = probing_direction
        del probing_direction

        self.ne_type = ne_type
        del ne_type

        self.memory_limit = memory_limit
        del memory_limit

        # working with 10% leeway in estimate for now
        if leeway_factor is not None:
            self.leeway_factor = leeway_factor
        else:
            # set to 1.1 by default, gives 10% leeway in prediction
            self.leeway_factor = 1.1

        if Np is not None:
            self.Np_total = np.int64(Np)
        else:
            self.Np_total = None

        self.ray_batch_count = 1

        if zeropoint is not None:
            # if 1 length given, assumes all are the same
            if isinstance(zeropoint, generic_valid_types):
                self.x_offset, self.y_offset, self.z_offset = zeropoint, zeropoint, zeropoint
                self.zeropoint = jnp.array([zeropoint, zeropoint, zeropoint])
            # if array given, checks len = 3 and assigns accordingly
            else:
                self.zeropoint = jnp.array(zeropoint)
                if self.zeropoint.shape != (3,):
                    raise Exception('zeropoint must have len = 3: (x,y,z)')

                self.x_offset, self.y_offset, self.z_offset = self.zeropoint[0], self.zeropoint[1], self.zeropoint[2]
        else:
            self.x_offset, self.y_offset, self.z_offset = 0, 0, 0
            self.zeropoint = jnp.array([0, 0, 0])

        # if 1 length given, assumes all are the same
        if isinstance(lengths, generic_valid_types):
            self.x_length, self.y_length, self.z_length = lengths, lengths, lengths
            self.lengths = jnp.array([lengths, lengths, lengths])
        # if array given, checks len = 3 and assigns accordingly
        else:
            self.lengths = jnp.array(lengths)
            if self.lengths.shape != (3,):
                raise Exception('lengths must have len = 3: (x,y,z)')

            self.x_length, self.y_length, self.z_length = self.lengths[0], self.lengths[1], self.lengths[2]

        del lengths

        #likewise for dims
        #self.dims = dims
        if isinstance(dims, generic_valid_types):
            self.x_n, self.y_n, self.z_n = dims, dims, dims
            self.dims = jnp.array([dims, dims, dims])
        else:
            self.dims = jnp.array(dims)
            if self.dims.shape != (3,):
                raise Exception('n must have len = 3: (x_n, y_n, z_n)')

            self.x_n, self.y_n, self.z_n = self.dims[0], self.dims[1], self.dims[2]

        del dims

        # changed function to pass to np.int64 to prevent overflow - this was causing the negatives
        # --> (exactly 0 in the case of a 1024^3 domain as it is right on the limit)

        predicted_domain_allocation = np.int64(self.x_n * self.y_n * self.z_n * 4)
        #if enable_x64:
        #    predicted_domain_allocation *= 2
        print("Predicted size in memory of domain:", mem_conversion(predicted_domain_allocation))

        if iteration == 1 and auto_batching:
            memory_stats = memory_report(memory_limit = self.memory_limit)

            print("\nMemory prior to domain creation:")
            print(f" - total : {memory_stats['total']}")
            print(f" - free  : {memory_stats['free']}")
            print(f" - used  : {memory_stats['used']}")

            ###
            ### Need to work out the max allocation at any point and that estimated size
            ###

            # 2 for ne and ne_nc in calc_dndr(...) before ne is deleted
            # at peak mem usage ne should have been deleted, therefore this contributes only 1 domain
            # +1 for ne_interp
            # +2 for the 2 sequentially repeated domain sized allocations in dndr(...)

            # 1 for ne
            # +1 for sequential interps (dndx, dndy, dndz)
            # no more need for domain allocation in dndr as now functionally interpolated
            allocation_count = 2

            # compare to max allocation in domain setup and return the greatest
            if self.ne_type == "test_null" or self.ne_type == "test_slab" or self.ne_type == "test_B":
                allocation_count = max(allocation_count, 2)
            elif self.ne_type == "test_linear_cos" or self.ne_type == "test_exponential_cos" or self.ne_type is None:
                allocation_count = max(allocation_count, 3)
            elif self.ne_type == "import" or self.ne_type == "quad_trough_test":
                allocation_count = max(allocation_count, 1)
            else:
                raise AssertionError("\nNo valid profile detected! Ensure passed name is correct or call yourself.")

            print("")
            if self.Np_total is not None:
                import simulator.beam as ray_test_case

                test_beam = ray_test_case.Beam(1, 1, 1, 1)
                single_ray = test_beam.s0 # just initialises 1 ray of any variety
                del test_beam

                ray_memory_raw = np.float64(getsizeof_default(single_ray) * self.Np_total) * self.leeway_factor
                del single_ray

                print("Est. ray size in memory:", mem_conversion(ray_memory_raw))

            estimate_limit = np.float64(predicted_domain_allocation * allocation_count * self.leeway_factor)
            print("Est. domain memory limit: {}".format(mem_conversion(estimate_limit)))
            print(" --> inc. +{}% variance margin".format(jnp.float32((self.leeway_factor - 1) * 100)))

            limiting_value = estimate_limit
            if self.Np_total is not None:
                limiting_value += ray_memory_raw * self.leeway_factor
                print("Total estimated maximum: {}".format(mem_conversion(limiting_value)))

            # when jnp.float32 is not used, will cause overflow error if 64 bit floats are not enabled
            if limiting_value > np.float64(memory_stats['free_raw']):
                if self.ne is None:
                    if self.Np_total is None:
                        print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                    else:
                        print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN + RAYS CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                        self.ray_batch_count = np.int64(ceil(ray_memory_raw * self.leeway_factor / np.float64(memory_stats['free_raw'])))
                    print(" --> Auto-batching domain based on memory available and domain size estimate...")

                    ##
                    ## Used backed up information to re-assign to ScalarDomain in propagator
                    ## Then call generate_electron_density_profile(...) and re-do calculations with end of prior domain
                    ##

                    #self.region_count = ceil((limiting_value - ray_memory_raw / self.ray_batch_count) / np.float64(memory_stats['free_raw']))
                    self.region_count = ceil(np.float64(predicted_domain_allocation * allocation_count) / (np.float64(memory_stats['free_raw']) - ceil(ray_memory_raw / self.ray_batch_count)))

                    self.coord_backup = jnp.float32(jnp.linspace(
                    -self.lengths[['x', 'y', 'z'].index(self.probing_direction)] / 2,
                        self.lengths[['x', 'y', 'z'].index(self.probing_direction)] / 2,
                        self.dims[['x', 'y', 'z'].index(self.probing_direction)]
                    ))

                    dim_per_region = self.dims[['x', 'y', 'z'].index(self.probing_direction)] // self.region_count
                    self.future_dims = jnp.concatenate([
                        jnp.expand_dims(0, axis = 0), jnp.array([dim_per_region] * self.region_count),
                        jnp.array([self.dims[['x', 'y', 'z'].index(self.probing_direction)] - dim_per_region * self.region_count])
                    ])

                    print(" --> Batching calculation completed. Domain will be split into " + str(self.region_count) + " regions with " + str(dim_per_region) + " dims per region.")
                    print(colour.BOLD + "\nWARNING:" + colour.END + " This functionality will cause the solver to run slower due to domain regeneration, for optimal performance, increase the memory available to this program.")
                    if self.Np_total is not None:
                        print(" --> The domain is batched with the goal of minimising ray batching. Ray batches introduce sequantiality which reduces speed.")
                else:
                    self.region_count = 1

                    print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN + RAYS CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                    self.ray_batch_count = np.int64(ceil(ray_memory_raw * self.leeway_factor / np.float64(memory_stats['free_raw'])))
                    print(" --> Auto-batching rays based on memory available and domain size estimate...")
                    print(" --> Can't auto-batch the domain as it is imported (limitation will be resolved in the future)")
            else:
                self.region_count = 1
        else:
            self.region_count = region_count

        if self.region_count == 1:
            self.coord_backup = None
            self.future_dims = None

            # define coordinate space
            self.x = jnp.float32(self.x_offset + jnp.linspace(0, self.x_length, self.x_n))
            self.y = jnp.float32(self.y_offset + jnp.linspace(0, self.y_length, self.y_n))
            self.z = jnp.float32(self.z_offset + jnp.linspace(0, self.z_length, self.z_n))
        else:
            if iteration != 1:
                self.coord_backup = coord_backup
                self.future_dims = future_dims

            if iteration == 1:
                lower = 0
                upper = 65#self.future_dims[1] + 1
            else:
                lower = 64#jnp.sum(self.future_dims[0:iteration])
                upper = 128#lower + self.future_dims[iteration]

            if self.probing_direction == 'x':
                # define coordinate space
                self.x = self.coord_backup[lower:upper]
                self.y = jnp.float32(self.y_offset + jnp.linspace(0, self.y_length, self.y_n))
                self.z = jnp.float32(self.z_offset + jnp.linspace(0, self.z_length, self.z_n))

                self.x_length = self.x[-1] - self.x[0]
                self.lengths = self.lengths.at[0].set(self.x_length)

                self.x_n = len(self.x)
                self.dims = self.dims.at[0].set(self.x_n)
            elif self.probing_direction == 'y':
                # define coordinate space
                self.y = self.coord_backup[lower:upper]
                self.x = jnp.float32(self.x_offset + jnp.linspace(0, self.x_length, self.x_n))
                self.z = jnp.float32(self.z_offset + jnp.linspace(0, self.z_length, self.z_n))

                self.y_length = self.y[-1] - self.y[0]
                self.lengths = self.lengths.at[1].set(self.y_length)

                self.y_n = len(self.y)
                self.dims = self.dims.at[1].set(self.y_n)
            elif self.probing_direction == 'z':
                # define coordinate space
                self.z = self.coord_backup[lower:upper]
                self.x = jnp.float32(self.x_offset + jnp.linspace(0, self.x_length, self.x_n))
                self.y = jnp.float32(self.y_offset + jnp.linspace(0, self.y_length, self.y_n))

                self.z_length = self.z[-1] - self.z[0]
                self.lengths = self.lengths.at[2].set(self.z_length)

                self.z_n = len(self.z)
                self.dims = self.dims.at[2].set(self.z_n)
            else:
                raise AssertionError(colour.BOLD + "Invalid entry for probing_direction!" + colour.END)

        print("\nCoordinates have shape of ({}, {}, {})".format(len(self.x), len(self.y), len(self.z)), end = " --> ")

        if self.x.shape == self.y.shape and self.y.shape == self.z.shape and self.z.shape == self.x.shape:
            self.coordinates = jnp.stack([self.x, self.y, self.z], axis = 1, dtype = jnp.float32)

            print("no padding required.")
        else:
            max_dim = self.dims[0]
            for dimension in self.dims:
                if dimension > max_dim:
                    max_dim = dimension

            # pad coordinates but not arrays themselves, that way only interpolator takes in padded values - no needless extra mem allocation by the domain
            self.coordinates = jnp.stack([
                    jnp.pad(self.x, (0, max_dim - self.x_n), constant_values = jnp.nan),
                    jnp.pad(self.y, (0, max_dim - self.y_n), constant_values = jnp.nan),
                    jnp.pad(self.z, (0, max_dim - self.z_n), constant_values = jnp.nan)
                ], axis = 1)

            print("padded up-to {} entries.".format(max_dim))
            print(" --> x padded with: {} nan's".format(max_dim - self.x_n))
            print(" --> y padded with: {} nan's".format(max_dim - self.y_n))
            print(" --> z padded with: {} nan's".format(max_dim - self.z_n))

        if self.ne is None:
            if self.ne_type is not None:
                self.generate_electron_density_profile()
            else:
                assert auto_batching == True, colour.BOLD + "\nne_type must be passed to domain creation in order to utilise auto-batching." + colour.END

                # can't initialise yourself as equinox.Module inherited class is not mutable and self.ne is set during creation -- FIX!
                print("\nWARNING: Electron density profile to generate not passed. You will need to initialise this yourself with a call to this library.")
                print("\t If you run low on memory, you can enforce a manual domain cleanup with a call to ScalarDomain.cleanup()")

                self.XX, self.YY, self.ZZ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)#False) - has to be true for jnp
                self.ne = jnp.zeros((self.dims[0], self.dims[1], self.dims[2]))
        else:
            print("\nUsing imported ne domain. Be careful that your import matches with other passed variables, this is not sanity checked by the init function.")

            self.XX = None
            self.YY = None
            self.ZZ = None

    #@partial(jax.jit, static_argnames=("self",))  
    def generate_electron_density_profile(self):
        print("\nGenerating test", end = " ")
        if self.ne_type == "test_null":
            print("null -e field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_null()
        elif self.ne_type == "test_slab":
            print("slab -e field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_slab()
        elif self.ne_type == "test_linear_cos":
            print("linear decay periodic -e field...")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            self.test_linear_cos()
        elif self.ne_type == "test_exponential_cos" or self.ne_type is None:
            print("exponential decay periodic -e field...")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            self.test_exponential_cos()
        elif self.ne_type == "test_B":
            print("testB field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_B()
        elif self.ne_type == "quad_trough_test":
            print("Generating field for the Quadratic Trough test case...")
            _, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.XX = None
            self.ZZ = None

            self.quad_trough()
        elif self.ne_type == "import":
            print("pre-generated ne field is auto-imported if passed (not None)...")
        else:
            raise AssertionError("\nNo valid profile detected! Ensure passed name is correct or call yourself.")

        self.cleanup()

    #@partial(jax.jit, static_argnames=("self",))  
    def test_null(self):
        self.ne = self.ne.at[:, :, :].set(jnp.zeros_like(self.XX))

    #@partial(jax.jit, static_argnames=("self",))  
    def test_slab(self, *, s = 1, ne_0 = 2e23):
        if self.s is not None:
            s = self.s
        if self.ne_0 is not None:
            ne_0 = self.ne_0

        self.ne = self.ne.at[:, :, :].set(ne_0 * (1.0 + s * self.XX / self.x_length))

    #@partial(jax.jit, static_argnames=("self",))  
    def test_linear_cos(self, *, s1 = 0.1, s2 = 0.1, ne_0 = 2e23, Ly = 1):
        if self.s1 is not None:
            s1 = self.s1
        if self.s2 is not None:
            s2 = self.s2
        if self.ne_0 is not None:
            ne_0 = self.ne_0
        if self.Ly is not None:
            Ly = self.Ly

        self.ne = self.ne.at[:, :, :].set(ne_0 * (1.0 + s1 * self.XX / self.x_length) * (1 + s2 * jnp.cos(2 * jnp.pi * self.YY / Ly)))

    #@partial(jax.jit, static_argnames=("self",))  
    def test_exponential_cos(self, *, ne_0 = 1e24, Ly = 1e-3, s = -2e-3):
        if self.ne_0 is not None:
            ne_0 = self.ne_0
        if self.Ly is not None:
            Ly = self.Ly
        if self.s is not None:
            s = self.s

        self.XX = self.XX.at[:, :, :].set(self.XX / s)
        self.XX = self.XX.at[:, :, :].set(10 ** self.XX)

        self.YY = self.YY.at[:, :, :].set(self.YY / Ly)
        self.YY = self.YY.at[:, :, :].set(jnp.pi * self.YY)
        self.YY = self.YY.at[:, :, :].set(2 * self.YY)
        self.YY = self.YY.at[:, :, :].set(jnp.cos(self.YY))
        self.YY = self.YY.at[:, :, :].set(1 + self.YY)

        # any difference if float32 (or even 64 if changed later) or not? shouldn't be.
        self.ne = self.XX * self.YY
        self.cleanup()

        self.ne = self.ne.at[:, :, :].set(ne_0 * self.ne)

        #self.ne = jnp.float32(ne_0 * 10 ** (self.XX / s) * (1 + jnp.cos(2 * jnp.pi * self.YY / Ly)))

    def quad_trough(self, *, n_cr = 9.049e27, y_c = 5e-2):
        if self.ne_0 is not None:
            n_cr = self.ne_0
        if self.s is not None:
            y_c = self.s

        self.YY = self.YY.at[:, :, :].set(self.YY / y_c)
        self.YY = self.YY.at[:, :, :].set(self.YY ** 2)
        self.YY = self.YY.at[:, :, :].set(self.YY + 1)

        self.ne = self.YY
        self.cleanup()

        self.ne = self.ne.at[:, :, :].set(n_cr * self.ne)

    #@partial(jax.jit, static_argnames=("self",))  
    def external_ne(self):
        self.ne = self.ne.at[:, :, :].set(self.ne)

    #@jax.jit
    def cleanup(self):
        """
        Deallocates unused temporary meshgrid variables. Calls to shared.utils.dalloc(...)

        :param self: Part of the ScalarDomain class and thus takes in a self object.
        :type self: simulator.domain.ScalarDomain

        :return: Updates the passed simulator.domain.ScalarDomain object.
        :rtype: None
        """

        if self.XX is not None:
            dalloc(self.XX)
        if self.YY is not None:
            dalloc(self.YY)
        if self.ZZ is not None:
            dalloc(self.ZZ)

from scipy.integrate import odeint, solve_ivp
from time import time
from sys import getsizeof as getsizeof_default

# object type of diffrax output
from diffrax import Solution

from scipy.constants import c
from scipy.constants import e

from itertools import product

from jax._src import dtypes
#from jax._src.numpy import (asarray, broadcast_arrays, empty, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact

# is overhead better using jnp.clip or a vectorised(?) if statement?
# if we can sort out my original solution to this - clip would not be necessary at all
# can we speed this up even further?

@jax.jit
def trilinearInterpolator(points, values, xi, method = "linear", bounds_error = False, fill_value = 0.0):
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
        fill_value = jnp.asarray(fill_value)
        if not dtypes.can_cast(fill_value.dtype, values.dtype, casting='same_kind'):
            ve = "fill_value must be either 'None' or of a type compatible with values"
            raise ValueError(ve)

    # TODO: assert sanity of `points` similar to SciPy but in a JIT-able way
    check_arraylike("RegularGridInterpolator", *points)
    grid = tuple(jnp.asarray(p) for p in points)

    ndim = len(grid)

    """Convert a tuple of coordinate arrays to a (..., ndim)-shaped array."""
    if isinstance(xi, tuple) and len(xi) == 1:
        # handle argument tuple
        xi = xi[0]
    if isinstance(xi, tuple):
        p = jnp.broadcast_arrays(*xi)
        for p_other in p[1:]:
            if p_other.shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        xi = jnp.empty(p[0].shape + (len(xi),), dtype=float)
        for j, item in enumerate(p):
            xi = xi.at[..., j].set(item)
    else:
        check_arraylike("_ndim_coords_from_arrays", xi)
        xi = jnp.asarray(xi)  # SciPy: asanyarray(xi)
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
    out_of_bounds = jnp.zeros((xi.T.shape[1],), dtype=bool)
    # iterate through dimensions
    for x, g in zip(xi.T, grid):
        i = jnp.searchsorted(g, x) - 1
        i = jnp.where(i < 0, 0, i)
        i = jnp.where(i > g.size - 2, g.size - 2, i)
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
    result = jnp.asarray(0.)
    for edge_indices in edges:
        weight = jnp.asarray(1.)
        for ei, i, yi in zip(edge_indices, indices, norm_distances):
            weight *= jnp.where(ei == i, 1 - yi, yi)
        result += values[edge_indices] * weight[vslice]

    if not bounds_error and fill_value is not None:
        bc_shp = result.shape[:1] + (1,) * (result.ndim - 1)
        result = jnp.where(out_of_bounds.reshape(bc_shp), fill_value, result)

    return result.reshape(xi_shape[:-1] + values.shape[ndim:])

def dndr(r, gradient_term, omega, x, y, z):
    grad = jnp.zeros_like(r.T)

    dndx = jnp.gradient(gradient_term, x, axis = 0)
    grad = grad.at[0, :].set(trilinearInterpolator((x, y, z), dndx, r, fill_value = 0.0))
    del dndx

    dndy = jnp.gradient(gradient_term, y, axis = 1)
    grad = grad.at[1, :].set(trilinearInterpolator((x, y, z), dndy, r, fill_value = 0.0))
    del dndy

    dndz = jnp.gradient(gradient_term, z, axis = 2)
    grad = grad.at[2, :].set(trilinearInterpolator((x, y, z), dndz, r, fill_value = 0.0))
    del dndz

    return grad

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, ne, x, y, z, omega, lengths, dims):
    # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
    s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    r = s[:3, :].T  # transposed so it is of the correct shape for interpolators
    v = s[3:6, :]

    # Amplitude, phase and polarisation
    amp = s[6, :]

    del s

    gradient_term = -0.5 * c ** 2 * ne / (3.14207787e-4 * omega ** 2)

    # must unpack x, y, z tuple here for the sake of dndr, could be earlier but this is easier to pass and more generalised
    # r must be transposed within dndr(...) else we get an AbstractTerm error due to the effect on the return value
    sprime = sprime.at[3:6, :].set(dndr(r, gradient_term, omega, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    del r
    del v
    del amp

    return sprime.flatten()

def process_results(solutions, depth_traced, trace_depth, probing_direction, duration, save_points_per_region, ray_batch_count, verbose):
    if ray_batch_count > 1:
        # Concatenate time and state arrays
        ts = jnp.concatenate([sol.ts for sol in solutions], axis = 0)
        ys = jnp.concatenate([sol.ys for sol in solutions], axis = 0)

        # Combine stats
        stats_keys = solutions[0].stats.keys()
        stats = {
            key: jnp.concatenate([sol.stats[key] for sol in solutions], axis = 0)
            for key in stats_keys
        }

        # Combine other fields
        t0 = solutions[0].t0
        t1 = solutions[-1].t1
        result = solutions[-1].result  # Use the last result

        del solutions

        # if info is missing that you need, this is why - implement it !
        solutions = Solution(
            t0 = t0,
            t1 = t1,
            ts = ts,
            ys = ys,
            interpolation = None,  # Optional: you can implement logic to keep interpolations
            stats = stats,
            result = result,
            solver_state = None,
            controller_state = None,
            made_jump = None,
            event_mask = None
        )

        solutions = np.asarray([solutions], dtype = Solution)

    if verbose:
        print("\nParallelised output has resulting 3D matrix of form: [batch_count, (save_points_per_region - 1) * ScalarDomain.region_count, 9]:", solutions[0].ys.shape)
        print(" - 2 to account for the start and end results (typical, can be greater if set)")
        print(" - 9 containing the 3 position and velocity components, amplitude, phase and polarisation")
        print(" - If batch_count is lower than expected, this is likely due to jax's forced integer batch sharding requirement over cpu cores.")

        print("\nWe slice the", end = " ")
        if len(solutions[0].ys.shape) == 3:
            print("results", end = " ")
        else:
            print("end result", end = " ")
        print("and transpose into the form:", solutions[0].ys.shape, "to work with later code.")

    if save_points_per_region == 2 or save_points_per_region == 1:
        rf = solutions[0].ys[:, -1, :].T

        # depth_traced + trace_depth or just trace_depth
        return ray_to_Jonesvector(rf, ne_extent = depth_traced + trace_depth, probing_direction = probing_direction), duration
    elif save_points_per_region > 2:
        slice_rf_list = []
        slice_Jf_list = []

        for i in range(len(solutions)):
            #save_point_depth = depth_traced
            for j in range(save_points_per_region):
                '''
                if j == save_points_per_region - 1:
                    save_point_depth = depth_traced + trace_depth
                else:
                    save_point_depth += trace_depth // save_points_per_region
                '''

                if j < save_points_per_region - 1 or (j == save_points_per_region - 1 and i == len(solutions) - 1):
                    # sol.ts having shape of (Np, save_points_per_region) per region is very inefficent given there are N - 1 duplications
                    # - issue with diffrax though I can't fix this
                    slice_rf_list.append(ray_to_Jonesvector(solutions[i].ys[:, j, :].T, ne_extent = depth_traced + trace_depth * solutions[i].ts[0, j], probing_direction = probing_direction, keep_current_plane = True))

        rf = jnp.stack(slice_rf_list, axis = 0)
        del slice_rf_list

        return rf, duration
    else:
        assert "\nWhat."

def solve(beam, ScalarDomain, probing_depth, *, jitted = True, save_points_per_region = 2, memory_debug = False, lwl = 1064e-9, keep_domain = False, return_raw_results = False, verbose = True, rtol = 1, atol = 1e-5):
    omega = 2 * jnp.pi * c / lwl

    region_count = ScalarDomain.region_count
    ray_batch_count = ScalarDomain.ray_batch_count

    print("\nNumber of domain batches:", region_count)
    print("Number of ray batches:", ray_batch_count)

    assert not isinstance(beam, Beam), "\nThis function does not take in the direct output of the Beam object, pass either Beam.s0 rays, or the parameters passed to be Beam here as a tuple if batching rays."

    unbatched_beam = False
    if ray_batch_count == 1:
        import array
        if isinstance(beam, array.array) or isinstance(beam, np.ndarray) or isinstance(beam, jax.Array):
            assert len(beam.shape) == 2, "\nExpected a matrix of pre-created rays."

            s0_import = beam
            del beam

            Np = s0_import.shape[1]

            Np_total = Np
            rays_per_batch = Np # not necessary, just so there is something to print if someone tries

            rays = np.array([Np], dtype = np.int64)
        elif isinstance(beam, tuple):
            unbatched_beam = True

            print("\nUsing tuple values to create the unbatched beam, domain must be used in the same fashion.")

            Np_total = ScalarDomain.Np_total
            rays_per_batch = Np_total

            rays = np.array([Np_total], dtype = np.int64)
    else:
        assert isinstance(beam, tuple), "\nExpect a tuple of Beam properties if you wish to batch rays."

        Np_total = ScalarDomain.Np_total

        #Np = Np_total // ray_batch_count
        rays_per_batch = Np_total // ray_batch_count
        rays = np.array([rays_per_batch] * (ray_batch_count - 1) + [Np_total - rays_per_batch * (ray_batch_count - 1)], dtype = np.int64)

    duration = np.float64(0.0)
    solutions = np.empty(ray_batch_count, dtype = Solution)

    for ray_index, Np in enumerate(rays):
        depth_traced = 0.0

        if ray_batch_count > 1 or unbatched_beam:
            s0_import = Beam(Np, beam_size = beam[0], divergence = beam[1], ne_extent = beam[2], probing_direction = beam[3], beam_type = beam[4], seeded = beam[5]).s0

        single_ray_size = getsizeof_default(s0_import[:, 0])
        print("\nEst. size in memory of rays (1 = {}): {}".format(mem_conversion(single_ray_size), mem_conversion(single_ray_size * Np)))
        total_ray_size_estimate_raw = getsizeof_default(s0_import[:, 0]) * Np_total
        if ray_batch_count > 1:
            print("Est. potential size in memory of total rays:", mem_conversion(total_ray_size_estimate_raw))
            print(" --> Np (total) = {} (in {} batches) - {} for this batch".format(Np_total, ray_batch_count, Np))
        else:
            print(" --> Np = {}".format(Np))

        for i in range(1, ScalarDomain.region_count + 1):
            if ScalarDomain.region_count == 1:
                print("\nNo need to generate any sections of the domain, batching not utilised.")

                trace_depth = probing_depth
            else:
                if i == 1:
                    print("\nUsing pre-generated 1st section of domain.")
                else:
                    print("\nGenerating section" + i + "of the domain...")

                    lengths = ScalarDomain.lengths
                    dims = ScalarDomain.dims
                    zeropoint = ScalarDomain.zeropoint

                    ne_type = ScalarDomain.ne_type

                    probing_direction = ScalarDomain.probing_direction

                    region_count = ScalarDomain.region_count

                    leeway_factor = ScalarDomain.leeway_factor

                    coord_backup = ScalarDomain.coord_backup
                    future_dims = ScalarDomain.future_dims

                    try:
                        del ScalarDomain
                    except:
                        ScalarDomain = None

                    import simulator.domain as d
                    ScalarDomain = d.ScalarDomain(
                        lengths, dims, zeropoint,
                        ne_type = ne_type,
                        probing_direction = probing_direction,
                        auto_batching = True,
                        iteration = i,
                        region_count = region_count,
                        leeway_factor = leeway_factor,
                        coord_backup = coord_backup,
                        future_dims = future_dims
                    )

                    del lengths
                    del dims
                    del zeropoint

                    del ne_type

                    del probing_direction

                    del region_count

                    del leeway_factor

                    del coord_backup
                    del future_dims

                # Need to make sure all rays have left volume
                # Conservative estimate of diagonal across volume
                # Then can backproject to surface of volume

                depth_remaining = probing_depth - depth_traced

                trace_depth = ScalarDomain.lengths[['x', 'y', 'z'].index(ScalarDomain.probing_direction)]
                if trace_depth > depth_remaining:
                    trace_depth = depth_remaining

                del depth_remaining

            target_depth = trace_depth + depth_traced

            # it isn't tracing up till this depth, it is tracing this amount further
            # at end positions are r(vector) + trace_depth (ish) NOT trace_depth(vector)
            print(" --> tracing a depth of", trace_depth, "mm's to the target depth of", target_depth, "mm's")

            t = jnp.linspace(0.0, jnp.sqrt(8.0) * trace_depth / c, 2)
            norm_factor = jnp.max(t)

            # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
            # think we should change this???

            # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not
            args = (
                ScalarDomain.ne,
                ScalarDomain.x, ScalarDomain.y, ScalarDomain.z,
                omega,
                ScalarDomain.lengths, ScalarDomain.dims
            )

            # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
            available_devices = jax.devices()

            running_device = jax.lib.xla_bridge.get_backend().platform # - deprecated, using still as needed for HPC
            #running_device = jax.extend.backend.get_backend().platform
            print("\nRunning device:", running_device, end='')

            if i == 1:
                s0_transformed = s0_import.T
                del s0_import
            else:
                # change target_depth back to trace_depth and check the difference
                s0_transformed = back_propogate(sol.ys[:, -1, :].T, target_depth, ScalarDomain.probing_direction).T
                del sol

            if running_device == 'cpu':
                core_count = int(os.environ['XLA_FLAGS'].replace("--xla_force_host_platform_device_count=", ''))
                print(", with:", core_count, "cores.")

                if Np >= core_count:
                    from jax.sharding import PartitionSpec as P, NamedSharding

                    # Create a Sharding object to distribute a value across devices:
                    # Assume self.core_count is the no. of core devices available
                    mesh = jax.make_mesh((core_count,), ('rows',))  # 1D mesh for columns

                    # Specify sharding: don't split axis 0 (rows), split axis 1 (columns) across devices
                    # then apply sharding to rewrite s0 as a sharded array from it's original matrix
                    # and use jax.device_put to distribute it across devices:
                    Np = ((Np // core_count) * core_count)
                    #assert Np > 0, "Not enough rays to parallelise over cores, increase to at least " + str(core_count)

                    # if you don't wish to transpose before operation you need to use the old call
                    # s0 = jax.device_put(s0_transformed[:, 0:Np], NamedSharding(mesh, P(None, 'cols')))
                    s0 = jax.device_put(s0_transformed[0:Np, :], NamedSharding(mesh, P('rows', None)))  # 'None' means don't shard axis 0

                    print(s0.sharding)            # See the sharding spec
                    #print(s0.addressable_shards)  # Check each device's shard
                    #jax.debug.visualize_array_sharding(s0)
                else:
                    s0 = jax.device_put(s0_transformed)

                    print(colour.BOLD + "Not enough rays to parallelise over cores" + colour.END + ": increase to at least " + str(core_count) + " to utilise parallelisation")
                    print(" --> Running CPU processes sequentially")
            elif running_device == 'gpu':
                gpu_devices = jax.devices('gpu')
                print("\nThere are", len(gpu_devices), "available GPU devices:", gpu_devices)
                assert len(gpu_devices) > 0, "Running on GPU yet none detected?"

                s0 = jax.device_put(s0_transformed, gpu_devices[0])
            elif running_device == 'tpu':
                pass

                s0 = s0_transformed
            else:
                assert "No suitable device detected!"

            del s0_transformed
            # optional for aggressive cleanup?
            #jax.clear_caches()

            # wrapper for same reason, diffrax.ODETerm instantiaties this and passes args
            # I have no idea why, but this has to be defined in solve rather than as a global function - else there is an abstract variable error
            def dsdt_ODE(t, y, args):
                return dsdt(t, y, *args) * norm_factor

            from diffrax import ODETerm, Tsit5, SaveAt, PIDController, diffeqsolve
            #import optax - diffrax uses as a dependency, don't need to import directly

            # using lengths and/or dims to set parameters of diffeqsolve(...) results in BooleanConversionError due to tracing variable resolution
            # rtol & atol are good here - setting too precise increases runtime dramatically for little change in results, it overcompensates
            def diffrax_solve(dydt, t0, t1, Nt, lengths, dims, *, rtol = 1, atol = 1e-5):
                """
                Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
                """

                # We convert our python function to a diffrax ODETerm
                # should use the function passed into the wrapper - not the local definition
                term = ODETerm(dydt)

                # We chose a solver (time-stepping) method from within diffrax library
                solver = Tsit5() # (RK45 - closest I could find to solve_ivp's default method)

                # At what time points you want to save the solution
                saveat = SaveAt(ts = jnp.linspace(t0, t1, Nt))
    
                # Diffrax uses adaptive time stepping to gain accuracy within certain tolerances
                # setting dtmax increases runtime significantly - maybe this is too high and thus calculations are not precise due to scale of change?
                #dtmax = 0.5 * ((lengths[0] / dims[0])**2 + (lengths[1] / dims[1])**2 + (lengths[2] / dims[2])**2) ** (1 / 2) / (c * norm_factor)
                stepsize_controller = PIDController(rtol = rtol, atol = atol)#, dtmax = dtmax)

                return lambda s0, args : diffeqsolve(
                    term,
                    solver,
                    y0 = jnp.array(s0),
                    args = args,# + (atten, ),
                    t0 = t0,
                    t1 = t1,
                    # None (leaving up to controller) shows better performance than setting ourselves
                    dt0 = None,#(t1 - t0) * norm_factor / Nt, # can set = 0 if dtmax is set apparently?
                    saveat = saveat,
                    stepsize_controller = stepsize_controller,
                    # set max steps to no. of cells x100
                    # cannot be passed as dims --> causes boolean conversion error, has to be passed directly
                    # need to pass this correctly so that it remains consistent with class when batching
                    max_steps = int(2e8) #dims[0] * dims[1] * dims[2] * 100 #10000 - default for solve_ivp?????
                ) # the 2e8 choice is very arbritrary

            # hardcode to normalise to 1 due to diffrax bug
            ODE_solve = diffrax_solve(dsdt_ODE, 0, 1, save_points_per_region, ScalarDomain.lengths, ScalarDomain.dims, rtol = rtol, atol = atol)

            from equinox import filter_jit
            ODE_solve = filter_jit(ODE_solve)

            start = time()
            sol = jax.block_until_ready( jax.vmap(ODE_solve, in_axes = (0, None))(s0, args) )
            duration += np.float64(time() - start)

            if i == ScalarDomain.region_count:
                if total_ray_size_estimate_raw >= memory_report("cpu")['free_raw']:
                    target_folder = os.getcwd() + "/saves"
                    if not os.path.isdir(target_folder):
                        try:
                            os.mkdir(target_folder)
                        except OSError as e:
                            print("\nFailed to create folder at " + target_folder)
                            if e.errno != errno.EEXIST:
                                raise

                    tar_gz_path = target_folder + "/ray_output_total_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".hdf5.tar.gz"

                    from utils.handle_filetypes import compress_matrix_to_hdf5_BytesIO
                    from utils.handle_filetypes import stream_data_to_tar_gz

                    filename = "run_" + str(ray_index)
                    stream_data_to_tar_gz(tar_gz_path, filename,
                        compress_matrix_to_hdf5_BytesIO(
                            ray_to_Jonesvector(sol.ys[:,-1].reshape(9, Np), ne_extent = probing_depth, probing_direction = ScalarDomain.probing_direction)[0]
                        )
                    )
                else:
                    solutions[ray_index] = sol
                    del sol

            depth_traced += trace_depth

    print("\nCompleted ray trace in", colour.BOLD + str(np.round(duration, 3).astype(np.float64)) + colour.END, "seconds.")

    if total_ray_size_estimate_raw < memory_report("cpu")['free_raw']:
        if return_raw_results:
            return solutions, duration
        else:
            # need to confirm there is no mismatch between total depth_traced and the target probing_depth
            return process_results(solutions, depth_traced, trace_depth, ScalarDomain.probing_direction, duration, save_points_per_region, ray_batch_count, verbose)
    else:
        print("\nData output as a hdf4.tar.gz file due to limitations of vram/ram space.")
        print("Graphs can be iteratively plotted by cycling through the 'run_n' entries after extraction from .tar.gz format.")

import matplotlib as mpl

from sympy import Matrix

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, *, ne_extent = None, probing_direction = 'z', keep_current_plane = False):
    # * forces keep_current_plane to be a keyword-only argument
    # meaning .. keep_current_plane = True (missing out others) will work as it will not rely on position

    if ne_extent is None and keep_current_plane == False:
        from shared.printing import colour
        print(colour.BOLD + "\nne_extent is only not required if keep_current_plane is set to True, setting keep_current_plane = True for you." + colour.END)

        keep_current_plane = True

    Np = rays.shape[1] # number of photons

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    ray_p = jnp.zeros((4, Np))

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        # Positions on plane
        if not keep_current_plane:
            t_bp = (x - ne_extent) / vx

            ray_p = ray_p.at[0].set(y - vy * t_bp)
            ray_p = ray_p.at[2].set(z - vz * t_bp)
        else:
            ray_p = ray_p.at[0].set(y)
            ray_p = ray_p.at[2].set(z)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vy / vx))
        ray_p = ray_p.at[3].set(jnp.arctan(vz / vx))
    # XZ plane
    elif(probing_direction == 'y'):
        #
        # I have switched x & z for the sake of consistent ordering of the axes
        # Standardised in keeping with positive 'forward' notation, etc. x * y = z but don't do y * x = -z
        # If memory is not a concern then will instead create a class to cover directions
        # This would entail both the array and a self.dir parameter of type char - containing 'x', 'y' or 'z'
        #

        # Positions on plane
        if not keep_current_plane:
            t_bp = (y - ne_extent) / vy

            ray_p = ray_p.at[0].set(z - vz * t_bp)
            ray_p = ray_p.at[2].set(x - vx * t_bp)
        else:
            ray_p = ray_p.at[0].set(z)
            ray_p = ray_p.at[2].set(x)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vz / vy))
        ray_p = ray_p.at[3].set(jnp.arctan(vx / vy))
    # XY plane
    elif(probing_direction == 'z'):
        # Positions on plane
        if not keep_current_plane:
            t_bp = (z - ne_extent) / vz

            ray_p = ray_p.at[0].set(x - vx * t_bp)
            ray_p = ray_p.at[2].set(y - vy * t_bp)
        else:
            ray_p = ray_p.at[0].set(x)
            ray_p = ray_p.at[2].set(y)

        # Angles to plane
        ray_p = ray_p.at[1].set(jnp.arctan(vx / vz))
        ray_p = ray_p.at[3].set(jnp.arctan(vy / vz))
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")
    
    del x
    del y
    del z
    del vx
    del vy
    del vz

    del Np

    # ray_p [x, phi, y, theta] +? [amp, phase], ray_J [E_x, E_y]

    return ray_p

def back_propogate(rays, ne_extent, probing_direction):
    Np = rays.shape[1] # number of photons

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x - ne_extent) / vx

        # Positions on plane
        rays = rays.at[0].set(ne_extent)
        rays = rays.at[1].set(y - vy * t_bp)
        rays = rays.at[2].set(z - vz * t_bp)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y - ne_extent) / vy

        #
        # I have switched x & z for the sake of consistent ordering of the axes
        # Standardised in keeping with positive 'forward' notation, etc. x * y = z but don't do y * x = -z
        # If memory is not a concern then will instead create a class to cover directions
        # This would entail both the array and a self.dir parameter of type char - containing 'x', 'y' or 'z'
        #

        # Positions on plane
        rays = rays.at[0].set(z - vz * t_bp)
        rays = rays.at[1].set(ne_extent)
        rays = rays.at[2].set(x - vx * t_bp)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z - ne_extent) / vz

        # Positions on plane
        rays = rays.at[0].set(x - vx * t_bp)
        rays = rays.at[1].set(y - vy * t_bp)
        rays = rays.at[2].set(ne_extent)
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")

    del x
    del vx

    del y
    del vy

    del z
    del vz

    return rays

def m_to_mm(r):
    rr = jnp.copy(r)
    rr = rr.at[0::2, :].set(rr[0::2, :] * 1e3)

    return rr

def mm_to_m(r):
    rr = jnp.copy(r)
    rr = rr.at[0::2, :].set(rr[0::2, :] * 1e-3)

    return rr

def lens(r, f1, f2):
    """
    4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    l1 = jnp.asarray([[1, 0],
            [-1 / f1, 1]])
    l2 = jnp.asarray([[1, 0],
            [-1 / f2, 1]])

    L = jnp.zeros((4, 4))
    L = L.at[:2, :2].set(l1)
    L = L.at[2:, 2:].set(l2)

    return jnp.matmul(L, r)

def sym_lens(r, f):
    """
    Helper function to create an axisymmetryic lens
    """

    return lens(r, f, f)

def travel(r, d):
    """4x4 matrix  matrix for travelling a travel d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    d = jnp.asarray([[1, d],
                     [0, 1]])

    L = jnp.zeros((4, 4))

    L = L.at[:2, :2].set(d)
    L = L.at[2:, 2:].set(d)

    return jnp.matmul(L, r)

def circular_aperture(r, R):
    """
    Rejects rays outside radius R
    """

    filt = r[0, :] ** 2 + r[2, :] ** 2 > R ** 2
    # if you want to reject rays outside of the radius, then when filt is true you should set equal to None
    return r.at[:, filt].set(jnp.nan)

def circular_stop(r, R):
    """
    Rejects rays inside a radius R
    """

    filt = r[0, :] ** 2 + r[2,:] ** 2 < R ** 2
    r = r.at[:, filt].set(jnp.nan)

    return r

def annular_stop(r, R1, R2):
    """
    Rejects rays which fall between R1 and R2
    """

    filt1 = (r[0,:]**2+r[2,:]**2 > R1**2)
    filt2 = (r[0,:]**2+r[2,:]**2 < R2**2)
    filt = (filt1 & filt2)

    return filt

def rect_aperture(r, Lx, Ly):
    """
    Rejects rays outside a rectangular aperture, total size 2*Lx x 2*Ly
    """

    filt1 = (r[0, :] ** 2 > Lx ** 2)
    filt2 = (r[2, :] ** 2 > Ly ** 2)

    filt = filt1 * filt2
    r = r.at[:, filt].set(jnp.nan)

    return r

def knife_edge(r, offset, axis, direction):
    """
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    """

    if axis == 'y':
        a = 2
    if axis == 'x':
        a = 0

    if direction > 0:
        filt = r[a,:] > offset
    if direction < 0:
        filt = r[a,:] < offset
    if direction == 0:
        print('Direction must be < 0 or > 0')

    r = r.at[:, filt].set(jnp.nan)

    return r

def clear_rays(self):
    """
    Clears the r0 and rf variables to save memory
    """
    # does this actually save memory in the best way?
    # would it be better to del self.r_ instead?

    self.r0 = None
    self.rf = None

def ray(x, θ, y, ϕ):
    """
    Returns a 4x1 matrix representing a ray. Spatial units must be consistent, angular units in radians.
    """

    return Matrix([x, θ, y, ϕ])

def d2r(d):
    # helper function, degrees to radians
    return d * jnp.pi / 180

def lens_cutoff(rf, *, L = 400, R = 25):
    mask = jnp.pow(jnp.pow(L * jnp.tan(rf[1]) + rf[0], 2) + jnp.pow(L * jnp.tan(rf[3]) + rf[2], 2), 0.5) <= R
    return jnp.asarray(rf)[:, mask]

class Diagnostic:
    # this is in mm's not metres - self.rf is converted to mm's (not sure if everything else is covered though)
    def __init__(self, wavelength, rf, *, focal_plane = 0, L = 400, R = 25, Lx = 18, Ly = 13.5, x = None, y = None, x_l = None, y_l = None, l_x = 0, u_x = 0.3, l_y = -5, u_y = 5):
        """
        Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """     

        self.wavelength, self.focal_plane, self.L, self.R, self.Lx, self.Ly = wavelength, focal_plane, L, R, Lx, Ly

        self.x, self.y, self.x_l, self.y_l = x, y, x_l, y_l

        # these HAVE to stay... for some reason - not entirely sure why you can't just reference self.Beam.r_ directly (or now just rf)
        # if you can make it without the memory duplication work please do, else DON'T REMOVE!

        # these are created as jax.Array's, yet received as a tuple here?
        # likely as they are passed externally where jax.numpy module is not loaded
        # just re-assert type here to fix

        if rf is not None:
            # separates out the amp/phase part of rf from raw values
            if rf.shape[0] == 6:
                rf = rf[:4, :]
            else:
                assert rf.shape[0] == 4, colour.BOLD + "\nIncorrect format for rf, are you sure you passed the right variable?" + colour.END

            # forces self.rf to the last slice if rf returns multiple samples
            # also preserves the whole pass if required
            if len(rf.shape) == 3:
                # rf might be 3-dimensional if it is a series of 2D ray solution slices
                rf = rf[-1, :, :]

            self.Np = rf.shape[-1]

            # masks rf to only hold entries corr. to rays that will be captured by the lense setup
            # also forces matrices to type jax.Array via jnp.asarray()
            self.rf = lens_cutoff(rf)

            self.Np_inc = self.rf.shape[-1]
            if self.Np == self.Np_inc:
                print("\nAll rays incident on lens!")
            else:
                print("\n{} rays received, {} incident on the first lens.".format(str(self.Np), str(self.Np_inc)))
                print(" --> {} % of rays wasted!".format(str(round_to_n((1 - self.Np_inc / self.Np) * 100, 3))))
        else:
            assert "rf should not be of Noneype! diffrax clearly failed."

        # however, doesn't have to be done manually now as already sorted in propagator.py, therefore no more duplication
        # still odd though... (hence the keeping of the comment)

        self.r0 = m_to_mm(self.rf)

    def histogram(self, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False, plain_plot = False):
        """
        Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 1.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """

        if plain_plot:
            x, y = count_nans(self.r0, ret = True)
        else:
            x, y = count_nans(self.rf, ret = True)

        self.H, self.xedges, self.yedges = jnp.histogram2d(x, y, bins=[np.floor(pix_x / bin_scale).astype(np.int64), np.floor(pix_y / bin_scale).astype(np.int64)], range=[[-self.Lx / 2, self.Lx / 2],[-self.Ly / 2, self.Ly / 2]])
        self.H = self.H.T

        #Optional - clear ray attributes to save memory
        if clear_mem:
            clear_rays(self)

    def plot(self, ax, clim = None, cmap = None):
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap, extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def histogram_legacy(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        # repeated across many functions, have made a wrapper for it instead of repeats to preserve backwards compatability
        # was this replaced by jnp.histogram2d function?
        # this function is far slower for a general histogram than the new function - yet is used for refractogram and interferogram so kept to be wrapped for those
        x_bins = jnp.linspace(-self.Lx // 2, self.Lx // 2, np.floor(pix_x / bin_scale).astype(np.int64))
        y_bins = jnp.linspace(-self.Ly // 2, self.Ly // 2, np.floor(pix_y / bin_scale).astype(np.int64))

        amplitude_x = jnp.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype = complex)
        amplitude_y = jnp.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype = complex)

        x_indices = jnp.digitize(self.rf[0, :], x_bins) - 1
        y_indices = jnp.digitize(self.rf[2, :], y_bins) - 1

        mask = (0 <= x_indices) & (x_indices < amplitude_x.shape[1]) & (0 <= y_indices) & (y_indices < amplitude_x.shape[0])

        # jax arrays are immutable - fix later
        amplitude_x = amplitude_x.at[y_indices[mask], x_indices[mask]].set(amplitude_x[y_indices[mask], x_indices[mask]])
        amplitude_y = amplitude_y.at[y_indices[mask], x_indices[mask]].set(amplitude_y[y_indices[mask], x_indices[mask]])

        amplitude = jnp.sqrt(jnp.real(amplitude_x) ** 2 + jnp.real(amplitude_y) ** 2)
        # amplitude_normalised = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) # this line needs work and is currently causing problems
        self.H = amplitude

    def plot_rays(self, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem, plain_plot = True)

class Shadowgraphy(Diagnostic):
    """
    Example shadowgraphy diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1 and a single lens system with M = 2. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    """

    def single_lens_solve(self):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L / 2)             # lens 1
        r4 = travel(r3, 3*self.L / 2)           # detector
        self.rf = r4

    def two_lens_solve(self):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L / 2)           # lens 1
        r4 = travel(r3, self.L * 2)           # displace rays to lens 2.
        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L / 2)           # lens 2
        r7 = travel(r6, self.L)             # displace rays to detector
        self.rf = r7
    
class Schlieren(Diagnostic):
    """
    Example dark field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point after the first lens which rejects rays which hit the focal planes at travel less than R [mm] from the optical axis.
    """

    def DF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
    
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = travel(r3, self.L) #displace rays to stop

        # this and positioning of lenses means schlieren ends up with less usable rays than other methods
        r5 = circular_stop(r4, R = R) # stop - blocker at focal point after the first lens of size R (1 mm?)

        r6 = travel(r5, self.L) #displace rays to lens 2

        r7 = circular_aperture(r6, self.R) # cut off

        r8 = sym_lens(r7, self.L) #lens 2

        r9 = travel(r8, self.L) #displace rays to detector

        self.rf = r9
    
    """
    Example light field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which accepts only rays which hit the focal planes at travel less than R [mm] from the optical axis.
    """

    def LF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = travel(r3, self.L) #displace rays to stop
        r5 = circular_aperture(r4, R = R) # stop

        r6 = travel(r5, self.L) #displace rays to lens 2
        r7 = circular_aperture(r6, self.R) # cut off
        r8 = sym_lens(r7, self.L) #lens 2

        r9 = travel(r8, self.L) #displace rays to detector
        self.rf = r9
        
class Refractometry(Diagnostic):
    """
    Example of Imaging Refractometer. Inherits from Rays, has custom solve method.
    Implements a spherical lens with focal length f1 = L/2 and M = 2 for the spatial axis and a cylindrical lens
    with focal length f1 and f2.
    """

    def incoherent_solve(self):
        ##
        ## Is there an efficient way to chain these so needlessly variables are not used without having 1 really long line
        ##

        ## Imaging the spatial axis - M = 2
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens 1. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1 - spherical
        r4 = travel(r3, 3*self.L/2)           # displace rays to lens 2 - hybrid
        r5 = rect_aperture(r4, 15, 30)          # rectangular lens cut-off
        r6 = circular_aperture(r5, self.R)      # cut off
        r7 = lens(r6, self.L/3, self.L/2)       # lens 2 - hybrid lens
        r8 = travel(r7, self.L)               # displace rays to detector
        self.rf = r8

    def coherent_solve(self):
        ## Imaging the spatial axis - M = 2 - Coherent Implementation of the Refractometer
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane)
        # propagate E field

        r2 = circular_aperture(self.r0, self.R)      # cut off
        r3 = sym_lens(r2, self.L / 2)          # lens 1 - spherical

        r4 = travel(r3, 3 * self.L / 2)                 # displace rays to lens 2 - hybrid

        r5 = circular_aperture(r4, self.R)      # cut off
        r6 = lens(r5, self.L / 3, self.L / 2)       # lens 2 - hybrid lens

        self.rf = travel(r6, self.L)               # displace rays to detector

    def coherent_solve_alt(self):
        ## Imaging the spatial axis - M = 2 - Coherent Implementation of the Refractometer
        r1 = travel(self.r0, 3 * self.L / 4 - self.focal_plane)

        r2 = circular_aperture(r1, self.R)      # cut off

        r3 = sym_lens(r2, self.L / 2)          # lens 1 - spherical

        r4 = travel(r3, 3 * self.L / 2)

        r5 = circular_aperture(r4, self.R)      # cut off                 # displace rays to lens 2 - hybrid                # displace rays to lens 2 - hybrid

        r6 = lens(r5, self.L / 3, self.L / 2)       # lens 2 - hybrid lens

        self.rf = travel(r6, self.L)               # displace rays to detector

    def refractogram(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram_legacy(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem)

class Interferometry(Diagnostic):
    """
    Simple class to keep all the ray properties together
    """

    def bkg(self, domain_length, n_fringes, deg, ne_extent, probing_direction):
        rr0 = ray_to_Jonesvector(self.rf, ne_extent, probing_direction = probing_direction, keep_current_plane = True)

        # assuming reference is recombined with the probe beam at the exit of the domain (should be changed)
        self.interfere_ref_beam(n_fringes, deg)
        ## 2 lens telescope, M = 1
        r1 = travel(rr0, self.L + domain_length) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L / 2)           # lens 1

        r4 = travel(r3, self.L * 2)           # displace rays to lens 2.
        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L / 2)                             # lens 2
        
        r7 = travel(r6, self.L)             # displace rays to detector
        rf = r7

        self.histogram(self)
        self.bkg_signal = self.H

    def two_lens_solve(self):
        # assuming reference is recombined with the probe beam at the exit of the domain (should be changed)
        self.interfere_ref_beam(10, 20)
        ## 2 lens telescope, M = 1
        r1 = travel(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth

        r2 = circular_aperture(r1, self.R)    # cut off

        r3 = sym_lens(r2, self.L/2)           # lens 1

        r4 = travel(r3, self.L*2)           # displace rays to lens 2.

        r5 = circular_aperture(r4, self.R)    # cut off

        r6 = sym_lens(r5, self.L/2)                             # lens 2
        
        r7 = travel(r6, self.L)             # displace rays to detector

        self.rf = r7

    def interferogram(self, *, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram_legacy(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem)