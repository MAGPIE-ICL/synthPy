import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

import os
import errno

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
parser.add_argument("-p", "--importPath", type = str)
args = parser.parse_args()

if args.rays is not None:
    Np = args.rays
else:
    Np = 3e5 # should generate ~ 60 GiB of data

cores = None
if args.cores is not None:
    cores = args.cores

importPath = None
if args.importPath is not None:
    importPath = args.importPath
else:
    # attempts to fix path issues - need to find a resolution to the problem of relative paths on the HPC
    importPath = '/rds/general/user/sm5625/home/synthPy/src/'

sys.path.insert(0, importPath)

import simulator.config as config
config.jax_init(core_limit = cores, jax_updated = False)

import jax.numpy as jnp

import importlib

import simulator.beam as beam_initialiser
import simulator.domain as d
import simulator.propagator as p
import processing.diagnostics as diag
import utils.handle_filetypes as utilIO

importlib.reload(beam_initialiser)
importlib.reload(d)
importlib.reload(p)
importlib.reload(diag)

from shared.printing import colour
from shared.utils import memory_report
from shared.utils import mem_conversion

# can't load domain on a regular laptop due to it's size
# - therefore I've created a separate test of ray saving capabilities with a standard domain that's generated per run

dims = np.array([128, 128, 128])

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

lengths = 2 * jnp.array([extent_x, extent_y, extent_z], dtype = jnp.int32)

probing_direction = 'z'
domain = d.ScalarDomain(lengths, 128, ne_type = "test_exponential_cos", probing_direction = probing_direction, Np = Np)

# define beam parameters
lwl = 1064e-9
beam_size = extent_x    # beam radius
probing_extent = extent_z
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
divergence = 5e-5
beam_type = "circular"

# setting seeded = True is what causes the spiral effect instead of an actual beam
rf, _, duration = p.solve((beam_size, divergence, ne_extent, probing_direction, beam_type, False), domain, probing_extent, verbose = False)

'''
from shared.propagation import ray_to_Jonesvector
from utils.handle_filetypes import save_jax_matrix_to_hdf5 as compressed_solution_export
_, _ = compressed_solution_export(
    ray_to_Jonesvector(rf, ne_extent = probing_extent, probing_direction = probing_direction, return_E = False)[0],
    filepath = target_folder
    #filename = None, filepath = ".", dataset_name = 'data', compression = 'gzip', compression_level = 4
)
'''

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

filename = "run_" + str(1)
stream_data_to_tar_gz(tar_gz_path, filename, compress_matrix_to_hdf5_BytesIO(rf))

from utils.handle_filetypes import load_array_member_from_hdf5_tar_gz
array = load_array_member_from_hdf5_tar_gz(tar_gz_path, filename)
print(array)

print(colour.BOLD + "\nDuration of " + str(duration) + " sec for domain of size " + str(dims) + " ^3 and " + str(Np) + " rays with legacy solver." + colour.END)