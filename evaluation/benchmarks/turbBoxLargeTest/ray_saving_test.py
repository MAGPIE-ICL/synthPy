import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
parser.add_argument("-p", "--importPath", type = str)
args = parser.parse_args()

if args.rays is not None:
    Np = args.rays
else:
    Np = 1e8 # should generate ~ 60 GiB of data

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

columns = ["dims", "rays", "runtime", "legacyRuntime", "domainSize", "raySize", "totalMemory"]
df = pd.DataFrame(columns=columns)

# can't load domain on a regular laptop due to it's size
# - therefore I've created a separate test of ray saving capabilities with a standard domain that's generated per run

dims = np.array([128, 128, 128])

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

lengths = 2 * jnp.array([extent_x, extent_y, extent_z], dtype = jnp.int32)

print("\n\n")

# is this baseline not decreasing after each run? - testing manually deleting objects first
baseline = memory_report()['used_raw']

probing_direction = 'z'
domain = d.ScalarDomain(lengths, dims, ne_type = "test_exponential_cos", probing_direction = probing_direction, Np = Np)

postDomain = memory_report()['used_raw']
domainAllocation = postDomain - baseline

plusRays = memory_report()['used_raw']

# define beam parameters
lwl = 1064e-9
beam_size = [extent_x, extent_y]    # beam radius
probing_extent = extent_z
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
divergence = 0.05e-3
beam_type = "rectangular"

_, _, duration = p.solve((beam_size, divergence, ne_extent, probing_direction, beam_type, True), domain, probing_extent, verbose = False)

total = memory_report()['used']

print(colour.BOLD + "\nDuration of " + str(duration) + " sec for domain of size " + str(dims[i]) + " ^3 and " + str(rays[j]) + " rays with legacy solver." + colour.END)

new_entry = pd.DataFrame([{
    "dims": dims[i],
    "rays": rays[j],
    "runtime": duration,
    "legacyRuntime": "N/A",
    "domainSize": mem_conversion(domainAllocation),
    "raySize": mem_conversion(plusRays - domainAllocation),
    "totalMemory": total
}])

df = pd.concat([df, new_entry], ignore_index=True)
print(df)

del domain