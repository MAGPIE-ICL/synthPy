import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
args = parser.parse_args()

if args.rays is not None:
    Np = args.Rays
else:
    Np = 1e9

cores = None
if args.cores is not None:
    cores = args.cores

# attempts to fix path issues - need to find a resolution to the problem of relative paths on the HPC
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/')
#sys.path.insert(0, 'C:/Users/samma/programming/synthPy/src/')

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

#load hdf
ne, dims, spacing = utilIO.hdf_readin(str("/rds/general/user/sm5625/home/synthPy/evaluation/benchmarks/turbBoxLargeTest/radmeshablation_3d_prp_CH_ug_3rd_hdf5_plt_cnt_0228"))

'''
# multiply domain to match real size experimental target
multi_x = 0 # get multiplier x
multi_y = 0 # get multiplier y
multi_z = 0 # get multiplier z

for i in range(multi_x):
    ne = np.append(ne, ne, axis = 0)
for j in range(multi_y):
    ne = np.append(ne, ne, axis = 1)
for k in range(multi_z):
    ne = np.append(ne, ne, axis = 2)
'''

dims = ne.shape

extent_x = ((dims[0]*spacing[0].v)/2) * 1e-2
extent_y = ((dims[1]*spacing[1].v)/2) * 1e-2
extent_z = ((dims[2]*spacing[2].v)/2) * 1e-2

lengths = 2 * jnp.array([extent_x, extent_y, extent_z], dtype = jnp.int32)

ne_x = np.linspace(-extent_x,extent_x, dims[0])
ne_y = np.linspace(-extent_y,extent_y, dims[1])
ne_z = np.linspace(-extent_z,extent_z, dims[2])

print(f'extent x: {extent_x}')
print(f'extent y: {extent_y}')
print(f'extent z: {extent_z}')

print(f'dims x: {dims[0]}')
print(f'dims y: {dims[1]}')
print(f'dims z: {dims[2]}')

print("\n\n")

# is this baseline not decreasing after each run? - testing manually deleting objects first
baseline = memory_report()['used_raw']

probing_direction = 'z'
domain = d.ScalarDomain(lengths, dims, ne_type = "import", probing_direction = probing_direction, Np = Np, ne = ne.v * 1e6)

del ne_x
del ne_y
del ne_z

del ne

postDomain = memory_report()['used_raw']
domainAllocation = postDomain - baseline

plusRays = memory_report()['used_raw']

# define beam parameters
lwl = 1064e-9
beam_size = [extent_x, extent_y]    # beam radius
probing_extent = extent_z
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
divergence = 0.05e-3
beam_type = "rectangle"

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