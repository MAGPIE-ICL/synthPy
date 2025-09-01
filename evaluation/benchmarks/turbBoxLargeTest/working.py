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
parser.add_argument("-s", "--simPath", type = str)
parser.add_argument("-m", "--memoryLimit", type = str)
args = parser.parse_args()

if args.rays is not None:
    Np = args.rays
else:
    Np = 1e9

cores = None
if args.cores is not None:
    cores = args.cores

importPath = None
if args.importPath is not None:
    importPath = args.importPath
else:
    # attempts to fix path issues - need to find a resolution to the problem of relative paths on the HPC
    importPath = '/rds/general/user/sm5625/home/synthPy/src/'

simPath = None
if args.simPath is not None:
    simPath = args.simPath
else:
    simPath = importPath + "../evaluation/benchmarks/turbBoxLargeTest/radmeshablation_3d_prp_CH_ug_3rd_hdf5_plt_cnt_0228"

memoryLimit = None
if args.memoryLimit is not None:
    memoryLimit = args.memoryLimit

sys.path.insert(0, importPath)

import simulator.config as config
config.jax_init(core_limit = cores, jax_updated = False, extra_info = True, debugging = True)

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
df = pd.DataFrame(columns = columns)

# is this baseline not decreasing after each run? - testing manually deleting objects first
baseline = memory_report(memory_limit = memoryLimit)['used_raw']

probing_direction = 'z'
#domain = d.ScalarDomain(lengths, dims, ne_type = "import", probing_direction = probing_direction, Np = Np, ne = ne.v * 1e6, memory_limit = memoryLimit)

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

lengths = 2 * np.array([extent_x, extent_y, extent_z])
dims = 128
domain = d.ScalarDomain(lengths, dims, ne_type = "test_exponential_cos", probing_direction = probing_direction, Np = Np)

postDomain = memory_report(memory_limit = memoryLimit)['used_raw']
domainAllocation = postDomain - baseline

plusRays = memory_report(memory_limit = memoryLimit)['used_raw']

# define beam parameters
lwl = 1064e-9
beam_size = [extent_x, extent_y]    # beam radius
probing_extent = extent_z
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
divergence = 0.05e-3
beam_type = "rectangular"

beam_definition = beam_initialiser.Beam(
    Np, beam_size, divergence, ne_extent,
    probing_direction = probing_direction,
    beam_type = beam_type
)

rf, _, duration = p.solve((beam_size, divergence, ne_extent, probing_direction, beam_type, False), domain, probing_extent, verbose = False)

refractometer = diag.Refractometry(1032e-9, rf)
# cam't clear_mem if you want to generate other graphs afterwards
refractometer.plot_rays(bin_scale = 1, clear_mem = False)

#information accessed by .H(istogram) , e.g plt.imshow(refractometer.H)

#plt.imshow(refractometer.H, cmap='hot', interpolation='nearest', clim = (0, 2))
plt.imshow(refractometer.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))
plt.show()

from processing.plotting import general_ray_plots
general_ray_plots(rf, dims)

total = memory_report(memory_limit = memoryLimit)['used']

print(colour.BOLD + "\nDuration of " + str(duration) + " sec for domain of size " + str(dims) + " and " + str(int(Np)) + " rays with legacy solver." + colour.END)

new_entry = pd.DataFrame([{
    "dims": dims,
    "rays": Np,
    "runtime": duration,
    "legacyRuntime": "N/A",
    "domainSize": mem_conversion(domainAllocation),
    "raySize": mem_conversion(plusRays - domainAllocation),
    "totalMemory": total
}])

df = pd.concat([df, new_entry], ignore_index=True)
print(df)

del domain