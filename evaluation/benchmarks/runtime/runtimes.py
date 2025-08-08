import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dims", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
args = parser.parse_args()

if args.dims is not None:
    dims = np.array(args.dims).astype(np.int32)
else:
    #dims = np.array([128, 256, 512], dtype = np.int32)
    dims = np.array([128], dtype = np.int32)

if args.rays is not None:
    rays = np.array(args.rays).astype(np.int32)
else:
    #rays = np.array([1e5, 1e6, 1e7, 1e8, 1e9], dtype = np.int32)
    rays = np.array([1e5, 1e6, 1e7], dtype = np.int32)

cores = None
if args.cores is not None:
    cores = args.cores

# attempts to fix path issues - need to find a resolution to the problem of relative paths on the HPC
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/')

import simulator.config as config
config.jax_init(core_limit = cores, jax_updated = False)

import jax.numpy as jnp

import importlib

import simulator.beam as beam_initialiser
import simulator.domain as d
import simulator.propagator as p
import processing.diagnostics as diag

importlib.reload(beam_initialiser)
importlib.reload(d)
importlib.reload(p)
importlib.reload(diag)

import legacy.full_solver as fs
import legacy.rtm_solver as rtm

importlib.reload(fs)
importlib.reload(rtm)

from shared.printing import colour
from shared.utils import memory_report

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

# legacy
ne_extent = extent_z

# updated - was reporting isse with lengths when actually was with dims, was passing full array not just one vale
# --> this is the issue with python not declaring expected types in functions, SO MUCH undefined behaviour
lengths = 2 * jnp.array([extent_x, extent_y, extent_z], dtype = jnp.int32)

# general
beam_size = extent_z * 0.9
divergence = 5e-5
probing_extent = extent_z
probing_direction = "z"
lwl = 1064e-9
beam_type = "square"

columns = ["dims", "rays", "runtime", "legacyRuntime", "domainSize", "raySize", "totalMemory"]
df = pd.DataFrame(columns=columns)

dims_len = len(dims)
rays_len = len(rays)

for i in range(dims_len):
    ne_x = np.linspace(-extent_x, extent_x, dims[i])
    ne_y = np.linspace(-extent_y, extent_y, dims[i])
    ne_z = np.linspace(-extent_z, extent_z, dims[i])

    for j in range(rays_len):
        print("\n\n\n")

        baseline = memory_report()['used']

        domain = d.ScalarDomain(lengths, dims[i], ne_type = "test_exponential_cos", probing_direction = probing_direction)

        postDomain = memory_report()['used']
        domainAllocation = postDomain - baseline

        beam_definition = beam_initialiser.Beam(
            rays[j], beam_size,
            divergence,
            probing_extent,
            probing_direction = probing_direction,
            wavelength = lwl,
            beam_type = beam_type
        )

        plusRays = memory_report()['used']

        _, _, duration = p.solve(beam_definition.s0, domain, probing_extent)

        total = memory_report()['used']



        slab = fs.ScalarDomain(ne_x, ne_y, ne_z, ne_extent)
        slab.test_exponential_cos(n_e0 = 2e17 * 1e6, Ly = 1e-3, s = -4e-3)
        slab.calc_dndr(lwl)

        ## Initialise rays and solve
        s0 = fs.init_beam(
            rays[j], beam_size, divergence, ne_extent,
            probing_direction = probing_direction,
            beam_type = beam_type
        )

        slab.solve(s0)



        print(colour.BOLD + "\n\nDuration of " + str(duration) + " sec for domain of size " + str(dims[j]) + " ^3 and " + str(rays[j]) + " rays with legacy solver." + colour.END)
        print(colour.BOLD + "Duration of " + str(slab.duration) + " sec for domain of size " + str(dims[j]) + " ^3 and " + str(rays[j]) + " rays with updated solver.\n" + colour.END)

        new_entry = pd.DataFrame([{
            "dims": dims[j],
            "rays": rays[j],
            "runtime": duration,
            "legacyRuntime": slab.duration,
            "domainSize": domainAllocation,
            "raySize": plusRays - domainAllocation,
            "totalMemory": total
        }])

        df = pd.concat([df, new_entry], ignore_index=True)

for i in range(dims_len):
    for j in range(rays_len):
        k = j + i * rays_len

        print(colour.BOLD + "\n\nDuration of " + str(df['runtime'][k]) + " sec for domain of size " + str(df['dims'][k]) + " ^3 and " + str(df['rays'][k]) + " rays with updated solver." + colour.END)
        print(colour.BOLD + "Duration of " + str(df['legacyRuntime'][k]) + " sec for domain of size " + str(df['dims'][k]) + " ^3 and " + str(df['rays'][k]) + " rays with legacy solver.\n" + colour.END)

df.to_csv("benchmark_results" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv", index=False)