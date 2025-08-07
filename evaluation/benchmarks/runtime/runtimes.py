import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dims", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
args = parser.parse_args()

dims = 128
if args.dims is not None:
    dims = args.dims

if args.rays is not None:
    rays = np.array(args.rays).astype(np.int32)
else:
    rays = np.array([1e5, 1e6, 1e7, 1e8, 1e9], dtype = np.int32)

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

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

# legacy
ne_x = np.linspace(-extent_x, extent_x, dims)
ne_y = np.linspace(-extent_y, extent_y, dims)
ne_z = np.linspace(-extent_z, extent_z, dims)
ne_extent = extent_z

# updated
lengths = 2 * jnp.array([extent_x, extent_y, extent_z])

# general
beam_size = extent_z * 0.9
divergence = 5e-5
probing_extent = extent_z
probing_direction = "z"
lwl = 1064e-9
beam_type = "square"

times = np.zeros((2, len(rays)))

for i in range(len(rays)):
    domain = d.ScalarDomain(lengths, dims, ne_type = "test_exponential_cos", probing_direction = probing_direction)

    beam_definition = beam_initialiser.Beam(
        rays[i], beam_size,
        divergence,
        probing_extent,
        probing_direction = probing_direction,
        wavelength = lwl,
        beam_type = beam_type
    )

    _, _, duration = p.solve(beam_definition.s0, domain, probing_extent)


    slab = fs.ScalarDomain(ne_x, ne_y, ne_z, ne_extent)
    slab.test_exponential_cos(n_e0 = 2e17 * 1e6, Ly = 1e-3, s = -4e-3)
    slab.calc_dndr(lwl)

    ## Initialise rays and solve
    s0 = fs.init_beam(
        rays[i], beam_size, divergence, ne_extent,
        probing_direction = probing_direction,
        beam_type = beam_type
    )

    slab.solve(s0)

    times[0][i] = duration
    times[1][i] = slab.duration

    print(colour.BOLD + "\n\nDuration of " + str(times[1][i]) + " sec for domain of size " + str(dims) + " ^3 and " + str(rays[i]) + " rays with legacy solver." + colour.END)
    print(colour.BOLD + "\n\nDuration of " + str(times[0][i]) + " sec for domain of size " + str(dims) + " ^3 and " + str(rays[i]) + " rays with updated solver.\n" + colour.END)

for i in range(len(rays)):
    print(colour.BOLD + "\n\nDuration of " + str(times[0][i]) + " sec for domain of size " + str(dims) + " ^3 and " + str(rays[i]) + " rays with updated solver." + colour.END)
    print(colour.BOLD + "\n\nDuration of " + str(times[1][i]) + " sec for domain of size " + str(dims) + " ^3 and " + str(rays[i]) + " rays with legacy solver.\n" + colour.END)
