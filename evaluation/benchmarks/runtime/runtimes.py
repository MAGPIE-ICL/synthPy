import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--cores", type = int)
args = parser.parse_args()

domain = 128
if args.domain is not None:
    domain = args.domain

if args.rays is not None:
    rays = args.rays
else:
    rays = np.array([1e5, 1e6, 1e7, 1e8, 1e9])

cores = None
if args.cores is not None:
    cores = args.cores

sys.path.insert(0, '../../../src/simulator')
sys.path.insert(0, '../../../src/solvers-legacy')

import config
config.jax_init(core_limit = cores)

import jax.numpy as jnp

import importlib

import beam as beam_initialiser
import propagator as p
import diagnostics as diag

importlib.reload(beam_initialiser)
importlib.reload(p)
importlib.reload(diag)

import full_solver as fs
import rtm_solver as rtm

importlib.reload(fs)
importlib.reload(rtm)

from printing import colour

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

probing_extent = extent_z

lengths = 2 * jnp.array([extent_x, extent_y, extent_z])
dims = 128

lwl = 1064e-9

divergence = 5e-5
beam_size = ne_extent * 0.9
beam_type = "square"

times = np.array(2, len(rays))

for i, Np in enumerate(rays):
    domain = d.ScalarDomain(lengths, dims, ne_type = "test_exponential_cos", probing_direction = probing_direction)

    beam_definition = beam_initialiser.Beam(
        Np, beam_size,
        divergence,
        probing_extent,
        probing_direction = probing_direction,
        wavelength = lwl,
        beam_type = beam_type
    )

    _, _, duration = p.solve(beam_definition.s0, domain, probing_extent)

    times[0][i] = duration
    print(colour.BOLD + "\n\nDuration of" + times[0][i] + "sec for domain of size" + dims + "^3 and" + Np + "rays with updated solver." + colour.END)

for i, Np in enumerate(rays):
    slab = fs.ScalarDomain(ne_x, ne_y, ne_z, ne_extent)
    slab.test_exponential_cos(n_e0 = 2e17 * 1e6, Ly = 1e-3, s = -4e-3)
    slab.calc_dndr(lwl)

    ## Initialise rays and solve
    s0 = fs.init_beam(
        Np, beam_size, divergence, ne_extent,
        probing_direction = probing_direction,
        beam_type = beam_type
    )

    slab.solve(s0)

    times[1][i] = slab.duration
    print(colour.BOLD + "\n\nDuration of" + times[1][i] + "sec for domain of size" + dims + "^3 and" + Np + "rays with legacy solver." + colour.END)