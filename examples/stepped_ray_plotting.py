import numpy as np
import matplotlib.pyplot as plt

import sys

#add path
sys.path.insert(0, '../')     # import path/to/synthpy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dims", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-s", "--samples", type = int)
parser.add_argument("-n", "--stepResolution", type = int)
args = parser.parse_args()

n_cells = 128
if args.dims is not None:
    n_cells = args.dims

Np = 100000
if args.rays is not None:
    Np = args.rays

samples = 64
if args.samples is not None:
    samples = args.samples

stepResolution = 128
if args.stepResolution is not None:
    stepResolution = args.stepResolution

import src.simulator.config as config
config.jax_init()

import simulator.beam as beam_initialiser
import simulator.domain as d
import simulator.propagator as p
import processing.diagnostics as diag

import importlib
importlib.reload(beam_initialiser)
importlib.reload(d)
importlib.reload(p)
importlib.reload(diag)

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

#x = np.linspace(-extent_x, extent_x, n_cells)
#y = np.linspace(-extent_y, extent_y, n_cells)
#z = np.linspace(-extent_z, extent_z, n_cells)

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

#domain = d.ScalarDomain(x = x, y = y, z = z, extent = probing_extent, probing_direction = probing_direction)     # create domain
# Much simpler domain function, no longer needlessly takes in beam values, they are fully seperated
domain = d.ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos", probing_direction = probing_direction) # B_on = False by default

lwl = 1064e-9 #define laser wavelength

# initialise beam
divergence = 5e-5   # realistic divergence value
beam_size = extent_x    # beam radius
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
beam_type = 'circular'

beam_definition = beam_initialiser.Beam(
    Np, beam_size, divergence, ne_extent,
    probing_direction = probing_direction,
    wavelength = lwl,
    beam_type = "circular"
)

rf, Jf, duration = p.solve(beam_definition.s0, domain, probing_extent, save_points_per_region = stepResolution, return_raw_results = True)

from processing.plotting import stepped_ray_plot
stepped_ray_plot(rf, domain, sample_size = samples)