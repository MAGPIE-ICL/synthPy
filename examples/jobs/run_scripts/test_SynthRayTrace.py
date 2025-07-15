import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support as vtk_np
import matplotlib.pyplot as plt
import gc
import argparse

import sys
#sys.path.insert(0, '/home/administrator/Work/UROP_ICL_Internship/synthPy/src/simulator')
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')     # import path/to/synthpy

import config
config.jax_init()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-f". "--force_device", type = str)
parser.add_argument("-m", "--memory", type = int)
args = parser.parse_args()

n_cells = 512
if args.domain is not None:
    n_cells = args.domain

Np = 1e7    # number of photons
if args.rays is not None:
    Np = args.rays

print("\nRunning job with a", n_cells, "domain and", Np, "rays.")
#print("Predicted size of domain is:", ((n_cells / 1024)**3) * 32 / 8)

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

import domain as d
import importlib
importlib.reload(d)

domain = d.ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos") # B_on = False by default

lwl = 1064e-9 #define laser wavelength

# initialise beam
divergence = 5e-5   # realistic divergence value
beam_size = extent_x    # beam radius
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
beam_type = 'circular'

import beam as beam_initialiser
importlib.reload(beam_initialiser)

import propagator as p
importlib.reload(p)

beam_definition = beam_initialiser.Beam(Np, beam_size, divergence, ne_extent, probing_direction = probing_direction, wavelength = lwl, beam_type = beam_type)

tracer = p.Propagator(domain, probing_direction = probing_direction, inv_brems = False, phaseshift = False)

# solve ray trace
tracer.calc_dndr(lwl)

if args.force_device is not None:
    force_device = args.force_device

if args.memory is not None:
    memory_debug = True

tracer.solve(beam_definition.s0, force_device = force_device, memory_debug = memory_debug)
print("\nCompleted ray trace in", np.round(tracer.duration, 3), "seconds.\n\n\n\n\n")