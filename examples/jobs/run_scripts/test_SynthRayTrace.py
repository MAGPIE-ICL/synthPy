import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support as vtk_np
import matplotlib.pyplot as plt
import gc

import sys

#add path
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')     # import path/to/synthpy

import importlib

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

n_cells = 512

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

import domain as d
importlib.reload(d)

domain = d.ScalarDomain(lengths, n_cells) # B_on = False by default

domain.test_exponential_cos()

lwl = 1064e-9 #define laser wavelength

# initialise beam
# force to interpret as 64 bit integer instead of float - should adjust code to convert it to an integer if not already
Np = np.int64(1e9)    # number of photons
divergence = 5e-5   # realistic divergence value
beam_size = extent_x    # beam radius
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
beam_type = 'circular'

import beam as beam_initialiser
importlib.reload(beam_initialiser)

beam_definition = beam_initialiser.Beam(Np, beam_size, divergence, ne_extent, probing_direction = probing_direction, wavelength = lwl, beam_type = beam_type)

importlib.reload(p)
import propagator as p

tracer = p.Propagator(domain, beam_definition.s0, probing_direction = probing_direction, inv_brems = False, phaseshift = False, parallelise = True)

# solve ray trace
tracer.calc_dndr(lwl)
tracer.solve(jitted = True)
print("\nCompleted ray trace in", np.round(tracer.duration, 3), "seconds.")