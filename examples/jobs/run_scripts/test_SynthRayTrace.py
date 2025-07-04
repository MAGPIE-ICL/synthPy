import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support as vtk_np
import matplotlib.pyplot as plt
import gc

import sys

#add path
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')     # import path/to/synthpy

import beam as beam_initialiser
import diagnostics as diag
import domain as d
import propagator as p
import utils

import importlib
importlib.reload(beam_initialiser)
importlib.reload(diag)
importlib.reload(d)
importlib.reload(p)
importlib.reload(utils)

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

n_cells = 512

probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])

domain = d.ScalarDomain(lengths, n_cells) # B_on = False by default

domain.test_exponential_cos()

lwl = 1064e-9 #define laser wavelength

# initialise beam
Np = 1e9    # number of photons
divergence = 5e-5   # realistic divergence value
beam_size = extent_x    # beam radius
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
beam_type = 'circular'

beam_definition = beam_initialiser.Beam(Np, beam_size, divergence, ne_extent, probing_direction, lwl, beam_type)

tracer = p.Propagator(domain, beam_definition.s0, probing_direction, inv_brems = False, phaseshift = False)

# solve ray trace
tracer.calc_dndr(lwl)
tracer.solve(parallelise = True, jitted = True)
print("\nCompleted ray trace in", np.round(tracer.duration, 3), "seconds.")