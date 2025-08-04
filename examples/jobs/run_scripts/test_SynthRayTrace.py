import numpy as np
import argparse

import sys
#sys.path.insert(0, '/home/administrator/Work/UROP_ICL_Internship/synthPy/src/simulator')
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')     # import path/to/synthpy

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-f", "--force-device", type = str)
parser.add_argument("-m", "--memory", type = str)
#parser.add_argument("-a", "--auto-batching", type = Bool)
parser.add_argument("-c", "--cores", type = int)
args = parser.parse_args()

force_device = None
if args.force_device is not None:
    force_device = args.force_device

cores = None
if args.cores is not None:
    cores = args.cores

import config
config.jax_init(force_device = force_device, core_limit = cores)

import jax

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

domain = d.ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos", auto_batching = False) # B_on = False by default

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

with jax.checking_leaks():
    beam_definition = beam_initialiser.Beam(
        Np,
        beam_size,
        divergence,
        ne_extent,
        probing_direction = probing_direction,
        wavelength = lwl,
        beam_type = "circular"
    )

    memory_debug = False
    if args.memory is not None:
        if args.memory.upper() == "TRUE":
            memory_debug = True
        elif args.memory.upper() == "FALSE":
            memory_debug = False
        else:
            pass    # error handling?

    rf, Jf, duration = p.solve(
        beam_definition.s0,
        domain.coordinates,
        (domain.x, domain.y, domain.z),
        (domain.x_n, domain.y_n, domain.z_n),   # domain.dim - this causes a TracerBoolConversionError, check why later, could be interesting and useful to know
        ne_extent,
        *p.calc_dndr(domain, lwl, keep_domain = True),
        memory_debug = memory_debug
    )

    print("\nCompleted ray trace in", np.round(duration, 3), "seconds.\n\n\n\n\n")