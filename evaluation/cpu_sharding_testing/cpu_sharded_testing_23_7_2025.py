import numpy as np

import sys
import os

sys.path.insert(0, '../../src/simulator')

import importlib
import propagator as p
importlib.reload(p)
import domain as d
importlib.reload(d)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--core", type = int)
args = parser.parse_args()

n_cells = 128
if args.domain is not None:
    n_cells = args.domain

Np = 10000
if args.rays is not None:
    Np = args.rays

print("\nStarting cpu sharding test run with", n_cells, "cells and", Np, "rays.")

if args.core is not None:
    core_limit = args.core
else:
    from multiprocessing import cpu_count
    core_limit = cpu_count()

assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(core_limit)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

probing_extent = extent_z

lengths = 2 * np.array([extent_x, extent_y, extent_z])

import jax.numpy as jnp

from scipy.constants import c
from scipy.constants import e

domain = d.ScalarDomain(lengths, n_cells, ne_type = "test_exponential_cos")

lwl = 1064e-9

Np = 10000
divergence = 5e-5
beam_size = extent_x
ne_extent = probing_extent
beam_type = 'circular'

def init_beam(Np, beam_size, divergence, ne_extent):
    s0 = jnp.zeros((9, Np))

    t  = 2 * jnp.pi * np.random.randn(Np)

    u  = np.random.randn(Np)

    ϕ = jnp.pi * np.random.randn(Np)
    χ = divergence * np.random.randn(Np)

    s0 = s0.at[0, :].set(beam_size * u * jnp.cos(t))
    s0 = s0.at[1, :].set(beam_size * u * jnp.sin(t))
    s0 = s0.at[2, :].set(-ne_extent)

    s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
    s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
    s0 = s0.at[5, :].set(c * jnp.cos(χ))

    s0 = s0.at[6, :].set(1.0)
    s0 = s0.at[8, :].set(0.0)
    s0 = s0.at[7, :].set(0.0)

    return s0

beam_definition = init_beam(Np, beam_size, divergence, ne_extent)

rf, Jf, duration = p.solve(
    s0,
    (domain.x, domain.y, domain.z),
    (domain.x_n, domain.y_n, domain.z_n),   # domain.dim - this causes a TracerBoolConversionError, check why later, could be interesting and useful to know
    ne_extent,
    *p.calc_dndr(domain, lwl, keep_domain = True)
)

print("\nRun complete!")