import numpy as np

import sys
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
parser.add_argument("-r", "--rays", type = int)
parser.add_argument("-c", "--core", type = int)
args = parser.parse_args()

n_cells = 16
if args.domain is not None:
    n_cells = args.domain

Np = 4
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

os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax

with jax.checking_leaks():
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_traceback_filtering', 'off')

    extent_x = 5e-3
    extent_y = 5e-3
    extent_z = 10e-3

    probing_extent = extent_z

    lengths = 2 * np.array([extent_x, extent_y, extent_z])

    from scipy.constants import c
    from scipy.constants import e

    class ScalarDomain():
        def __init__(self, lengths, dim):
            self.x_length, self.y_length, self.z_length = lengths[0], lengths[1], lengths[2]
            self.x_n, self.y_n, self.z_n = dim, dim, dim

            self.x = np.float32(np.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
            self.y = np.float32(np.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
            self.z = np.float32(np.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))
            self.coordinates = np.stack([self.x, self.y, self.z], axis = 1)

            self.XX, self.YY, _ = np.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)
            self.ZZ = None

            self.XX = self.XX / 2e-3
            self.XX = 10 ** self.XX

            self.YY = self.YY / 1e-3
            self.YY = np.pi * self.YY
            self.YY = 2 * self.YY
            self.YY = np.cos(self.YY)
            self.YY = 1 + self.YY

            self.ne = self.XX * self.YY

            self.ne = 1e24 * self.ne

    domain = ScalarDomain(lengths, n_cells)

    lwl = 1064e-9

    divergence = 5e-5
    beam_size = extent_x
    ne_extent = probing_extent
    beam_type = 'circular'

    def init_beam(Np, beam_size, divergence, ne_extent):
        s0 = np.zeros((9, Np))

        t  = 2 * np.pi * np.random.randn(Np)

        u  = np.random.randn(Np)

        ϕ = np.pi * np.random.randn(Np)
        χ = divergence * np.random.randn(Np)

        s0[0, :] = beam_size * u * np.cos(t)
        s0[1, :] = beam_size * u * np.sin(t)
        s0[2, :] = -ne_extent

        s0[3, :] = c * np.sin(χ) * np.cos(ϕ)
        s0[4, :] = c * np.sin(χ) * np.sin(ϕ)
        s0[5, :] = c * np.cos(χ)

        s0[6, :] = 1.0
        s0[8, :] = 0.0
        s0[7, :] = 0.0

        return s0

    beam_definition = init_beam(Np, beam_size, divergence, ne_extent)

    from scipy.integrate import odeint, solve_ivp
    from time import time

    def trilinearInterpolator(coordinates, length, dim, values, query_points, *, fill_value = np.nan):
        idr_o = np.clip(np.floor(((query_points / length) + 0.5) * (np.asarray(dim, dtype = np.int64) - 1)).astype(np.int64), 0, len(coordinates) - 2)    # enforcing that it should be an array of integers to index with
        wr_o = (query_points - coordinates[idr_o[:, np.arange(3)], np.arange(3)]) / (coordinates[idr_o[:, np.arange(3)] + 1, np.arange(3)] - coordinates[idr_o[:, np.arange(3)], np.arange(3)])

        def get_indices_and_weights(coord_grid, points):
            idx = np.searchsorted(coord_grid, points, side = 'right') - 1
            idx = np.clip(idx, 0, len(coord_grid) - 2)

            x0 = coord_grid[idx]
            return idx, (points - x0) / (coord_grid[idx + 1] - x0)

        ix, wx = get_indices_and_weights(coordinates[:, 0], query_points[:, 0])
        iy, wy = get_indices_and_weights(coordinates[:, 1], query_points[:, 1])
        iz, wz = get_indices_and_weights(coordinates[:, 2], query_points[:, 2])

        idr = np.array([ix, iy, iz]).T

        print(idr_o)
        print(idr)

        offsets = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])  # shape: (8, 3)

        # idr has shape (N, 3) --> None's convert both arrays to matching shape for 8 3D offsets of N points
        neighbors = idr[:, None, :] + offsets[None, :, :]   # shape: (N, 8, 3)

        val_neighbors = values[
            neighbors[:, :, 0], 
            neighbors[:, :, 1], 
            neighbors[:, :, 2]
        ]  # shape: (N, 8)

        #wx, wy, wz = wr[:, 0], wr[:, 1], wr[:, 2]  # shape: (N, 1)
        weights = np.stack([
            (1 - wx) * (1 - wy) * (1 - wz),  # 000
            wx       * (1 - wy) * (1 - wz),  # 100
            (1 - wx) * wy       * (1 - wz),  # 010
            (1 - wx) * (1 - wy) * wz,        # 001
            wx       * wy       * (1 - wz),  # 110
            wx       * (1 - wy) * wz,        # 101
            (1 - wx) * wy       * wz,        # 011
            wx       * wy       * wz         # 111
        ], axis = 1)  # shape: (N, 8)

        return np.sum(weights * val_neighbors, axis = 1)# / 8

    def calc_dndr(ScalarDomain, lwl = 1064e-9):
        omega = 2 * np.pi * c / lwl
        nc = 3.14207787e-4 * omega ** 2

        return (np.array(ScalarDomain.ne / nc, dtype = np.float32), omega)

    def dndr(r, ne, omega, coordinates, length, dim):
        grad = np.zeros_like(r)

        dndx = -0.5 * c ** 2 * np.gradient(ne / (3.14207787e-4 * omega ** 2), coordinates[:, 0], axis = 0)
        grad[0, :] = trilinearInterpolator(coordinates, length, dim, dndx, r.T, fill_value = 0.0)
        del dndx

        #dndy = -0.5 * c ** 2 * np.gradient(ne / (3.14207787e-4 * omega ** 2), coordinates[:, 1], axis = 1)
        #grad[1, :] = trilinearInterpolator(coordinates, length, dim, dndy, r.T, fill_value = 0.0)
        #del dndy

        #dndz = -0.5 * c ** 2 * np.gradient(ne / (3.14207787e-4 * omega ** 2), coordinates[:, 2], axis = 2)
        #grad[2, :] = trilinearInterpolator(coordinates, length, dim, dndz, r.T, fill_value = 0.0)
        #del dndz

        return grad

    def dsdt(s, ne, coordinates, omega, length, dim):
        s = np.reshape(s, (9, 1))
        sprime = np.zeros_like(s)

        r = s[:3, :]
        v = s[3:6, :]

        sprime[3:6, :] = dndr(r, ne, omega, coordinates, length, dim)
        sprime[:3, :] = v

        del r
        del v

        return sprime.flatten()

#print("Ray:", beam_definition[:, 0])

ne, omega = calc_dndr(domain)
value = dsdt(beam_definition[:, 0], ne, domain.coordinates, omega, (domain.x_length, domain.y_length, domain.z_length), (domain.x_n, domain.y_n, domain.z_n))