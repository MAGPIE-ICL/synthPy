import jax
import jax.numpy as jnp
import time
from functools import partial
 
### ========== 1. Create Synthetic Test Grid ==========
 
def generate_test_grid(nx, ny, nz):
    x = jnp.linspace(-1, 1, nx)
    y = jnp.linspace(-1, 1, ny)
    z = jnp.linspace(-1, 1, nz)
    grid_x, grid_y, grid_z = jnp.meshgrid(x, y, z, indexing='ij')
    values = jnp.sin(jnp.pi * grid_x) * jnp.cos(jnp.pi * grid_y) * jnp.exp(-grid_z**2)
    return values, jnp.stack([x, y, z], axis=1)
 
# Grid dimensions
dim = (64, 64, 64)
length = (2.0, 2.0, 2.0)  # [-1, 1] in each direction
values, coordinates = generate_test_grid(*dim)
 
# Random query points inside domain
key = jax.random.PRNGKey(0)
query_points = jax.random.uniform(key, (100000000, 3), minval=-1.0, maxval=1.0)
 
### ========== 2. Original Implementation ==========

def trilinearInterpolator_original(coordinates, length, dim, values, query_points, *, fill_value=jnp.nan):
    idr = jnp.clip(jnp.floor(((query_points / jnp.asarray(length)) + 0.5) * (jnp.asarray(dim, dtype=jnp.int32) - 1)).astype(jnp.int32), 0, coordinates.shape[0] - 2)
    wr = (query_points - coordinates[idr[:, jnp.arange(3)], jnp.arange(3)]) / (
        coordinates[idr[:, jnp.arange(3)] + 1, jnp.arange(3)] - coordinates[idr[:, jnp.arange(3)], jnp.arange(3)]
    )
    offsets = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    neighbors = idr[:, None, :] + offsets[None, :, :]
    val_neighbors = values[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]
    wx, wy, wz = wr[:, 0], wr[:, 1], wr[:, 2]
    return (
        val_neighbors[:, 0] * (1 - wx) * (1 - wy) * (1 - wz) +
        val_neighbors[:, 1] * wx       * (1 - wy) * (1 - wz) +
        val_neighbors[:, 2] * (1 - wx) * wy       * (1 - wz) +
        val_neighbors[:, 3] * (1 - wx) * (1 - wy) * wz       +
        val_neighbors[:, 4] * wx       * wy       * (1 - wz) +
        val_neighbors[:, 5] * wx       * (1 - wy) * wz       +
        val_neighbors[:, 6] * (1 - wx) * wy       * wz       +
        val_neighbors[:, 7] * wx       * wy       * wz
    )
 
### ========== 3. Optimized Version ==========

def get_cube(idr, values):
    return jax.lax.dynamic_slice(values, idr, (2, 2, 2))
 
def trilinear(cube, wx, wy, wz):
    return (
        cube[0, 0, 0] * (1 - wx) * (1 - wy) * (1 - wz) +
        cube[1, 0, 0] * wx       * (1 - wy) * (1 - wz) +
        cube[0, 1, 0] * (1 - wx) * wy       * (1 - wz) +
        cube[0, 0, 1] * (1 - wx) * (1 - wy) * wz       +
        cube[1, 1, 0] * wx       * wy       * (1 - wz) +
        cube[1, 0, 1] * wx       * (1 - wy) * wz       +
        cube[0, 1, 1] * (1 - wx) * wy       * wz       +
        cube[1, 1, 1] * wx       * wy       * wz
    )
 
def trilinearInterpolator_optimized(coordinates, length, dim, values, query_points, *, fill_value=jnp.nan):
    norm_pos = ((query_points / jnp.asarray(length)) + 0.5) * (jnp.asarray(dim) - 1)
    idr = jnp.clip(jnp.floor(norm_pos).astype(jnp.int32), 0, jnp.asarray(dim) - 2)
 
    axis = jnp.arange(3)
    i = idr[:, axis]
    coord0 = coordinates[i, axis]
    coord1 = coordinates[i + 1, axis]
    wr = (query_points[:, axis] - coord0) / (coord1 - coord0)

    return jax.vmap(trilinear)(jax.vmap(get_cube, in_axes=(0, None))(idr, values), wr[:, 0], wr[:, 1], wr[:, 2])

### ========== 4. Working Version ==========
 
def trilinearInterpolator_working(coordinates, length, dim, values, query_points, *, fill_value=jnp.nan):
    norm_pos = ((query_points / jnp.asarray(length)) + 0.5) * (jnp.asarray(dim) - 1)
    idr = jnp.clip(jnp.floor(norm_pos).astype(jnp.int32), 0, jnp.asarray(dim) - 2)
 
    axis = jnp.arange(3)
    i = idr[:, axis]
    coord0 = coordinates[i, axis]
    coord1 = coordinates[i + 1, axis]
    wr = (query_points[:, axis] - coord0) / (coord1 - coord0)

    offsets = jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])

    neighbors = idr[:, None, :] + offsets[None, :, :]
    val_neighbors = values[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]

    wx, wy, wz = wr[:, 0], wr[:, 1], wr[:, 2]

    return (
        val_neighbors[:, 0] * (1 - wx) * (1 - wy) * (1 - wz) +
        val_neighbors[:, 1] * wx       * (1 - wy) * (1 - wz) +
        val_neighbors[:, 2] * (1 - wx) * wy       * (1 - wz) +
        val_neighbors[:, 3] * (1 - wx) * (1 - wy) * wz       +
        val_neighbors[:, 4] * wx       * wy       * (1 - wz) +
        val_neighbors[:, 5] * wx       * (1 - wy) * wz       +
        val_neighbors[:, 6] * (1 - wx) * wy       * wz       +
        val_neighbors[:, 7] * wx       * wy       * wz
    )
 
### ========== 5. Benchmarking Helper ==========
 
def benchmark(fn, name="Function"):
    fn()  # Warm-up (especially for JIT)
    start = time.time()
    result = fn()
    jax.block_until_ready(result)
    elapsed = time.time() - start
    print(f"{name:<25}: {elapsed:.4f} seconds")
 
### ========== 6. Run Benchmarks ==========
 
print("Running trilinear interpolation benchmark...\n")
 
# JIT versions
#jit_original = jax.jit(trilinearInterpolator_original)
jit_optimized = jax.jit(trilinearInterpolator_optimized)
jit_working = jax.jit(trilinearInterpolator_working)
 
# Partial bind fixed args for cleaner timing
args = (coordinates, length, dim, values, query_points)
#benchmark(lambda: trilinearInterpolator_original(*args),    "Original (No JIT)")
#benchmark(lambda: jit_original(*args),                      "Original (JIT)")
#benchmark(lambda: trilinearInterpolator_optimized(*args), "Optimized (No JIT)")
#benchmark(lambda: jit_optimized(*args), "Optimized (JIT)")
benchmark(lambda: trilinearInterpolator_working(*args), "Working (No JIT)")
benchmark(lambda: jit_working(*args), "Working (JIT)")