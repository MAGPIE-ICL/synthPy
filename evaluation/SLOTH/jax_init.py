import sys
import os

class colour:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

print(colour.BOLD)

# has to be disabled by default to prevent possible interference with jax parallelisation (caused issues on the cluster)
disable_python_multithreading = True
if disable_python_multithreading:
    print("Disabling python multi-threading...")

    thread_count = str(1)
    os.environ["OMP_NUM_THREADS"]        = thread_count
    os.environ["OPENBLAS_NUM_THREADS"]   = thread_count
    os.environ["MKL_NUM_THREADS"]        = thread_count
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"]    = thread_count

from multiprocessing import cpu_count

print("Initialising jax...")

assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"

core_count = cpu_count()
core_limit = None
if core_limit is not None:
    if core_limit > core_count:
        print("\nWARNING: Core limit was set greater than the number of available cores. Defaulting to max available.")
    else:
        core_count = core_limit

os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(core_count)

force_device = None
if force_device == "cpu":
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
else:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax

# enables float data types to use 64-bit instead of 32 for greater precision
# currently disabled by default as greater precision will vastly increase run times
enable_x64 = False
if enable_x64:
    print("\nWARNING: x64 bit currently disabled by default as greater precision will vastly increase run times")
    jax.config.update('jax_enable_x64', True)

print(colour.END)

print("Default jax backend:", jax.default_backend())

available_devices = jax.devices()
print(f"Available devices: {available_devices}")