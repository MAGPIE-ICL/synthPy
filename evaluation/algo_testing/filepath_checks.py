import sys
import os

print("CWD", os.getcwd())

from multiprocessing import cpu_count

print("\nInitialising jax...\n")

assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(cpu_count())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

jax.print_environment_info()

print("\nDefault jax backend:", jax.default_backend())

available_devices = jax.devices()
print(f"Available devices: {available_devices}")

folder_name = "memory_benchmarks/"
rel_path_to_folder = "../../evaluation/"

path = rel_path_to_folder + folder_name

if os.path.isdir(os.getcwd() + "/" + folder_name):
    print("a")
    path = folder_name
elif os.path.isdir(os.getcwd() + "/" + path):
    print("b")
    pass
else:
    print("c")
    path = os.getcwd() + "/" + folder_name

    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

from datetime import datetime
path += "memory-domain" + " - " + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
jax.profiler.save_device_memory_profile(path)

print("\nResultant path:", path)