class ValueHolder:
    def __init__(self, name, default, help):
        self.name = name
        self.default = default
        self.value = default
        self.help = help

    def set(self, value):
        self.value = value

class flags:
    def __init__(self):
        self.error_message = (f"Unrecognized config option: {name} - is this in the docs?, check the case?")

        self.value_holders: dict[str, ValueHolder] = {
            'MEMORY_DEBUG': ValueHolder(
                name = 'memory_debug',
                default = False,
                help = (
                    'Setting this to true enables memory profiling features to debug code.'
                )
            ),

            'JITTED': ValueHolder(
                name = 'jitted',
                default = True,
                help = (
                    'Setting this to false disables jax.jit of previously implicated functions.'
                )
            ),

            'PARALLELISED': ValueHolder(
                name = 'jitted',
                default = True,
                help = (
                    'Setting this to false switches from diffrax to solve_ivp and disables parallelisation.'
                )
            ),

            'SEEDED': ValueHolder(
                name = 'seeded',
                default = False,
                help = (
                    'Setting this to true seeds all random functions, useful for consistent benchmarking.'
                )
            ),

            'JAX_INITIALISED': ValueHolder(
                name = 'jax_initialised',
                default = False,
                help = (
                    'Should set this to true after importing jax for the first time so that any following imports will not try to set environment variables or assert conditions based on if jax is already imported.'
                )
            )
        }

    def update(self, name, value):
        if name not in self.value_holders:
            raise AttributeError(error_message.format(name = name))

        self.value_holders[name].set(value)

    def reset(self, name):
        if name not in self._value_holders:
            raise AttributeError(error_message.format(name = name))

        self.value_holders[name].set(self.value_holders[name].default)

    def reset_all(self):
        for i, (k, v) in enumerate(self.value_holders):
            self.value_holders[k].value = self.value_holders[k].default

def jax_init(force_device = None, core_limit = None, extra_info = False):
    import sys
    import os

    from utils import colour
    print(colour.BOLD)

    print("\nDisabling python multi-threading...")

    thread_count = str(1)
    os.environ["OMP_NUM_THREADS"]        = thread_count
    os.environ["OPENBLAS_NUM_THREADS"]   = thread_count
    os.environ["MKL_NUM_THREADS"]        = thread_count
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"]    = thread_count

    from multiprocessing import cpu_count

    print("\nInitialising jax...\n")

    ### THIS NEEDS TO BE SET BEFORE JAX IS INITIALISED IN ANY WAY, INCLUDING IMPORTING
    # - XLA_FLAGS are read WHEN jax is IMPORTED

    assert "jax" not in sys.modules, "jax already imported: you must restart your runtime - DO NOT RUN THIS FUNCTION TWICE"
    # bring up issue to see if it can be made a on the run configurable variable
    #jax.config.update('xla_force_host_platform_device_count', self.core_count)

    core_count = cpu_count()
    if core_limit is not None:
        if core_limit > core_count:
            print("\nWARNING: Core limit was set greater than the number of available cores. Defaulting to max available.")
        else:
            core_count = core_limit

    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=" + str(core_count)
    #os.environ['JAX_ENABLE_X64'] = "True"
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

    # triggers a jax breakpoint for debugging on error - works with filter_jit not jax.jit
    #os.environ["EQX_ON_ERROR"] = "breakpoint"

    if force_device == "cpu":
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    else:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        #os.environ["TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC"] = "0.95"

    import jax

    # defaults float data types to 64-bit instead of 32 for greater precision
    jax.config.update('jax_enable_x64', True)
    # HPC doesn't recognise this config option
    #jax.config.update('jax_captured_constants_report_frames', -1)
    #jax.config.update('jax_captured_constants_warn_bytes', 128 * 1024 ** 2)
    jax.config.update('jax_traceback_filtering', 'off')
    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    #jax.config.update('xla_python_client_preallocate', False)
    #jax.config.update('xla_python_client_allocator', '\"platform\"')
    # can't set via jax.config.update for some reason

    #jax.config.update('jax_compiler_enable_remat_pass', False)

    print(colour.END)

    if extra_info:
        jax.print_environment_info()

    # look further into what this actually means...
    print("\nDefault jax backend:", jax.default_backend())

    available_devices = jax.devices()
    print(f"Available devices: {available_devices}")