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

def resolve_path(path: str) -> str:
    # Split the path into parts
    parts = path.strip().split('/')
    stack = []

    for part in parts:
        if part == '' or part == '.':
            continue  # Skip empty or current dir
        elif part == '..':
            if stack:
                stack.pop()  # Go back one directory
        else:
            stack.append(part)

    # Handle root path (if path starts with '/')
    return '/' + '/'.join(stack) if path.startswith('/') else '/'.join(stack)

def jax_init(force_device = None, core_limit = None, extra_info = False, disable_python_multithreading = True, enable_x64 = False, debugging = False):
    import sys
    import os

    ### DO NOT REMOVE UNLESS YOU ARE VERY CERTAIN OF CORRECT PACKAGING

    # os.getcwd()                                   # - don't want cwd, want the dir of this file
    # os.path.dirname(os.path.realpath(__file__))   # cannot be called interactively - ? - seems's fine though
    # sys.path[0]                                   # haven't tested...
    # os.path.abspath(sys.argv[0])                  # haven't tested...

    top_level_path = resolve_path(str(os.path.dirname(os.path.realpath(__file__))) + "/../")
    print("Setting top level path for imports: " + top_level_path)

    # makes sure top level directory path is present in system so that relative imports work
    sys.path.insert(0, top_level_path)

    from shared.printing import colour
    print(colour.BOLD)

    # has to be disabled by default to prevent possible interference with jax parallelisation (caused issues on the cluster)
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

    # https://docs.jax.dev/en/latest/gpu_memory_allocation.html
    # can't set via jax.config.update as jax requires this to be initialised on first use

    #jax.config.update('jax_compiler_enable_remat_pass', False) # look into for future reference to debug mem use.

    if force_device == "cpu":
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    else:
        #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        #os.environ["TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC"] = "0.95"

    # triggers a jax breakpoint for debugging on error - works with filter_jit not jax.jit
    # if this is causing erroneous errors see equinox issue #1047: https://github.com/patrick-kidger/equinox/issues/1047
    if debugging:
        os.environ["EQX_ON_ERROR"] = "breakpoint"

    import jax

    # enables float data types to use 64-bit instead of 32 for greater precision
    # currently disabled by default as greater precision will vastly increase run times
    if enable_x64:
        print("\nWARNING: x64 bit currently disabled by default as greater precision will vastly increase run times")
        jax.config.update('jax_enable_x64', True)

    if debugging:
        jax.config.update('jax_traceback_filtering', 'off')
        # HPC doesn't recognise these config options due to old jax version (added in jax-0.6.0)
        # - you need to speak to RCS to get an updated version (either forcing them to do it or to find out how to do it yourself)
        jax.config.update('jax_captured_constants_report_frames', -1)
        jax.config.update('jax_captured_constants_warn_bytes', 128 * 1024 ** 2)

    print(colour.END)

    if extra_info:
        jax.print_environment_info()
        print("\n")

    # look further into what this actually means...
    print("Default jax backend:", jax.default_backend())

    available_devices = jax.devices()
    print(f"Available devices: {available_devices}")