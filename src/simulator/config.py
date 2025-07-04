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
            )

            'JITTED': ValueHolder(
                name = 'jitted',
                default = True,
                help = (
                    'Setting this to false disables jax.jit of previously implicated functions.'
                )
            )

            'PARALLELISED': ValueHolder(
                name = 'jitted',
                default = True,
                help = (
                    'Setting this to false switches from diffrax to solve_ivp and disables parallelisation.'
                )
            )

            'SEEDED': ValueHolder(
                name = 'seeded',
                default = False,
                help = (
                    'Setting this to true seeds all random functions, useful for consistent benchmarking.'
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