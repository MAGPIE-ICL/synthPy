Usage
===================

.. _usage:

Regular usage
-------------------

To setup the jax environment,
you can use the ``simulator.config.jax_init()`` function:

.. autofunction:: simulator.config.jax_init

This will setup jax and in the default configuration for this program, it will also configure file paths for imports.
There are no required parameters - but there are many options.

After this, there are 3 main parts to running a simulation:
   -  Beam initialisation
   -  Domain generation
   -  Propagation

There is then post-processing to get sensible information and plots out of your raw data, but this will be covered separately.

Beam initialisation
^^^^^^^^^^^^^^^^^^^

Beam initialisation is essential down to instantiating an object from the ``simulator.beam.Beam`` class:

.. autoclass:: simulator.beam.Beam
   :members:

   .. automethod:: __init__

Domain generation
^^^^^^^^^^^^^^^^^^^

Domain generation is again essential down to instantiating an object from the ``simulator.domain.ScalarDomain`` class:

.. autoclass:: simulator.domain.ScalarDomain
   :members:

   .. automethod:: __init__

This class inherits from ``equinox.Module`` in order to work properly with jax, hence attributes cannot be added, removed or had their shape changed after creation.
To work around this you either have to re-create your instance of the ``simulator.domain.ScalarDomain`` class or update your current version using ``equinox.tree_at``

.. note::

   We recommend leaving this operation up to us, but if necessary, just re-create the object rather than messing with tree edits - it's likely safer for jax.
   (If you want to know why this is an issue then look here at jax tracers: https://docs.jax.dev/en/latest/key-concepts.html#tracing)

   We are considering switching this domain creation to a functional approach rather than class based to avoid this very issue and make its creation more versatile, so this may change in the future.

Propagation
^^^^^^^^^^^^^^^^^^^

For example:

>>> import synthPy
>>> synthPy.simulator.config.jax_init()

There is also a generic default run jupyter notebook in ``synthPy/examples/notebooks/SynthRayTracer.ipynb``

Batching
-------------------

Initial jax setup is the same as beforere-creating