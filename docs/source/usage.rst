Usage
=====

.. _installation:

Installation
------------

To use synthpy, first install it using pip:

.. code-block:: console

   (.venv) $ pip install synthPy

Regular usage
----------------

To setup the jax environment,
you can use the ``simulator.config.jax_init()`` function:

.. autofunction:: simulator.config.jax_init

There are no required parameters.

For example:

>>> import synthPy
>>> synthPy.simulator.config.jax_init()

Batching
----------------

Initial jax setup is the same as before.