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

To retrieve a list of random ingredients,
you can use the ``simulator.config.jax_init()`` function:

.. autofunction:: simulator.config.jax_init

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`src.simulator.config.jax_init`
will raise an exception.