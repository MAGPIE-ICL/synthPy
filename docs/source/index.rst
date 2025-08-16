Welcome to synthPy's documentation!
===================================

.. include:: ../../README.md
   :parser: myst_parser.sphinx_

.. note::

   This project is under active development.

Check out the :doc:`installation` section to find out how to get access to synthPy on your computer.
Check out the :doc:`usage` section for further information and examples.

.. automodapi:: EoN

Contents
--------

.. toctree::

   installation
   usage
   api


..
   If you want to register modules they actually HAVE TO BE MODULES
   Should figure out which ones you want to be modules or not (if you want any - I doubt it)
   Then re-organise directories into that structure and give each directory its own __init__.py so it will be detected as a module

..
   Modules
   =======

   .. autosummary::
      :toctree: modules

..
   simulator.config

   simulator.beam
   simulator.domain
   simulator.propagator

   simulator.interpolator
   simulator.fresnel_integral

   processing.diagnostics
   processing.plotting

   shared.printing
   shared.propagation
   shared.utils

..
   Auto-Summary
   ------------

   .. autosummary::
      :toctree: _autosummary
      :template: custom-module-template.rst
      :recursive:

      simulator
      legacy

      field_generator

      processing

      shared
      utils