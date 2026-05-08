##
## Configuration file for the Sphinx documentation builder.
##

# -- Project information

project = 'synthPy'
copyright = '2025, Imperial College London'
author = 'Sam MacKay'

release = '0.1'
version = '0.1.0'

##
## General configuration
##

# docs build without this, there are warnings though - use to harden against these and ensure everything is picked up

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',   # core library for html generation from docstrings
    'sphinx.ext.autosummary',   # create neat summary tables
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_automodapi.automodapi'
]

autosummary_generate = False  # Turn on sphinx.ext.autosummary

# maps referenced functions automatically to other documentation with intersphinx
# - can set others I am sure...
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'equinox': ('https://docs.kidger.site/equinox/', None)
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# suffixes to distinguish the README's
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}