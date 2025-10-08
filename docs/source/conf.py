# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath('../../NiTROM'))

project   = 'NiTROM'
copyright = "2024, Alberto Padovan"
author    = "Alberto Padovan"
release   = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     "sphinx.ext.duration",
#     "sphinx.ext.doctest",
#     "sphinx.ext.autodoc",
#     "sphinx.ext.autosummary",
#     "sphinx.ext.mathjax",
#     "sphinx.ext.napoleon",
#     "sphinx.ext.viewcode",
#     "sphinx_gallery.gen_gallery",
# ]

# sphinx_gallery_conf = {
#      'examples_dirs': [
#          '../../NiTROM/examples/cavity_flow',
#          '../../NiTROM/examples/cavity/cgl',
#          #'../../NiTROM/examples/toymodel',
#      ],  # path to your example scripts
#      'gallery_dirs': [
#          'auto_examples/cavity_flow',
#          'auto_examples/cgl',
#          'auto_examples/toymodel',
#      ],             # path to where to save gallery generated output
# }

# html_logo = "logo.png"
html_show_sphinx = False
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
