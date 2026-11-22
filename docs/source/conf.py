# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import symintegral  # noqa: E402

# -- Project information -----------------------------------------------------

project = "symintegral"
copyright = "2026, Stefan Hiemer"
author = "Stefan Hiemer"
release = symintegral.__version__

# -- General configuration ---------------------------------------------------

extensions = ["myst_parser",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "sphinx_autodoc_typehints"]
myst_enable_extensions = ["colon_fence",
                          "dollarmath",
                          "deflist"]

# NumPy-style docstrings; single backticks refer to Python objects
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
default_role = "py:obj"
autodoc_member_order = "bysource"
# document each object once, where it is defined, not again where it is
# re-exported via __all__
autodoc_default_options = {"members": True,
                           "ignore-module-all": True,
                           "undoc-members": False,
                           "show-inheritance": True}

templates_path = ["_templates"]
exclude_patterns = ["build"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
