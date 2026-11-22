"""
NumPy/SciPy source code from symintegral results.

`numpy_function` turns expressions (scalars, matrices, lists) into the
source of a Python function with common-subexpression elimination.
Unevaluated SymPy Integrals left in a partial result become calls to
``scipy.integrate.nquad`` over just the remaining variables, so a
reduced integral (e.g. a Boys-like 1-D remainder) is usable right away.
`module_source` assembles several functions into one module and
`compile_function` executes a source string and returns the function.
Special functions map to scipy.special; the `special_funcs` argument of
`numpy_function` lists the mappings added to those SymPy's SciPyPrinter
already knows.

Submodules
----------
generate
    `numpy_function`, `module_source`, `compile_function`.
printing
    `HEADER` and `SPECIAL_FUNCS` (SymPy -> scipy.special mappings).
layout
    Return value, signature and array-argument unpacking.
quadrature
    ``scipy.integrate.nquad`` code for unevaluated Integrals.
"""

from .generate import compile_function, module_source, numpy_function
from .printing import HEADER, SPECIAL_FUNCS

__all__ = ["numpy_function", "module_source", "compile_function",
           "HEADER", "SPECIAL_FUNCS"]
