Code generation
===============

Results, including partial ones, can be turned into NumPy/SciPy source
code. This needs the ``numeric`` extra (``pip install
"symintegral[numeric]"``).

`~symintegral.codegen.generate.numpy_function`
   Source of one function returning a scalar, a list or a matrix of
   expressions. Common subexpressions are computed once; array arguments
   are unpacked into their symbols; special functions map to
   ``scipy.special`` (``ellipk``, ``iv``, ``erf``, ``erfi``, ...); and
   remaining ``Integral`` objects become ``scipy.integrate.nquad`` calls
   (bounds that depend on outer variables included).

`~symintegral.codegen.generate.module_source`
   Joins several generated functions into one module, e.g. to write to a
   file.

`~symintegral.codegen.generate.compile_function`
   Executes generated source and returns the function.

Example: element mass matrices on the reference triangle, then NumPy code
for a variable coefficient.

.. literalinclude:: ../../examples/02_fem_element_matrices.py
   :language: python

Partial results are evaluated by quadrature:

.. literalinclude:: ../../examples/04_partial_results.py
   :language: python
