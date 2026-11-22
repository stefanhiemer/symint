Getting started
===============

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install symintegral              # SymPy + passagemath (Maxima/FriCAS)
   pip install "symintegral[numeric]"   # + NumPy/SciPy, to run generated code

For development, from a clone of the repository:

.. code-block:: bash

   pip install -e ".[test]"
   pytest

symintegral requires Python 3.10 or newer and runs on Linux and macOS
(the Maxima/FriCAS back ends come from passagemath, and the step
timeouts use Unix signals). Sage is imported only when a CAS step
actually runs, so rule-only calls start fast.

First integrals
---------------

`~symintegral.engine.pipeline.integrate` takes an unevaluated ``sympy.Integral``:

.. code-block:: python

   import sympy as sp
   from symintegral import integrate, simplex_integral

   x, y = sp.symbols("x y", real=True)
   a = sp.Symbol("a", positive=True)

   # a Gaussian moment, solved by the built-in Gaussian rule
   integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)))
   # sqrt(pi)/(2*a**(3/2))

   # a moment over the reference triangle
   integrate(simplex_integral(x**2 * y, (x, y)))
   # 1/60

Assumptions on symbols matter: ``a = Symbol("a", positive=True)`` lets
the Gaussian rule and the back ends know that the integral converges.

Partial results
---------------

If neither a rule nor a back end can integrate a variable, that part
stays an ``Integral``:

.. code-block:: python

   leftover = integrate(sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), backends=())
   # Integral(exp(-a*x**4), (x, 0, 1))

`~symintegral.codegen.generate.numpy_function` turns any result, partial or not, into NumPy
code; remaining integrals become ``scipy.integrate.nquad`` calls:

.. code-block:: python

   from symintegral import compile_function, numpy_function

   f = compile_function(numpy_function("f", leftover, [a]))
   f(1.0)   # 0.8448385947...

What happened?
--------------

Pass a list as ``log`` to see which rule or back end handled each step,
and why the others failed:

.. code-block:: python

   log = []
   integrate(sp.Integral(sp.exp(-x**2) * sp.cos(x), (x, 0, sp.oo)), log=log)
   log[0]["method"]   # "maxima"
   log[0]["errors"]   # methods that failed before it: {} here

Each entry has the keys ``"var"`` (the variable), ``"group"`` (the
variables of the separated factor it belongs to), ``"method"`` (the
successful rule or back end, or ``None``) and ``"errors"``.
