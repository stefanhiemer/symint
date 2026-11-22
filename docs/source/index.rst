symintegral documentation
=========================

**symintegral** evaluates multidimensional SymPy
integrals in closed form, as far as possible.

It never hands the whole integral to a computer algebra system. Instead
it rewrites the integral, splits it into small one-variable problems, and
solves these with fast closed-form rules where it can and with SymPy,
Maxima or FriCAS otherwise. Whatever cannot be evaluated stays a SymPy
``Integral``: you get a partial result, never an exception, and that
partial result can be turned into NumPy/SciPy code that does the rest by
quadrature.

.. code-block:: python

   import sympy as sp
   from symintegral import integrate

   x = sp.Symbol("x", real=True)
   a = sp.Symbol("a", positive=True)
   integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)))
   # sqrt(pi)/(2*a**(3/2))

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   pipeline
   rules
   backends
   transforms
   codegen
   verification
   contexts
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   symintegral
   roadmap
