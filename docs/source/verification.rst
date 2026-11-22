Checking results
================

`~symintegral.verify` checks results independently of how they were
obtained.

Antiderivatives
---------------

`~symintegral.verify.check_antiderivative` ``(F, f, x)`` checks
:math:`dF/dx = f`. Simplifying the residual :math:`r = F' - f` as a
whole is slow and sometimes inconclusive, so the check works in stages
and stops at the first that decides:

1. **Grouping.** :math:`r` is expanded and its terms are sorted by their
   transcendental factors: the product of factors that are not rational
   in :math:`x`, such as :math:`e^{-a x^2}`, :math:`\log(1+x)` or
   :math:`\operatorname{erf}(x)`. Exponentials are split first
   (:math:`e^{(x-b)^2} = e^{x^2} e^{-2bx} e^{b^2}`) and radicals reduced
   to one fractional power of their base, so the same factor written
   differently lands in the same group. Each group's rational
   coefficient is cancelled separately.
2. **Common denominator.** What remains is put over a common
   denominator and its numerator grouped again. This handles
   transcendental denominators such as :math:`1/(1 + e^x)`.
3. **simplify** on what is still left.
4. **Numerical test** at random points that respect the symbols'
   assumptions.

The result is a `~symintegral.verify.Check`: truthy if confirmed, with
``method`` set to ``"groups"``, ``"simplify"`` or ``"numeric"`` (the last
means likely, not proven), and the ``residual`` that was left.

.. code-block:: python

   import sympy as sp
   from symintegral import check_antiderivative

   x, a = sp.symbols("x a", positive=True)
   F = -(a * x + 1) * sp.exp(-a * x) / a**2
   check_antiderivative(F, x * sp.exp(-a * x), x).method   # "groups"
   check_antiderivative(x**2, x, x).ok                      # False

Definite integrals
------------------

`~symintegral.verify.check_definite` ``(result, integral)`` compares a result with
numerical quadrature of the original integral at random parameter values.
It is a numerical check only.

.. literalinclude:: ../../examples/06_verify.py
   :language: python
