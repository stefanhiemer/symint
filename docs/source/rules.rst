Closed-form rules
=================

Rules are formulas for one variable over one range, tried before any
back end. They exist for three reasons:

Speed
   A multidimensional integral is split into many one-variable steps, and
   the same patterns recur: the Boys integral
   :math:`\int_0^1 u^{2n} e^{-T u^2}\,du` in every :math:`1/r_{12}`
   integral, or many polynomial terms in finite elements. A formula
   avoids a general algorithm and, for Maxima and FriCAS, the round trip
   through Sage.
Back-end gaps
   Some closed forms the back ends often miss, or find only with help;
   for instance Gaussians whose coefficient has a sign SymPy cannot
   prove, where Maxima asks sign questions.
Usable results
   Back-end results can be long (e.g. ``erf`` terms that cancel). A rule
   builds a compact result in SymPy and keeps the symbols' assumptions.

Whatever every rule declines goes to the back ends, so a rule only needs
to cover its own pattern.

Built-in rules
--------------

``"gaussian"`` (`~symintegral.rules.gaussian.gaussian_rule`)
   :math:`\int P(x)\, e^{-A x^2 + B x + C}\,dx`, with :math:`P` a
   polynomial, over the whole line and half-lines (:math:`A` provably
   positive) and over finite intervals (any :math:`A \neq 0`;
   ``erfi``-type results for :math:`A < 0`), via ``erf`` and a moment
   recurrence.

   Also linear exponents (:math:`A = 0`),
   :math:`\int P(x)\, e^{B x + C}\,dx`: over half-lines when the
   exponential provably decays, as sums of Gamma functions, and over
   finite intervals by the exact antiderivative. On :math:`[0, \infty)`
   an extra factor :math:`x^s` with symbolic or fractional :math:`s` is
   allowed (:math:`s + 1` provably positive), which covers Slater-type
   integrals

   .. math::

      \int_0^\infty x^n e^{-k x}\,dx = \frac{\Gamma(n+1)}{k^{n+1}}.

``"polynomial"`` (`~symintegral.rules.polynomial.polynomial_rule`)
   Polynomials over finite ranges, by their exact antiderivative: FEM
   integrands on affine or collapsed cells.

Context subpackages add more; see :doc:`contexts`.

The registry
------------

The rules live in the ordered list `~symintegral.rules.base.RULES`, consulted first to
last. To use other rules for one call, pass ``rules=[...]`` to
`~symintegral.engine.pipeline.integrate`; ``rules=()`` disables them.

Writing a rule
--------------

A `~symintegral.rules.base.Rule` has a name, a cheap test on the bounds, and the
evaluation itself:

.. code-block:: python

   Rule(name, applies, evaluate)

``applies(bounds)``
   Receives ``[lower, upper]`` and returns whether the rule could apply
   on this range at all.
``evaluate(expr, var, bounds)``
   Returns the integral of ``expr`` over ``var`` from ``bounds[0]`` to
   ``bounds[1]``, or ``None`` to decline. It must decline rather than
   return something it is not sure of.

`~symintegral.rules.base.register_rule` adds a rule at the end of the registry, or at
the front with ``first=True``; `~symintegral.rules.base.unregister_rule` removes it by
name. This rule handles Slater-type moments
:math:`\int_0^\infty x^n e^{-c x}\,dx = \Gamma(n+1)/c^{n+1}`:

.. literalinclude:: ../../examples/05_custom_rule.py
   :language: python
   :lines: 1-48

Check new rules against numerical quadrature, e.g. with
`~symintegral.verify.check_definite` (:doc:`verification`).
