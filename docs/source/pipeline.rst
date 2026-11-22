How integration works
=====================

`~symintegral.engine.pipeline.integrate` runs these stages for each integral. Each can be
switched off or configured with a keyword argument.

1. Transform (optional)
-----------------------

``transform="gaussian"``, ``"laplace"`` or ``"fourier"``. Factors
:math:`Q^{-\nu/2}`, with :math:`Q` quadratic in the integration
variables, couple the variables: :math:`1/|r_1 - r_2|` in Coulomb or
gravitational kernels is the typical case. The transform replaces such a
factor by an integral over a new *outer* variable, after which the
original variables appear only in Gaussians and decouple. For example

.. math::

   \int_{-\infty}^{\infty} \frac{e^{-x^2}}{\sqrt{1 + x^2}}\,dx
   = \frac{2}{\sqrt{\pi}} \int_0^\infty \left[\int_{-\infty}^{\infty}
     e^{-x^2 - t^2 (1 + x^2)}\,dx\right] dt .

See :doc:`transforms`.

2. Separation
-------------

``separate=True`` (default). On a rectangular domain (no limit depends
on another integration variable) the integrand is expanded and each
term split into factors that share no variables, so the integral becomes
a sum of products of lower-dimensional integrals:

.. math::

   \int_0^1\!\!\int_0^1 (x^2 y^2 + x^2)\,dx\,dy
   = \left(\int_0^1 x^2\,dx\right)\left(\int_0^1 y^2\,dy\right)
   + \left(\int_0^1 x^2\,dx\right)\left(\int_0^1 dy\right).

Every distinct factor is evaluated once and cached, so
:math:`\int_0^1 x^2\,dx` above is computed only once.

Domains that are not rectangular are integrated iteratively, innermost
variable first. Simplices can be made rectangular with
`~symintegral.domain.simplex_integral` (see below).

3. One variable at a time
-------------------------

For each factor:

- the closed-form rules (:doc:`rules`) are tried first, on any variable
  if the domain is rectangular, otherwise on the innermost one;
- if no rule applies, the back ends (:doc:`backends`) are tried in the
  order given by ``backends``;
- before each back-end step, Gaussian exponents are square-completed
  (``complete_square=True``) and, for the Sage back ends, the integrand
  is pre-simplified (``simplify="full"``).

With ``step_timeout=<seconds>`` every rule pass and back-end call is
interrupted after that time and the next method is tried.

4. Outer variables
------------------

Variables introduced by the transform are integrated last. For the
Gaussian and Laplace transforms the substitution
:math:`u^2 = t^2/(p + t^2)` first maps :math:`t \in [0, \infty)` onto
:math:`u \in [0, 1]` (``substitute_outer=True``); in quantum chemistry
this is the Boys-function form.

5. Result
---------

A closed form, or an expression containing ``Integral`` objects over the
variables nothing could integrate.

Building integrals over common domains
--------------------------------------

`~symintegral.engine.pipeline.integrate` takes any ``sympy.Integral``. Two helpers build one
from an integrand, volume element included:

`~symintegral.domain.chart_integral`
   Over a coordinate chart given as a mapping with the keys ``"limits"``
   and ``"volume_element"`` (e.g. spherical or prolate-spheroidal
   coordinates).

`~symintegral.domain.simplex_integral`
   Over the unit d-simplex (reference triangle, tetrahedron, ...),
   Duffy-collapsed onto :math:`[0, 1]^d`. This makes the domain
   rectangular, so separation applies and the integration order is free.
