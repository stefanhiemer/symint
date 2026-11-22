Context subpackages
===================

The core of symintegral is general-purpose. Rules for one field of
application live in context subpackages, which register them when
imported.

Quantum chemistry: ``symintegral.qc``
-------------------------------------

.. code-block:: python

   import symintegral.qc   # registers "azimuthal" for all later integrate() calls

``"azimuthal"`` (`~symintegral.qc.rules.azimuthal_rule`) evaluates full periods of an
angle, term by term:

.. math::

   \int_0^{2\pi} e^{A + k\cos\varphi + m\sin\varphi}\,d\varphi
   = 2\pi\, e^A I_0(M),
   \qquad
   \int_0^{2\pi} \frac{d\varphi}{\sqrt{A + k\cos\varphi + m\sin\varphi}}
   = \frac{4\, K\!\left(\frac{2M}{A+M}\right)}{\sqrt{A + M}},

with :math:`M = \sqrt{k^2 + m^2}`, :math:`K` in SymPy's parameter
convention (``elliptic_k``) and :math:`A > M` for the second. These are
the azimuthal parts of :math:`1/r_{12}` in spherical or elliptic
coordinates.

.. literalinclude:: ../../examples/07_qc_azimuthal.py
   :language: python

Writing a context subpackage
----------------------------

A context subpackage is a regular subpackage of ``symintegral`` whose
``__init__`` registers its rules with `~symintegral.rules.base.register_rule`, guarded so
that importing it twice does not register them twice:

.. code-block:: python

   from ..rules import RULES, Rule, register_rule
   from .rules import my_rule

   if not any(rule.name == "my_rule" for rule in RULES):
       register_rule(Rule("my_rule", applies, my_rule))
