"""
symintegral.qc -- quantum-chemistry extensions.

Importing this subpackage registers its closed-form rules in
`symintegral.RULES`:

"azimuthal" (`azimuthal_rule`)
    Full periods of exp(A + k cos(phi) + m sin(phi)) (Bessel I_0) and of
    (A + k cos(phi) + m sin(phi))^(-1/2) (elliptic K), the azimuthal
    part of 1/r12 in spherical or elliptic coordinates.

>>> import sympy as sp
>>> import symintegral.qc
>>> from symintegral import integrate
>>> phi = sp.Symbol("phi", real=True); k = sp.Symbol("k", positive=True)
>>> integrate(sp.Integral(sp.exp(k * sp.cos(phi)), (phi, 0, 2 * sp.pi)),
...           backends=())
2*pi*besseli(0, k)
"""

from .._util import is_full_period
from ..rules import RULES, Rule, register_rule
from .rules import azimuthal_rule

if not any(rule.name == "azimuthal" for rule in RULES):
    register_rule(Rule("azimuthal", is_full_period, azimuthal_rule))

__all__ = ["azimuthal_rule"]
