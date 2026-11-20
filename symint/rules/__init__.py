"""
Closed-form integration rules, tried before any CAS to accelerate solution
or cover gaps in the back ends (SymPy, Maxima, FriCAS) are general algorithms
A rule is worth writing if at least one of these holds:

1. Speed. Multidimensional integrals are split into many one-variable
   steps, and the same patterns recur (e.g. the Boys integral
   int_0^1 u^(2n) exp(-T u^2) du for every 1/r12 integral, or many
   polynomial terms in FEM). A formula avoids a general algorithm
   and, for Maxima/FriCAS, the round trip through Sage.
2. Back-end gaps. Some closed forms the back ends often miss or find
   only with help, e.g. the azimuthal Bessel/elliptic-K integrals of
   `symint.qc`, or Gaussians whose coefficient has a sign SymPy cannot
   prove (`_util.is_positive`), where Maxima asks sign questions.
3. Usable results. Back-end results can be long (e.g. erf terms that
   cancel); a rule returns a compact form built in SymPy, keeping the
   symbols' assumptions.

Rules complement the back ends rather than replace them: they cover a
narrow set of patterns, and whatever every rule declines goes to the
back ends. A rule is not worth writing for patterns that are rare or
that the back ends already handle quickly.

Unlike SymPy's internal methods, rules here are user-extensible via 
`register_rule`.

Submodules
----------
base
    The framework: `Rule`, the registry `RULES`, `register_rule`,
    `unregister_rule` and `apply_rules`.
gaussian
    "gaussian" (`gaussian_rule`): int P(x) exp(-A x^2 + B x + C) dx over
    the whole line, half-lines and finite intervals.
polynomial
    "polynomial" (`polynomial_rule`): polynomials over finite ranges.

Both built-in rules are registered on import, in that order.
Context-specific rules live in context subpackages and are registered
when those are imported (e.g. ``import symint.qc`` adds "azimuthal").
"""

from .._util import is_finite_range
from .base import RULES, Bounds, Rule, apply_rules, register_rule, unregister_rule
from .gaussian import _has_two_bounds, gaussian_rule
from .polynomial import polynomial_rule

register_rule(Rule("gaussian", _has_two_bounds, gaussian_rule))
register_rule(Rule("polynomial", is_finite_range, polynomial_rule))

__all__ = ["Bounds", "Rule", "RULES",
           "register_rule", "unregister_rule",
           "apply_rules",
           "gaussian_rule", "polynomial_rule",]
