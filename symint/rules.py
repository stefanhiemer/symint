"""
rules.py -- closed-form integration rules, tried before any CAS.

A `Rule` integrates one variable over one range in closed form, or
declines (returns None) so that the next rule or the CAS takes over.
Rules live in the registry `RULES`, which the engine consults in order;
add your own with `register_rule`.

Built-in rules:

- "gaussian": moments int P(x) exp(-A x^2 + B x + C) dx over the whole
  line, half-lines and finite intervals (erf + recurrence).
- "azimuthal": full periods of exp(A + k cos + m sin) (Bessel I_0) and
  (A + k cos + m sin)^(-1/2) (elliptic K, the azimuthal 1/r12 kernel).
- "polynomial": polynomials in the variable over finite ranges (FEM
  integrands on affine or Duffy-collapsed cells), via exact
  antiderivatives.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Symbol

from ._util import (additive_terms, is_finite_range, is_full_period,
                    is_positive, is_whole_line)

Bounds = Sequence[Any]


@dataclass
class Rule:
    """
    A closed-form integration rule.

    Attributes
    ----------
    name : str
        Shown in the engine's log.
    applies : callable(bounds) -> bool
        Cheap test on the integration range.
    evaluate : callable(expr, var, bounds) -> sp.Expr or None
        The definite integral of `expr` over `var` in `bounds`, or None
        if the rule does not apply to this integrand.
    """

    name: str
    applies: Callable[[Bounds], bool]
    evaluate: Callable[[sp.Expr, Symbol, Bounds], Optional[sp.Expr]]


RULES: List[Rule] = []


def register_rule(rule: Rule,
                  first: bool = False) -> Rule:
    """
    Add a rule to the registry.

    Parameters
    ----------
    rule : Rule
        The rule; a registered rule of the same name is replaced.
    first : bool, optional
        Insert at the front (tried before the built-in rules) instead of
        appending.

    Returns
    -------
    Rule
        `rule`, so that the call can be used as an expression.
    """
    unregister_rule(rule.name)
    if first:
        RULES.insert(0, rule)
    else:
        RULES.append(rule)
    return rule


def unregister_rule(name: str) -> None:
    """
    Remove a rule from the registry.

    Parameters
    ----------
    name : str
        Name of the rule; nothing happens if no rule has this name.

    Returns
    -------
    None
    """
    RULES[:] = [r for r in RULES if r.name != name]


def apply_rules(expr: sp.Expr,
                var: Symbol,
                bounds: Bounds,
                rules: Sequence[Rule] = None) -> Optional[Tuple[str, sp.Expr]]:
    """
    Integrate with the first rule that can.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : sequence
        ``(lower, upper)``.
    rules : sequence of Rule, optional
        Rules to try in order; default the registry `RULES`.

    Returns
    -------
    (str, sp.Expr) or None
        Name of the successful rule and the integral, or None if every
        rule declined.
    """
    for rule in RULES if rules is None else rules:
        if rule.applies(bounds):
            result = rule.evaluate(expr, var, bounds)
            if result is not None:
                return rule.name, result
    return None


# --------------------------------------------------------------- polynomial

def polynomial_rule(expr: sp.Expr,
                    var: Symbol,
                    bounds: Bounds) -> Optional[sp.Expr]:
    """
    Exact integral of a polynomial over a finite range.

    Parameters
    ----------
    expr : sp.Expr
        Integrand, polynomial in `var` with arbitrary coefficients.
    var : Symbol
        Integration variable.
    bounds : sequence
        Finite ``(lower, upper)``.

    Returns
    -------
    sp.Expr or None
        F(upper) - F(lower) with F the antiderivative, or None if `expr`
        is not polynomial in `var`.

    Examples
    --------
    >>> import sympy as sp
    >>> x, a = sp.symbols("x a")
    >>> polynomial_rule(a * x**2, x, (0, 1))
    a/3
    """
    if not expr.has(var):
        return expr * (bounds[1] - bounds[0])
    if not expr.is_polynomial(var):
        return None
    F = sp.Poly(sp.expand(expr), var).integrate().as_expr()
    return F.subs(var, bounds[1]) - F.subs(var, bounds[0])


# ----------------------------------------------------------------- gaussian

def _erf_at(t: sp.Expr,
            root: sp.Expr) -> sp.Expr:
    """
    erf(root * t), with its limit +-1 at infinite `t`.

    Parameters
    ----------
    t : sp.Expr
        End point of the interval.
    root : sp.Expr
        sqrt(A).

    Returns
    -------
    sp.Expr
        sign(t) for infinite `t`, else erf(root t).
    """
    return sp.sign(t) if t.is_infinite else sp.erf(root * t)


def _boundary_term(t: sp.Expr,
                   k: int,
                   A: sp.Expr) -> sp.Expr:
    """
    t^k exp(-A t^2), with its limit 0 at infinite `t` (A > 0).

    Parameters
    ----------
    t : sp.Expr
        End point of the interval.
    k : int
        Power of t.
    A : sp.Expr
        Gaussian exponent.

    Returns
    -------
    sp.Expr
        The boundary term of the moment recurrence at `t`.
    """
    return sp.Integer(0) if t.is_infinite else t ** k * sp.exp(-A * t ** 2)


def _interval_moments(A: sp.Expr,
                      lo: sp.Expr,
                      hi: sp.Expr,
                      n: int) -> List[sp.Expr]:
    """
    Gaussian moments over an interval.

    M_k = int_lo^hi y^k exp(-A y^2) dy for k = 0..n, from

        M_0 = sqrt(pi)/(2 sqrt(A)) (erf(sqrt(A) hi) - erf(sqrt(A) lo)),
        M_1 = (exp(-A lo^2) - exp(-A hi^2)) / (2A),
        M_k = (lo^(k-1) exp(-A lo^2) - hi^(k-1) exp(-A hi^2)) / (2A)
              + (k-1)/(2A) M_(k-2),

    with erf -> +-1 and the boundary terms -> 0 at infinite ends (A > 0
    there). Analytic in A, so for finite ends any A != 0 works (A < 0
    gives erfi-type values through erf of imaginary arguments).

    Parameters
    ----------
    A : sp.Expr
        Gaussian exponent (nonzero; positive if an end is infinite).
    lo, hi : sp.Expr
        Interval ends, possibly -oo / oo.
    n : int
        Highest moment.

    Returns
    -------
    list of sp.Expr
        M_0, ..., M_n.
    """
    root = sp.sqrt(A)
    M = [sp.sqrt(sp.pi) / (2 * root) * (_erf_at(hi, root) - _erf_at(lo, root))]
    if n >= 1:
        M.append((_boundary_term(lo, 0, A) - _boundary_term(hi, 0, A))
                 / (2 * A))
    for k in range(2, n + 1):
        M.append((_boundary_term(lo, k - 1, A) - _boundary_term(hi, k - 1, A))
                 / (2 * A) + (k - 1) / (2 * A) * M[k - 2])
    return M


def _split_gaussian(term: sp.Expr,
                    var: Symbol) -> Optional[Tuple[sp.Expr, sp.Expr]]:
    """
    Split a term into polynomial and Gaussian exponent.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Integration variable.

    Returns
    -------
    (rest, exponent) or None
        With term = rest * exp(exponent), `rest` polynomial and
        `exponent` a quadratic polynomial in `var`; None otherwise.
    """
    exponent, rest = sp.Integer(0), sp.Integer(1)
    for f in sp.Mul.make_args(sp.powsimp(term, combine="exp")):
        if isinstance(f, sp.exp) and f.has(var):
            exponent += f.args[0]
        else:
            rest *= f
    exponent = sp.expand(exponent)
    if not (exponent.is_polynomial(var) and rest.is_polynomial(var)):
        return None
    if sp.Poly(exponent, var).degree() != 2:
        return None
    return rest, exponent


def _gaussian_term(term: sp.Expr,
                   var: Symbol,
                   lo: sp.Expr,
                   hi: sp.Expr) -> Optional[sp.Expr]:
    """
    int_lo^hi P(x) exp(-A x^2 + B x + C) dx for one term.

    Shifts y = x - B/(2A) and sums the moments of P(y + shift): Gamma
    functions over the whole line, `_interval_moments` otherwise.

    Parameters
    ----------
    term : sp.Expr
        One additive term, polynomial times Gaussian in `var`.
    var : Symbol
        Integration variable.
    lo, hi : sp.Expr
        Interval ends, possibly -oo / oo.

    Returns
    -------
    sp.Expr or None
        The integral, or None if the term is not of this form, an end is
        infinite and A is not provably positive, or A is zero.
    """
    split = _split_gaussian(term, var)
    if split is None:
        return None
    rest, exponent = split
    minus_A, B, C = [sp.cancel(c)
                     for c in sp.Poly(exponent, var).all_coeffs()]
    A = -minus_A
    infinite_end = lo.is_infinite or hi.is_infinite
    if infinite_end and not is_positive(A):
        return None
    if not infinite_end and (A == 0 or A.is_zero):
        return None
    shift = sp.cancel(B / (2 * A))
    y = sp.Dummy("y", real=True)
    shifted = sp.Poly(sp.expand(rest.subs(var, y + shift)), y)
    if lo == -sp.oo and hi == sp.oo:
        moments = sum(coeff * sp.gamma(sp.Rational(j + 1, 2))
                      / A ** sp.Rational(j + 1, 2)
                      for (j,), coeff in shifted.terms() if j % 2 == 0)
    else:
        M = _interval_moments(A, lo - shift, hi - shift, shifted.degree())
        moments = sum(coeff * M[j] for (j,), coeff in shifted.terms())
    return sp.exp(sp.cancel(B ** 2 / (4 * A) + C)) * moments


def gaussian_rule(expr: sp.Expr,
                  var: Symbol,
                  bounds: Bounds = None) -> Optional[sp.Expr]:
    """
    Integral of polynomials times Gaussians, term by term.

    int P(x) exp(-A x^2 + B x + C) dx over the whole line and half-lines
    (A provably positive) and over finite intervals (A not identically
    zero; generic in A, the A -> 0 limit is removable) -- e.g. the Boys
    integral int_0^1 u^(2n) exp(-T u^2) du for any sign of T. See
    `_gaussian_term`.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : sequence, optional
        ``(lower, upper)``; default the whole line.

    Returns
    -------
    sp.Expr or None
        The integral, or None if any term does not fit.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
    >>> gaussian_rule(x**2 * sp.exp(-a * x**2), x)
    sqrt(pi)/(2*a**(3/2))
    """
    if bounds is None or is_whole_line(bounds):
        lo, hi = -sp.oo, sp.oo
    else:
        lo, hi = (sp.sympify(bd) for bd in bounds)
    total = sp.Integer(0)
    for term in additive_terms(expr):
        value = _gaussian_term(term, var, lo, hi)
        if value is None:
            return None
        total += value
    return total


def _has_two_bounds(bounds: Bounds) -> bool:
    """
    Whether the range is definite (the gaussian rule's ``applies``).

    Parameters
    ----------
    bounds : sequence
        Integration range.

    Returns
    -------
    bool
        True for ``(lower, upper)``.
    """
    return len(bounds) == 2


# ---------------------------------------------------------------- azimuthal

def _pythagoras(expr: sp.Expr) -> sp.Expr:
    """
    Expand and replace every sin(x)^2 by 1 - cos(x)^2.

    Parameters
    ----------
    expr : sp.Expr
        Trigonometric expression.

    Returns
    -------
    sp.Expr
        `expr` without squared sines, so that sin^2 + cos^2 cancels.
    """
    e = sp.expand(expr)
    for s in list(e.atoms(sp.sin)):
        e = sp.expand(e.subs(s ** 2, 1 - sp.cos(s.args[0]) ** 2))
    return e


def _linear_in_cos_sin(expr: sp.Expr,
                       var: Symbol) -> Optional[Tuple[sp.Expr, ...]]:
    """
    Coefficients of `expr` as a linear function of cos(var), sin(var).

    Parameters
    ----------
    expr : sp.Expr
        Expression in `var`, e.g. an exponent or r12^2.
    var : Symbol
        The angle.

    Returns
    -------
    (A, kappa, mu) or None
        With expr = A + kappa cos(var) + mu sin(var) (sin^2 + cos^2 of
        other angles, e.g. phi_2 in r12^2, cancelled), or None.
    """
    c, s = sp.Dummy("c"), sp.Dummy("s")
    e = sp.expand(sp.expand_trig(expr)).subs({sp.cos(var): c,
                                              sp.sin(var): s})
    e = sp.expand(e.subs(s ** 2, 1 - c ** 2))
    if e.has(var) or not e.is_polynomial(c, s):
        return None
    poly = sp.Poly(e, c, s)
    if poly.total_degree() > 1:
        return None
    return tuple(_pythagoras(poly.coeff_monomial(m))
                 for m in ((0, 0), (1, 0), (0, 1)))


def _amplitude(kappa: sp.Expr,
               mu: sp.Expr) -> sp.Expr:
    """
    Amplitude sqrt(kappa^2 + mu^2) of kappa cos + mu sin.

    Parameters
    ----------
    kappa, mu : sp.Expr
        Coefficients of cos and sin.

    Returns
    -------
    sp.Expr
        The amplitude M, simplified with `_pythagoras`.
    """
    return sp.sqrt(sp.factor(_pythagoras(kappa ** 2 + mu ** 2)))


def _split_azimuthal(term: sp.Expr,
                     var: Symbol
                     ) -> Optional[Tuple[sp.Expr, sp.Expr, List[sp.Expr]]]:
    """
    Split a term into prefactor, exponent and inverse square roots.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        The angle.

    Returns
    -------
    (rest, exponent, roots) or None
        term = rest * exp(exponent) * prod(root**(-1/2)), `rest` free of
        `var`; None if a factor in `var` is of another kind.
    """
    exponent, rest, roots = sp.Integer(0), sp.Integer(1), []
    for f in sp.Mul.make_args(sp.powsimp(term, combine="exp")):
        if not f.has(var):
            rest *= f
        elif isinstance(f, sp.exp):
            exponent += f.args[0]
        elif f.is_Pow and f.exp == sp.Rational(-1, 2):
            roots.append(f.base)
        else:
            return None
    return rest, exponent, roots


def _azimuthal_term(term: sp.Expr,
                    var: Symbol) -> Optional[sp.Expr]:
    """
    Full-period integral of one term, see `azimuthal_rule`.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        The angle.

    Returns
    -------
    sp.Expr or None
        The integral, or None if the term does not fit.
    """
    split = _split_azimuthal(term, var)
    if split is None:
        return None
    rest, exponent, roots = split
    if exponent != 0:
        parts = _linear_in_cos_sin(exponent, var)
        if parts is None:
            return None
        if parts[1] == 0 and parts[2] == 0:
            rest *= sp.exp(parts[0])
            exponent = sp.Integer(0)
    if (roots and exponent != 0) or len(roots) > 1:
        return None
    if roots:
        parts = _linear_in_cos_sin(roots[0], var)
        if parts is None:
            return None
        A, kappa, mu = parts
        M = _amplitude(kappa, mu)
        return rest * 4 * sp.elliptic_k(2 * M / (A + M)) / sp.sqrt(A + M)
    if exponent == 0:
        return rest * 2 * sp.pi
    A, kappa, mu = _linear_in_cos_sin(exponent, var)
    return rest * 2 * sp.pi * sp.exp(A) * sp.besseli(0, _amplitude(kappa, mu))


def azimuthal_rule(expr: sp.Expr,
                   var: Symbol,
                   bounds: Bounds = None) -> Optional[sp.Expr]:
    """
    Integrals over a full period of an angle, term by term.

        int exp(A + k cos + m sin) dphi = 2 pi exp(A) I_0(M)
        int (A + k cos + m sin)^(-1/2) dphi = 4 K(2M/(A+M)) / sqrt(A+M)

    with M = sqrt(k^2 + m^2) and K in the parameter convention (SymPy's
    elliptic_k); the second needs A > M. Factors free of the angle are
    carried along.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    var : Symbol
        The angle.
    bounds : sequence, optional
        A full period (checked by the rule's ``applies``, not here).

    Returns
    -------
    sp.Expr or None
        The integral, or None if any term does not fit.

    Examples
    --------
    >>> import sympy as sp
    >>> phi = sp.Symbol("phi", real=True); k = sp.Symbol("k", positive=True)
    >>> azimuthal_rule(sp.exp(k * sp.cos(phi)), phi)
    2*pi*besseli(0, k)
    """
    total = sp.Integer(0)
    for term in additive_terms(expr):
        value = _azimuthal_term(term, var)
        if value is None:
            return None
        total += value
    return total


register_rule(Rule("gaussian", _has_two_bounds, gaussian_rule))
register_rule(Rule("azimuthal", is_full_period, azimuthal_rule))
register_rule(Rule("polynomial", is_finite_range, polynomial_rule))
