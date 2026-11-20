"""
Closed-form rules for quantum-chemistry integrals.

"azimuthal" (`azimuthal_rule`): full periods of
exp(A + k cos(phi) + m sin(phi)) -> 2 pi e^A I_0(M) and of
(A + k cos(phi) + m sin(phi))^(-1/2) -> 4 K(2M/(A+M))/sqrt(A+M), the
azimuthal part of 1/r12; M = sqrt(k^2 + m^2).
"""

from typing import List, Optional, Tuple

import sympy as sp
from sympy import Symbol

from .._util import additive_terms
from ..rules.base import Bounds


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
