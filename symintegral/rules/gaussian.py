"""
"gaussian" rule: moments int P(x) exp(-A x^2 + B x + C) dx over the
whole line, half-lines and finite intervals, via erf plus a moment
recurrence. Covers the Boys integral int_0^1 u^(2n) exp(-T u^2) du.
"""

from typing import List, Optional, Tuple

import sympy as sp
from sympy import Symbol

from .._util import additive_terms, is_positive, is_whole_line
from .base import Bounds


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
    if t.is_infinite:
        return sp.Integer(0)
    else:
        return  t ** k * sp.exp(-A * t ** 2)


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
    Split a term into polynomial prefactor and Gaussian exponent,

        term = prefactor(var) * exp(exponent(var)),

    where prefactor is a polynomial in `var` (other symbols allowed) and
    exponent is quadratic in `var`. Terms not of this form give None.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Integration variable.

    Returns
    -------
    (prefactor, exponent) or None
        With term = prefactor * exp(exponent), `prefactor` polynomial and
        `exponent` a quadratic polynomial in `var`; None otherwise.
    """
    exponent, prefactor = sp.Integer(0), sp.Integer(1)
    # exp(..var..) into exponent, everything else to prefactor
    for f in sp.Mul.make_args(sp.powsimp(term, combine="exp")):
        if isinstance(f, sp.exp) and f.has(var):
            exponent += f.args[0]
        else:
            prefactor *= f
    exponent = sp.expand(exponent)
    #
    if not (exponent.is_polynomial(var) and prefactor.is_polynomial(var)):
        return None
    elif sp.Poly(exponent, var).degree() != 2:
        return None
    else:
        return prefactor, exponent


def _gaussian_term(term: sp.Expr,
                   var: Symbol,
                   lo: sp.Expr,
                   hi: sp.Expr) -> Optional[sp.Expr]:
    """
    Integrate one Gaussian term,

        int_lo^hi prefactor(var) * exp(-A var^2 + B var + C) dvar,

    with prefactor a polynomial in `var` (see `_split_gaussian`).

    Completes the square, y = var - B/(2A), and sums the moments of
    prefactor(y + B/(2A)): Gamma functions over the whole line,
    `_interval_moments` otherwise.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Integration variable.
    lo, hi : sp.Expr
        Interval ends, possibly -oo / oo.

    Returns
    -------
    sp.Expr or None
        The integral, or None if the term is not of this form, an end is
        infinite and A is not provably positive, or both ends are finite
        and A is zero.
    """
    split = _split_gaussian(term, var)
    if split is None:
        return None
    prefactor, exponent = split
    # exponent = -A var^2 + B var + C
    minus_A, B, C = [sp.cancel(c)
                     for c in sp.Poly(exponent, var).all_coeffs()]
    A = -minus_A
    # infinite ends need a decaying Gaussian; finite ends a nonzero A
    infinite_end = lo.is_infinite or hi.is_infinite
    if (infinite_end and not is_positive(A)) or \
       (not infinite_end and (A == 0 or A.is_zero)):
        return None
    # complete the square: prefactor as polynomial in y = var - shift
    else:
        shift = sp.cancel(B / (2 * A))
        y = sp.Dummy("y", real=True)
        shifted = sp.Poly(sp.expand(prefactor.subs(var, y + shift)), y)
        # sum_j c_j int y^j exp(-A y^2) dy (odd moments vanish on the line)
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
