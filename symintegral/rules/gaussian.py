"""
"gaussian" rule: polynomials times exponentials of quadratic or linear
exponents.

- Gaussians, int P(x) exp(-A x^2 + B x + C) dx, over the whole line,
  half-lines and finite intervals, via erf plus a moment recurrence.
  Covers the Boys integral int_0^1 u^(2n) exp(-T u^2) du.
- Linear exponents, int P(x) exp(B x + C) dx, over half-lines (Gamma
  functions, e.g. the Slater-type int_0^oo x^n exp(-k x) dx =
  Gamma(n + 1)/k^(n + 1), n also symbolic) and finite intervals (exact
  antiderivative).
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
                    var: Symbol) -> Optional[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """
    Split a term into a power of `var`, a polynomial prefactor and an
    exponent,

        term = var^power * prefactor(var) * exp(exponent(var)),

    where prefactor is a polynomial in `var` (other symbols allowed),
    exponent is quadratic or linear in `var`, and power collects the
    factors var^s whose exponent s is not a nonnegative integer (symbolic,
    fractional or negative; 0 if there are none). Terms not of this form
    give None.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Integration variable.

    Returns
    -------
    (power, prefactor, exponent) or None
        As above; None otherwise.
    """
    exponent, power, prefactor = sp.Integer(0), sp.Integer(0), sp.Integer(1)
    # exp(..var..) into exponent, var**s (s not a nonnegative integer)
    # into power, everything else to prefactor
    for f in sp.Mul.make_args(sp.powsimp(term, combine="exp")):
        base, exp_ = f.as_base_exp()
        if isinstance(f, sp.exp) and f.has(var):
            exponent += f.args[0]
        elif base == var and not (exp_.is_Integer and exp_ >= 0):
            power += exp_
        else:
            prefactor *= f
    exponent = sp.expand(exponent)
    #
    if not (exponent.is_polynomial(var) and prefactor.is_polynomial(var)):
        return None
    elif sp.Poly(exponent, var).degree() not in (1, 2):
        return None
    else:
        return power, prefactor, exponent


def _exponential_term(power: sp.Expr,
                      prefactor: sp.Expr,
                      B: sp.Expr,
                      C: sp.Expr,
                      var: Symbol,
                      lo: sp.Expr,
                      hi: sp.Expr) -> Optional[sp.Expr]:
    """
    Integrate one term with a linear exponent,

        int_lo^hi var^power * prefactor(var) * exp(B var + C) dvar,

    with prefactor a polynomial in `var` (see `_split_gaussian`):

    - [lo, oo), B provably negative, k = -B: with y = var - lo and
      prefactor(y + lo) = sum_m c_m y^m,
      exp(B lo + C) sum_m c_m m!/k^(m + 1).
    - (-oo, hi], B provably positive: the mirror image, y = hi - var.
    - [0, oo) with power != 0 (e.g. symbolic n in x^n), power + 1
      provably positive:
      exp(C) sum_m c_m Gamma(power + m + 1)/k^(power + m + 1).
    - finite [lo, hi], B not zero, power 0: the exact antiderivative
      exp(B x) sum_m (-1)^m P^(m)(x)/B^(m + 1) of P(x) exp(B x).

    Parameters
    ----------
    power : sp.Expr
        Exponent of the extra factor var^power (0 if none).
    prefactor : sp.Expr
        Polynomial in `var`.
    B, C : sp.Expr
        Coefficients of the exponent B var + C.
    var : Symbol
        Integration variable.
    lo, hi : sp.Expr
        Interval ends, possibly -oo / oo.

    Returns
    -------
    sp.Expr or None
        The integral, or None if the sign conditions cannot be proven,
        the range is the whole line, or a non-polynomial power is
        combined with a range other than [0, oo).
    """
    if lo.is_infinite and hi.is_infinite:
        return None  # diverges for every B != 0
    # var^power with power not a nonnegative integer: only [0, oo)
    elif power != 0:
        if not (lo == 0 and hi == sp.oo and is_positive(-B)
                and is_positive(power + 1)):
            return None
        k = -B
        poly = sp.Poly(prefactor, var)
        return sp.exp(C) * sum(
            c * sp.gamma(power + m + 1) / k ** (power + m + 1)
            for (m,), c in poly.terms())
    # half-lines: Gamma integrals int_0^oo y^m exp(-k y) dy = m!/k^(m+1)
    elif hi == sp.oo or lo == -sp.oo:
        if hi == sp.oo:
            k, end, y_of = -B, lo, (lambda y: y + lo)
        else:
            k, end, y_of = B, hi, (lambda y: hi - y)
        if not is_positive(k):
            return None
        y = sp.Dummy("y", real=True)
        shifted = sp.Poly(sp.expand(prefactor.subs(var, y_of(y))), y)
        return sp.exp(sp.expand(B * end + C)) * sum(
            c * sp.factorial(m) / k ** (m + 1)
            for (m,), c in shifted.terms())
    # finite interval: exact antiderivative (generic in B, B = 0 removable)
    elif B == 0 or B.is_zero:
        return None
    else:
        poly = sp.Poly(prefactor, var)
        derivatives, d = [], poly
        while not d.is_zero:
            derivatives.append(d)
            d = d.diff(var)
        def antiderivative(t):
            return sp.exp(B * t) * sum((-1) ** m * dm.as_expr().subs(var, t)
                                       / B ** (m + 1)
                                       for m, dm in enumerate(derivatives))
        return sp.exp(C) * (antiderivative(hi) - antiderivative(lo))


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
    power, prefactor, exponent = split
    # exponent linear in var: no Gaussian, see _exponential_term
    if sp.Poly(exponent, var).degree() == 1:
        B, C = [sp.cancel(c) for c in sp.Poly(exponent, var).all_coeffs()]
        return _exponential_term(power, prefactor, B, C, var, lo, hi)
    elif power != 0:
        return None
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
    Integral of polynomials times Gaussians or exponentials, term by term.

    int P(x) exp(-A x^2 + B x + C) dx over the whole line and half-lines
    (A provably positive) and over finite intervals (A not identically
    zero; generic in A, the A -> 0 limit is removable) -- e.g. the Boys
    integral int_0^1 u^(2n) exp(-T u^2) du for any sign of T. See
    `_gaussian_term`.

    Linear exponents (A = 0), int P(x) exp(B x + C) dx, over half-lines
    (decaying: B provably negative on [lo, oo), positive on (-oo, hi])
    and finite intervals (B not identically zero). On [0, oo) a factor
    x^s with symbolic or fractional s is allowed if s + 1 is provably
    positive: Slater-type integrals int_0^oo x^n exp(-k x) dx =
    Gamma(n + 1)/k^(n + 1). See `_exponential_term`.

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
    >>> n = sp.Symbol("n", nonnegative=True)
    >>> gaussian_rule(x**n * sp.exp(-a * x), x, (0, sp.oo))
    a**(-n - 1)*gamma(n + 1)
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
