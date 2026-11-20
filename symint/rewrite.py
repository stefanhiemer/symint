"""
rewrite.py -- exact rewrites of integrands before integration.

- Completing the square in Gaussian exponents (`gaussian_shift`,
  `common_shift`, `complete_square`), so that rules and CAS see
  exp(-c (x - s)^2) instead of exp(-c x^2 + b x).
- Integral transforms of inverse distances (`apply_transform`): factors
  Q^(-nu/2), Q quadratic in the integration variables, are replaced by
  an integral over a new variable whose integrand is Gaussian
  ("gaussian", Boys), exponential ("laplace") or a plane wave
  ("fourier") in Q, which makes the spatial integrals tractable.
- The Boys substitution (`boys_substitute`) of the leftover transform
  variable onto [0, 1], giving the Boys-function form of the result.
"""

from functools import partial
from typing import Dict, List, Sequence, Set, Tuple

import sympy as sp
from sympy import Symbol

from ._util import additive_terms, fresh, is_exp, tidy_exponent

TRANSFORMS = ("gaussian", "laplace", "fourier")

Limits = List[Tuple]


def _exponent(term: sp.Expr) -> sp.Expr:
    """
    The combined exponent of all exponentials in a product.

    Parameters
    ----------
    term : sp.Expr
        One additive term.

    Returns
    -------
    sp.Expr
        Sum of the arguments of the exponential factors of `term` (0 if
        there are none).
    """
    merged = sp.powsimp(term, combine="exp")
    return sp.Add(*[f.args[0] for f in sp.Mul.make_args(merged)
                    if isinstance(f, sp.exp)])


def _quadratic_coefficients(term: sp.Expr,
                            var: Symbol) -> Tuple[sp.Expr, sp.Expr]:
    """
    Quadratic and linear coefficient of the Gaussian exponent of `term`.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Integration variable.

    Returns
    -------
    (c2, c1) or None
        Coefficients of the exponent c2 var^2 + c1 var + c0, or None if
        the exponent is not a quadratic polynomial in `var`.
    """
    exponent = _exponent(term)
    if not (exponent.has(var) and exponent.is_polynomial(var)):
        return None
    poly = sp.Poly(sp.expand(exponent), var)
    if poly.degree() != 2:
        return None
    return poly.coeff_monomial(var ** 2), poly.coeff_monomial(var)


def gaussian_shift(term: sp.Expr,
                   var: Symbol) -> sp.Expr:
    """
    Shift that removes the linear part of a Gaussian exponent.

    Parameters
    ----------
    term : sp.Expr
        One additive term with exponent c2 var^2 + c1 var + c0.
    var : Symbol
        Integration variable.

    Returns
    -------
    sp.Expr or None
        The real shift -c1/(2 c2) (0 without linear part), or None if
        `term` has no such exponent, c2 is not provably negative or the
        shift is not provably real.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
    >>> gaussian_shift(sp.exp(-a * x**2 + 2 * x), x)
    1/a
    """
    coefficients = _quadratic_coefficients(term, var)
    if coefficients is None:
        return None
    c2, c1 = coefficients
    if not (-c2).is_positive:
        return None
    shift = sp.cancel(-c1 / (2 * c2))
    return shift if shift.is_real else None


def common_shift(expr: sp.Expr,
                 var: Symbol) -> sp.Expr:
    """
    The Gaussian shift shared by all terms of `expr`.

    On a half-line the shift moves the finite bound, so it can only be
    applied if every term has the same one.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.

    Returns
    -------
    sp.Expr or None
        The common, nonzero shift (`gaussian_shift`), or None.
    """
    shifts = {gaussian_shift(t, var) for t in additive_terms(expr)}
    if len(shifts) != 1:
        return None
    shift = shifts.pop()
    return None if shift is None or shift == 0 else shift


def complete_square(expr: sp.Expr,
                    var: Symbol) -> sp.Expr:
    """
    Complete the square in Gaussian exponents, term by term.

    Each term with exponent c2 var^2 + c1 var + c0 (c2 provably negative,
    c1 != 0, real shift) is rewritten with var -> var - c1/(2 c2). This
    is exact over (-oo, oo) whatever the rest of the term; terms that do
    not fit are left alone.

    Parameters
    ----------
    expr : sp.Expr
        Integrand over the whole line.
    var : Symbol
        Integration variable.

    Returns
    -------
    sp.Expr
        Integrand with the same integral over (-oo, oo).

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
    >>> complete_square(sp.exp(-a * x**2 + 2 * x), x)
    exp(-a*(x + 1/a)**2 + 2*x + 2/a)
    """
    shifted = []
    for term in additive_terms(expr):
        coefficients = _quadratic_coefficients(term, var)
        if coefficients is None:
            shifted.append(term)
            continue
        c2, c1 = coefficients
        shift = -c1 / (2 * c2)
        if not (-c2).is_positive or c1 == 0 or shift.is_real is not True:
            shifted.append(term)
            continue
        shifted.append(term.subs(var, var + shift))
    return sp.Add(*shifted)


def inverse_distance_factors(term: sp.Expr,
                             variables: Set[Symbol]
                             ) -> List[Tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """
    Inverse-distance factors of a product.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    variables : set of Symbol
        Integration variables.

    Returns
    -------
    list of (factor, Q, nu)
        Every factor Q**(-nu/2) of `term` with nu > 0 rational and Q a
        polynomial of total degree 2 in `variables`, e.g.
        1/sqrt((x1 - x2)^2 + ...) gives Q = r12^2, nu = 1.
    """
    found = []
    for factor in sp.Mul.make_args(term):
        base, power = factor.as_base_exp()
        if not (power.is_Rational and power < 0
                and base.free_symbols & variables):
            continue
        if not base.is_polynomial(*variables):
            continue
        used = base.free_symbols & variables
        if sp.Poly(base, *used).total_degree() != 2:
            continue
        found.append((factor, base, -2 * power))
    return found


def _fourier_factor(base: sp.Expr,
                    nu: sp.Expr,
                    taken: Set[str]) -> Tuple[sp.Expr, Limits]:
    """
    Fourier representation of 1/|l| in spherical k coordinates.

    1/|l| = 1/(2 pi^2) int d^3k exp(i k.l)/k^2, with d^3k/k^2 =
    sin(theta_k) dphi_k dtheta_k dkappa.

    Parameters
    ----------
    base : sp.Expr
        l1^2 + l2^2 + l3^2.
    nu : sp.Expr
        Must be 1.
    taken : set of str
        Symbol names in use; updated with the new ones.

    Returns
    -------
    (integrand, limits)
        The k-space integrand and limits (phi_k, theta_k, kappa).
    """
    squares = sp.Add.make_args(base)
    if nu != 1 or len(squares) != 3 or not all(
            sq.is_Pow and sq.exp == 2 for sq in squares):
        raise ValueError("fourier needs 1/sqrt(l1^2 + l2^2 + l3^2)")
    ell = [sq.base for sq in squares]
    k = fresh("kappa", taken, positive=True)
    th = fresh("theta_k", taken, real=True)
    ph = fresh("phi_k", taken, real=True)
    n_hat = (sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th))
    phase = sp.I * k * sum(n * l for n, l in zip(n_hat, ell))
    return (sp.sin(th) * sp.exp(phase) / (2 * sp.pi ** 2),
            [(ph, 0, 2 * sp.pi), (th, 0, sp.pi), (k, 0, sp.oo)])


def transform_factor(base: sp.Expr,
                     nu: sp.Expr,
                     mode: str,
                     taken: Set[str]) -> Tuple[sp.Expr, Limits]:
    """
    Integral representation of one inverse-distance factor.

    Parameters
    ----------
    base : sp.Expr
        Q, quadratic in the integration variables.
    nu : sp.Expr
        Exponent: the factor is Q**(-nu/2).
    mode : {"gaussian", "laplace", "fourier"}
        "gaussian": 2/Gamma(nu/2) int_0^oo t^(nu-1) exp(-t^2 Q) dt (Boys);
        "laplace": 1/Gamma(nu/2) int_0^oo s^(nu/2-1) exp(-s Q) ds;
        "fourier": only 1/|l| (see `_fourier_factor`).
    taken : set of str
        Symbol names in use; updated with the new variables' names.

    Returns
    -------
    (integrand, limits)
        With ``base**(-nu/2) == Integral(integrand, *limits)``.
    """
    if mode == "gaussian":
        t = fresh("t", taken, positive=True)
        return (2 / sp.gamma(nu / 2) * t ** (nu - 1) * sp.exp(-t ** 2 * base),
                [(t, 0, sp.oo)])
    if mode == "laplace":
        s = fresh("s", taken, positive=True)
        return (s ** (nu / 2 - 1) / sp.gamma(nu / 2) * sp.exp(-s * base),
                [(s, 0, sp.oo)])
    if mode == "fourier":
        return _fourier_factor(base, nu, taken)
    raise ValueError(f"unknown transform {mode!r}; use one of {TRANSFORMS}")


def apply_transform(expr: sp.Expr,
                    limits: Sequence[Tuple],
                    mode: str,
                    taken: Set[str]) -> List[Tuple[sp.Expr, Limits]]:
    """
    Replace inverse-distance factors by integral representations.

    Works term by term; the same factor in different terms gets the same
    new variables. Terms with the same set of factors are collected into
    one piece.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    limits : sequence of (Symbol, lower, upper)
        The original integration limits.
    mode : {"gaussian", "laplace", "fourier"}
        Transform, see `transform_factor`.
    taken : set of str
        Symbol names in use; updated with the new variables' names.

    Returns
    -------
    list of (integrand, outer_limits)
        Pieces whose integrals (over `limits` first, then their
        `outer_limits`) sum to the original integral. Terms without
        inverse distances have no outer limits.
    """
    variables = {lim[0] for lim in limits}
    replacements: Dict[sp.Expr, Tuple[sp.Expr, Limits]] = {}
    pieces: Dict[Tuple, List[sp.Expr]] = {}
    for term in additive_terms(expr):
        found = inverse_distance_factors(term, variables)
        key = tuple(sorted(str(f) for f, _, _ in found))
        for factor, base, nu in found:
            if factor not in replacements:
                replacements[factor] = transform_factor(base, nu, mode, taken)
            term = term.subs(factor, replacements[factor][0])
        pieces.setdefault(key, []).append(term)
    result = []
    for key, terms in pieces.items():
        outer = [lim for factor, (_, lims) in replacements.items()
                 if str(factor) in key for lim in lims]
        result.append((sp.Add(*terms), outer))
    return result


def boys_parameter(term: sp.Expr,
                   var: Symbol,
                   mode: str) -> sp.Expr:
    """
    The parameter p of the Boys substitution.

    After the spatial integrals, a Gaussian-transformed term contains
    powers of (p + t^2) ("gaussian") or (p + s) ("laplace").

    Parameters
    ----------
    term : sp.Expr
        One additive term of the integrand in the transform variable.
    var : Symbol
        The transform variable t or s.
    mode : {"gaussian", "laplace"}
        Transform used.

    Returns
    -------
    sp.Expr or None
        The unique p over all factors c (p + var^2) resp. c (p + var), or
        None if there is none or more than one.
    """
    degree = 2 if mode == "gaussian" else 1
    found = set()
    for power in term.atoms(sp.Pow):
        base = power.base
        if not (base.is_Add and base.has(var) and base.is_polynomial(var)):
            continue
        poly = sp.Poly(base, var)
        if poly.degree() != degree:
            continue
        if any(c != 0 for c in poly.all_coeffs()[1:-1]):
            continue
        found.add(sp.cancel(poly.all_coeffs()[-1] / poly.LC()))
    return found.pop() if len(found) == 1 else None


def _is_compound_power(var: Symbol,
                       expr: sp.Expr) -> bool:
    """
    Whether `expr` is a power of a non-symbol base containing `var`
    (``expr.replace`` query, bind `var` with ``functools.partial``).

    Parameters
    ----------
    var : Symbol
        Variable the base must contain.
    expr : sp.Expr
        Any subexpression.

    Returns
    -------
    bool
        True for e.g. (1 - u**2)**(-3/2) with var = u.
    """
    return expr.is_Pow and expr.base.has(var) and not expr.base.is_Symbol


def _factor_base(expr: sp.Pow) -> sp.Expr:
    """
    The power `expr` with its base combined and factored.

    Parameters
    ----------
    expr : sp.Pow
        A power.

    Returns
    -------
    sp.Expr
        ``factor(together(base)) ** exp``.
    """
    return sp.factor(sp.together(expr.base)) ** expr.exp


def boys_substitute(term: sp.Expr,
                    var: Symbol,
                    mode: str,
                    u: Symbol) -> sp.Expr:
    """
    Map the outer transform integral onto [0, 1].

    Rewrites the integrand of int_0^oo d(var) over u in [0, 1] via
    u^2 = t^2/(p + t^2) ("gaussian") or u^2 = s/(p + s) ("laplace"),
    Jacobian included, with p from `boys_parameter`. For the two-electron
    integral of s Gaussians this gives the Boys-function integrand
    exp(-T u^2).

    Parameters
    ----------
    term : sp.Expr
        One additive term of the integrand in `var`.
    var : Symbol
        The transform variable t or s.
    mode : {"gaussian", "laplace"}
        Transform used.
    u : Symbol
        New integration variable on [0, 1].

    Returns
    -------
    sp.Expr or None
        The integrand in `u`, or None without a unique, provably positive
        p.
    """
    p = boys_parameter(term, var, mode)
    if p is None or p.is_positive is not True:
        return None
    if mode == "gaussian":
        new_var = u * sp.sqrt(p / (1 - u ** 2))
        jacobian = sp.sqrt(p) * (1 - u ** 2) ** sp.Rational(-3, 2)
    else:
        new_var = p * u ** 2 / (1 - u ** 2)
        jacobian = 2 * p * u / (1 - u ** 2) ** 2
    new = term.subs(var, new_var) * jacobian
    new = new.replace(partial(_is_compound_power, u), _factor_base)
    new = sp.powdenest(sp.powsimp(new, force=True, combine="all"),
                       force=True)
    return new.replace(is_exp, tidy_exponent)
