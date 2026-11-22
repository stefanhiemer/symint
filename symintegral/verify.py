"""
verify.py -- checking integration results.

`check_antiderivative` decides whether dF/dx = f symbolically, cheaply
where possible: instead of simplifying the whole residual

    r = dF/dx - f,

it expands r into additive terms, sorts them by their transcendental
"kernel" (the product of factors that are not rational in x, e.g.
exp(-a x^2), log(x + 1), erf(x)) and adds up the rational coefficients
of each kernel separately. Terms that cancel exactly do so within their
group after a cheap ``cancel``, so a full ``simplify`` is only needed on
what is left -- typically little or nothing. Anything that still
survives is checked numerically at random points.

`check_definite` compares a definite result with numerical quadrature at
random parameter values.

Both return a `Check` with the method that decided it: "groups"
(term sorting alone), "simplify" (simplification of the remainder),
"numeric" (only numerically zero -- likely, not proven) or None
(the check failed).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Integral, Symbol


@dataclass
class Check:
    """
    Outcome of a check.

    Attributes
    ----------
    ok : bool
        Whether the result was confirmed (symbolically or numerically).
    method : str or None
        "groups", "simplify", "numeric", or None if not confirmed.
    residual : sp.Expr
        What was left after the symbolic stages (0 if proven).
    """

    ok: bool
    method: Optional[str]
    residual: sp.Expr

    def __bool__(self) -> bool:
        return self.ok


def _split_rational(term: sp.Expr,
                    var: Symbol) -> Tuple[sp.Expr, sp.Expr]:
    """
    Split a product into a rational coefficient and a kernel.

    Parameters
    ----------
    term : sp.Expr
        One additive term.
    var : Symbol
        Variable.

    Returns
    -------
    (coefficient, kernel)
        term = coefficient * kernel, with coefficient a rational function
        of `var` (other symbols allowed) and kernel the product of all
        other factors (1 if there are none).
    """
    coeff, kernel = [], []
    for f in sp.Mul.make_args(term):
        base, power = f.as_base_exp()
        if f.is_rational_function(var):
            coeff.append(f)
        elif (power.is_Rational and not power.is_Integer
              and base.is_rational_function(var)):
            # radicals: (x^2 + 1)^(3/2) -> (x^2 + 1) * (x^2 + 1)^(1/2), so
            # every power of the same base shares one kernel
            whole = sp.floor(power)
            coeff.append(base**whole)
            kernel.append(base**(power - whole))
        else:
            kernel.append(f)
    return sp.Mul(*coeff), sp.Mul(*kernel)


def _expand_exponent(expr: sp.Expr) -> sp.Expr:
    """
    exp(arg) -> exp(expand(arg)) (``replace`` helper).

    Parameters
    ----------
    expr : sp.exp
        Exponential.

    Returns
    -------
    sp.Expr
        The exponential with its argument expanded, so that a following
        ``power_exp`` expansion splits it into one factor per term.
    """
    return sp.exp(sp.expand(expr.args[0]))


def _split_exponentials(expr: sp.Expr) -> sp.Expr:
    """
    Expand all exponents and split exponentials into one factor per term.

    Parameters
    ----------
    expr : sp.Expr
        Expression.

    Returns
    -------
    sp.Expr
        E.g. exp((x - b)^2) -> exp(x^2) * exp(-2 b x) * exp(b^2).
    """
    expr = expr.replace(lambda e: isinstance(e, sp.exp), _expand_exponent)
    return sp.expand(expr, power_exp=True, mul=False, multinomial=False,
                     log=False, power_base=False)


def group_terms(expr: sp.Expr,
                var: Symbol) -> Dict[sp.Expr, sp.Expr]:
    """
    Collect `expr` by transcendental kernel.

    Exponents are expanded and exponentials split (exp((x - b)^2) ->
    exp(x^2) exp(-2 b x) exp(b^2)), radicals reduced to a fractional power
    of their base, and products of sums distributed, so equal kernels
    written differently meet.

    Parameters
    ----------
    expr : sp.Expr
        Expression, typically a residual.
    var : Symbol
        Variable.

    Returns
    -------
    dict
        kernel -> sum of its rational coefficients after ``cancel``; only
        nonzero groups are kept.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x")
    >>> group_terms(x * sp.exp(x + 1) - sp.E * x * sp.exp(x) + 1 / x, x)
    {1: 1/x}
    """
    expanded = sp.expand(_split_exponentials(expr), log=False,
                         multinomial=True, mul=True)
    groups: Dict[sp.Expr, List[sp.Expr]] = {}
    for term in sp.Add.make_args(expanded):
        coeff, kernel = _split_rational(term, var)
        groups.setdefault(kernel, []).append(coeff)
    collected = {k: sp.cancel(sp.Add(*cs)) for k, cs in groups.items()}
    return {k: c for k, c in collected.items() if c != 0}


def _sample(symbol: Symbol,
            rng: random.Random) -> sp.Expr:
    """
    A random value consistent with the symbol's assumptions.

    Parameters
    ----------
    symbol : Symbol
        Symbol to replace.
    rng : random.Random
        Random source.

    Returns
    -------
    sp.Expr
        A rational number: integers for integer symbols, signs respected.
    """
    if symbol.is_integer:
        value = sp.Integer(rng.randint(1, 4))
    else:
        value = sp.Rational(rng.randint(30, 170), 100)
    if symbol.is_negative or symbol.is_nonpositive:
        value = -value
    return value


def numerically_zero(expr: sp.Expr,
                     samples: int = 4,
                     tol: float = 1e-9,
                     seed: int = 0,
                     points: Optional[Dict[Symbol, sp.Expr]] = None) -> bool:
    """
    Whether `expr` vanishes at random points.

    Parameters
    ----------
    expr : sp.Expr
        Expression to test.
    samples : int, optional
        Number of random points.
    tol : float, optional
        Relative tolerance, relative to the largest term's magnitude.
    seed : int, optional
        Seed of the random points (reproducible checks).
    points : dict, optional
        Fixed values for some symbols (e.g. to stay inside a domain).

    Returns
    -------
    bool
        True if abs(expr) <= tol * max(1, abs(largest term)) at every point;
        False otherwise, also if it does not evaluate to a number.
    """
    rng = random.Random(seed)
    symbols = sorted(expr.free_symbols, key=str)
    terms = sp.Add.make_args(expr)
    for _ in range(samples):
        values = {s: _sample(s, rng) for s in symbols}
        values.update(points or {})
        try:
            value = complex(sp.N(expr.subs(values), 30))
            scale = max([1.0] + [abs(complex(sp.N(t.subs(values), 30)))
                                 for t in terms])
        except (TypeError, ValueError):
            return False
        if not abs(value) <= tol * scale:
            return False
    return True


def check_antiderivative(F: sp.Expr,
                         f: sp.Expr,
                         var: Symbol,
                         simplify: bool = True,
                         samples: int = 4) -> Check:
    """
    Check that F is an antiderivative of f, i.e. dF/dvar = f.

    Parameters
    ----------
    F : sp.Expr
        Claimed antiderivative.
    f : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    simplify : bool, optional
        Try ``simplify`` on the grouped remainder before going numeric.
    samples : int, optional
        Random points for the numerical fallback (0: no fallback).

    Returns
    -------
    Check
        See the module docstring.

    Examples
    --------
    >>> import sympy as sp
    >>> x, a = sp.symbols("x a", positive=True)
    >>> F = -(a * x + 1) * sp.exp(-a * x) / a**2
    >>> check_antiderivative(F, x * sp.exp(-a * x), x).method
    'groups'
    >>> check_antiderivative(sp.atan(x), 1 / (1 + x**2), x).method
    'groups'
    >>> check_antiderivative(sp.asinh(x), 1 / sp.sqrt(1 + x**2), x).ok
    True
    >>> check_antiderivative(x**2, x, x).ok
    False
    """
    residual = sp.diff(F, var) - f
    groups = group_terms(residual, var)
    if not groups:
        return Check(True, "groups", sp.Integer(0))
    remainder = sp.Add(*[c * k for k, c in groups.items()])
    # transcendental denominators (e.g. 1/(1 + exp(x))) become kernels of
    # their own; over a common denominator only the numerator must vanish
    numerator = sp.together(_split_exponentials(remainder)).as_numer_denom()[0]
    groups = group_terms(numerator, var)
    if not groups:
        return Check(True, "groups", sp.Integer(0))
    if simplify:
        simplified = sp.simplify(remainder)
        if simplified == 0:
            return Check(True, "simplify", sp.Integer(0))
        remainder = simplified
    if samples and numerically_zero(remainder, samples):
        return Check(True, "numeric", remainder)
    return Check(False, None, remainder)


def check_definite(result: sp.Expr,
                   integral: Integral,
                   samples: int = 2,
                   tol: float = 1e-7,
                   seed: int = 0) -> Check:
    """
    Compare a definite result with numerical quadrature.

    Parameters
    ----------
    result : sp.Expr
        Claimed value (may itself contain Integrals; they are evaluated
        numerically too).
    integral : sympy.Integral
        The original integral.
    samples : int, optional
        Random parameter points.
    tol : float, optional
        Relative tolerance.
    seed : int, optional
        Seed of the random parameter values.

    Returns
    -------
    Check
        ``method`` "numeric" if all points agree, None otherwise.

    Examples
    --------
    >>> import sympy as sp
    >>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
    >>> I = sp.Integral(sp.exp(-a * x**2), (x, -sp.oo, sp.oo))
    >>> check_definite(sp.sqrt(sp.pi / a), I).ok
    True
    """
    rng = random.Random(seed)
    params = sorted(integral.free_symbols | result.free_symbols, key=str)
    for _ in range(samples):
        values = {s: _sample(s, rng) for s in params}
        try:
            got = complex(sp.N(result.subs(values), 20))
            want = complex(sp.N(integral.subs(values), 20))
        except (TypeError, ValueError):
            return Check(False, None, result)
        if not abs(got - want) <= tol * max(1.0, abs(want)):
            return Check(False, None, result - integral)
    return Check(True, "numeric", sp.Integer(0))


__all__: Sequence[str] = ["Check", "check_antiderivative", "check_definite",
                          "group_terms", "numerically_zero"]
