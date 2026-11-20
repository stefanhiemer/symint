"""
_util.py -- small helpers shared by the symint modules.

- `additive_terms`: split an integrand into terms, distributing products
  only, so that exponentials stay intact for the rules;
- `fresh`: new symbol names (transform variables) that do not clash;
- `is_whole_line`, `is_half_line`, `is_finite_range`, `is_full_period`:
  classify the range of one integration variable, used by the rules'
  ``applies`` tests and the engine;
- `is_rectangular`: whether a set of limits is a box, which separation
  and free choice of integration order need;
- `is_positive`: a sign test for Gaussian exponents that goes beyond
  SymPy's assumption system on rational functions;
- `is_exp`, `tidy_exponent`: predicate and rewrite for
  ``expr.replace``, putting exponents over a common denominator.

The range predicates take ``bounds = (lower, upper)``; anything else
(e.g. an indefinite limit) gives False. They are conservative: a bound
whose finiteness or value SymPy cannot decide makes them False.
"""

from typing import Any, Sequence, Tuple

import sympy as sp
from sympy import Symbol

Limit = Tuple[Symbol, Any, Any]


def additive_terms(expr: sp.Expr) -> Tuple[sp.Expr, ...]:
    """
    Additive terms of `expr` after distributing products.

    Only products of sums are multiplied out: powers of sums, sums in
    exponents and logarithms are left alone, so ``exp(x + y)`` is not
    split into ``exp(x)*exp(y)`` and ``(x + 1)**2`` is not expanded.

    Parameters
    ----------
    expr : sp.Expr
        Expression to split.

    Returns
    -------
    tuple of sp.Expr
        Terms whose sum is `expr`; ``(expr,)`` if it is not a sum.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> additive_terms((x + 1) * sp.exp(x + y))
    (x*exp(x + y), exp(x + y))
    >>> additive_terms((x + 1)**2 * (y + 2))
    (2*(x + 1)**2, y*(x + 1)**2)
    """
    return sp.Add.make_args(sp.expand(expr,
                                      power_exp=False,
                                      log=False,
                                      multinomial=False,
                                      mul=True))


def fresh(name: str,
          taken: set,
          **assumptions: Any) -> Symbol:
    """
    A new Symbol whose name is not yet in `taken`.

    Parameters
    ----------
    name : str
        Preferred name; ``name1``, ``name2``, ... are tried if it is taken.
    taken : set of str
        Names already in use. Updated in place with the returned name, so
        repeated calls with the same set give distinct symbols.
    **assumptions
        SymPy assumptions of the new symbol (e.g. ``positive=True``).

    Returns
    -------
    Symbol
        The new symbol.

    Examples
    --------
    >>> taken = {"t"}
    >>> fresh("t", taken, positive=True), fresh("t", taken)
    (t1, t2)
    >>> sorted(taken)
    ['t', 't1', 't2']
    """
    candidate, n = name, 0
    while candidate in taken:
        n += 1
        candidate = f"{name}{n}"
    taken.add(candidate)
    return Symbol(candidate, **assumptions)


def is_whole_line(bounds: Sequence[Any]) -> bool:
    """
    Whether the range is the whole real line.

    Parameters
    ----------
    bounds : sequence
        ``(lower, upper)``.

    Returns
    -------
    bool
        True if the range is (-oo, oo).
    """
    return len(bounds) == 2 and bounds[0] == -sp.oo and bounds[1] == sp.oo


def is_half_line(bounds: Sequence[Any]) -> bool:
    """
    Whether the range is a half-line.

    Parameters
    ----------
    bounds : sequence
        ``(lower, upper)``.

    Returns
    -------
    bool
        True for [lo, oo) or (-oo, hi] with the other end provably
        finite (a number, or a symbol that is real or declared finite).

    Examples
    --------
    >>> import sympy as sp
    >>> is_half_line([0, sp.oo]), is_half_line([sp.Symbol("c"), sp.oo])
    (True, False)
    """
    if len(bounds) != 2:
        return False
    lo, hi = (sp.sympify(bd) for bd in bounds)
    return (hi == sp.oo and lo.is_finite) or (lo == -sp.oo and hi.is_finite)


def is_finite_range(bounds: Sequence[Any]) -> bool:
    """
    Whether both ends of the range are finite.

    Parameters
    ----------
    bounds : sequence
        ``(lower, upper)``.

    Returns
    -------
    bool
        True if both bounds are provably finite. A plain ``Symbol("c")``
        is not (SymPy does not know it is finite); a real or positive
        one is.

    Examples
    --------
    >>> import sympy as sp
    >>> a = sp.Symbol("a", positive=True)
    >>> is_finite_range([0, a]), is_finite_range([0, sp.Symbol("c")])
    (True, False)
    """
    return len(bounds) == 2 and all(sp.sympify(b).is_finite for b in bounds)


def is_full_period(bounds: Sequence[Any]) -> bool:
    """
    Whether the range is one full period of an angle.

    Parameters
    ----------
    bounds : sequence
        ``(lower, upper)``.

    Returns
    -------
    bool
        True if upper - lower simplifies to 2 pi, wherever the period
        starts (e.g. [0, 2 pi] or [-pi, pi]).

    Examples
    --------
    >>> import sympy as sp
    >>> is_full_period([-sp.pi, sp.pi]), is_full_period([0, sp.pi])
    (True, False)
    """
    if len(bounds) != 2:
        return False
    return sp.simplify(sp.sympify(bounds[1]) - sp.sympify(bounds[0])
                       - 2 * sp.pi) == 0


def is_rectangular(limits: Sequence[Tuple]) -> bool:
    """
    Whether the integration domain is a box.

    Parameters
    ----------
    limits : sequence of (Symbol, lower, upper)
        Integration limits, as in ``sympy.Integral``.

    Returns
    -------
    bool
        True if no bound depends on any of the integration variables, so
        that the variables can be integrated in any order and the
        integral separates over products.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> is_rectangular([(x, 0, 1), (y, 0, 1)])
    True
    >>> is_rectangular([(x, 0, y), (y, 0, 1)])  # triangle
    False
    """
    variables = {lim[0] for lim in limits}
    return all(not (sp.sympify(b).free_symbols & variables)
               for lim in limits for b in lim[1:])


def is_positive(expr: sp.Expr) -> bool:
    """
    Whether `expr` is provably positive.

    `expr` is brought to one cancelled fraction num/den first and both are
    expanded; it counts as positive if SymPy's assumptions show that num
    and den have the same sign. This decides cases SymPy's ``is_positive``
    leaves open, such as differences that cancel to sums of positive
    terms. The test is sufficient, not necessary: False means "not
    proven", not "negative".

    Parameters
    ----------
    expr : sp.Expr
        Expression to test, typically the coefficient A of a Gaussian
        exp(-A x^2 + ...).

    Returns
    -------
    bool
        True if positivity could be proven.

    Examples
    --------
    >>> import sympy as sp
    >>> a, b = sp.symbols("a b", positive=True)
    >>> e = 1 / a - 1 / (a + b)
    >>> e.is_positive is None, is_positive(e)
    (True, True)
    >>> is_positive(a - b)
    False
    """
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    num, den = sp.expand(num), sp.expand(den)
    return bool((num.is_positive and den.is_positive)
                or (num.is_negative and den.is_negative))


def is_exp(expr: sp.Expr) -> bool:
    """
    Whether `expr` is an exponential, for use as an ``expr.replace``
    query.

    Parameters
    ----------
    expr : sp.Expr
        Any subexpression.

    Returns
    -------
    bool
        True if `expr` is ``exp(...)``.
    """
    return isinstance(expr, sp.exp)


def tidy_exponent(expr: sp.exp) -> sp.Expr:
    """
    The exponential `expr` with its exponent cancelled and factored.

    Parameters
    ----------
    expr : sp.exp
        An exponential.

    Returns
    -------
    sp.Expr
        ``exp(factor(cancel(exponent)))``, e.g. one fraction instead of a
        sum of terms that cancel only jointly.

    Examples
    --------
    >>> import sympy as sp
    >>> a, b, x = sp.symbols("a b x")
    >>> tidy_exponent(sp.exp(a * x / b - x / b))
    exp(x*(a - 1)/b)
    """
    return sp.exp(sp.factor(sp.cancel(expr.args[0])))
