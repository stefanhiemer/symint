"""
separate.py -- split integrals over rectangular domains into sums of
products of lower-dimensional integrals, and tidy exponentials in
results.

On a rectangular domain, a term f(x) g(y) h(y, z) integrates to
int f dx * int g h dy dz: factors are grouped by the integration
variables they share, and each group becomes its own, smaller integral.
The engine evaluates every distinct such integral only once.
"""

from typing import List, Sequence, Set, Tuple

import sympy as sp
from sympy import Integral, Symbol

from ._util import additive_terms, is_exp, tidy_exponent

Group = Tuple[Set[Symbol], List[sp.Expr]]


def _split_exponential(expr: sp.exp) -> sp.Expr:
    """
    Split one exponential of a sum into a product of exponentials.

    Parameters
    ----------
    expr : sp.exp
        ``exp(u1 + u2 + ...)``; the exponent is expanded first.

    Returns
    -------
    sp.Expr
        ``exp(u1) * exp(u2) * ...``.
    """
    terms = sp.Add.make_args(sp.expand(expr.args[0]))
    return sp.Mul(*[sp.exp(t) for t in terms])


def split_exponentials(expr: sp.Expr) -> sp.Expr:
    """
    Split every exponential of a sum in `expr` into a product.

    Parameters
    ----------
    expr : sp.Expr
        Expression containing exponentials.

    Returns
    -------
    sp.Expr
        `expr` with each exp(u1 + u2 + ...) replaced by
        exp(u1) * exp(u2) * ..., so that factors depending on different
        variables can be told apart.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> sp.Mul.make_args(split_exponentials(sp.exp(-x**2 - y)))
    (exp(-y), exp(-x**2))
    """
    return expr.replace(is_exp, _split_exponential)


def _group_factors(factors: Sequence[sp.Expr],
                   variables: Set[Symbol]) -> Tuple[sp.Expr, List[Group]]:
    """
    Group the factors of one product by shared integration variables.

    Two factors end up in the same group if they are connected by a
    chain of factors sharing variables (connected components).

    Parameters
    ----------
    factors : sequence of sp.Expr
        Factors of one additive term.
    variables : set of Symbol
        Integration variables.

    Returns
    -------
    prefactor : sp.Expr
        Product of the factors free of integration variables.
    groups : list of (set of Symbol, list of sp.Expr)
        The variables of each group and its factors.
    """
    prefactor = sp.Integer(1)
    groups: List[Group] = []
    for factor in factors:
        used = factor.free_symbols & variables
        if not used:
            prefactor *= factor
            continue
        merged_vars, merged_factors = set(used), [factor]
        for group in [g for g in groups if g[0] & used]:
            groups.remove(group)
            merged_vars |= group[0]
            merged_factors += group[1]
        groups.append((merged_vars, merged_factors))
    return prefactor, groups


def _has_denominator(factors: Sequence[sp.Expr],
                     variables: Set[Symbol]) -> bool:
    """
    Whether a product has a genuine denominator in the variables.

    Parameters
    ----------
    factors : sequence of sp.Expr
        Factors of one additive term.
    variables : set of Symbol
        Integration variables.

    Returns
    -------
    bool
        True if a factor is a negative power of a variable-dependent base.
        Exponentials do not count (``sp.denom`` would count exp(-x) as
        1/exp(x)).
    """
    return any(isinstance(f, sp.Pow) and f.exp.is_negative
               and f.base.free_symbols & variables for f in factors)


def separate(expr: sp.Expr,
             limits: Sequence[Tuple]) -> List[Tuple[sp.Expr, List[Integral]]]:
    """
    Split an integral over a rectangular domain into a sum of products.

    Each additive term has its exponentials split and its factors grouped
    into connected components by shared integration variables
    (`_group_factors`); each component becomes an integral over just its
    variables, and variables a term does not depend on contribute the
    length of their range. Terms that are not fully split into 1-D
    integrals *and* have a denominator in the integration variables are
    summed back into one integral over all variables, so that
    cancellations between them survive (e.g. a prolate Laplacian over
    xi^2 - eta^2); all other terms are separated.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    limits : sequence of (Symbol, lower, upper)
        Rectangular integration limits (no bound depends on a variable).

    Returns
    -------
    list of (sp.Expr, list of sympy.Integral)
        ``(prefactor, integrals)`` pairs; the original integral is
        ``sum(prefactor * prod(integrals))``.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y, a = sp.symbols("x y a")
    >>> separate(a * x * sp.exp(-y), [(x, 0, 1), (y, 0, sp.oo)])
    [(a, [Integral(x, (x, 0, 1)), Integral(exp(-y), (y, 0, oo))])]
    """
    variables = [lim[0] for lim in limits]
    varset = set(variables)
    limit_of = {lim[0]: lim for lim in limits}
    separated = []
    coupled: List[sp.Expr] = []
    for term in additive_terms(expr):
        term = sp.cancel(term)
        factors = sp.Mul.make_args(split_exponentials(term))
        prefactor, groups = _group_factors(factors, varset)
        # variables the term does not depend on still contribute their
        # (possibly infinite) range length
        covered = set().union(*[g[0] for g in groups]) if groups else set()
        groups += [({v}, [sp.Integer(1)]) for v in variables
                   if v not in covered]
        if (any(len(vs) > 1 for vs, _ in groups)
                and _has_denominator(factors, varset)):
            coupled.append(term)
            continue
        # merge split exponentials back per group (Maxima prefers it)
        integrals = [Integral(sp.powsimp(sp.Mul(*fs), combine="exp"),
                              *[limit_of[v] for v in variables if v in vs])
                     for vs, fs in groups]
        separated.append((prefactor, integrals))
    if coupled:
        separated.append((sp.Integer(1),
                          [Integral(sp.Add(*coupled), *limits)]))
    return separated


def _tidy(expr: sp.Expr) -> sp.Expr:
    """
    Merge products of exponentials and tidy their exponents.

    Parameters
    ----------
    expr : sp.Expr
        Expression (not descending into Integrals).

    Returns
    -------
    sp.Expr
        `expr` with exp(a) exp(b) -> exp(a + b) and every exponent over a
        common denominator (`tidy_exponent`).
    """
    return sp.powsimp(expr, combine="exp").replace(is_exp, tidy_exponent)


def _is_integral(expr: sp.Expr) -> bool:
    """
    Whether `expr` is an unevaluated Integral (``expr.replace`` query).

    Parameters
    ----------
    expr : sp.Expr
        Any subexpression.

    Returns
    -------
    bool
        True for a ``sympy.Integral``.
    """
    return isinstance(expr, Integral)


def _tidy_integral(integral: Integral) -> Integral:
    """
    The Integral with its integrand tidied by `_tidy`.

    Parameters
    ----------
    integral : sympy.Integral
        Unevaluated integral.

    Returns
    -------
    sympy.Integral
        Same limits, tidied integrand.
    """
    return Integral(_tidy(integral.function), *integral.limits)


def tidy_exponentials(expr: sp.Expr) -> sp.Expr:
    """
    Merge exponentials and put exponents over a common denominator.

    Applied to results and to the integrands of unevaluated Integrals in
    them: split exponentials that cancel only jointly would otherwise
    overflow to inf * 0 when evaluated numerically.

    Parameters
    ----------
    expr : sp.Expr
        Result of an integration, possibly containing Integrals.

    Returns
    -------
    sp.Expr
        The tidied expression.

    Examples
    --------
    >>> import sympy as sp
    >>> x, a = sp.symbols("x a")
    >>> tidy_exponentials(sp.exp(x**2 / a) * sp.exp(-x**2 / a - 1))
    exp(-1)
    """
    expr = expr.replace(_is_integral, _tidy_integral)
    return _tidy(expr)
