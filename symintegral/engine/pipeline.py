"""
`integrate`: transform, separate, integrate inner then outer variables.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Integral, Symbol

from .. import sage as sage_mod
from .._util import additive_terms, fresh, is_rectangular
from ..rewrite import TRANSFORMS, apply_transform, boys_substitute
from ..rules import Rule
from ..separate import separate as separate_terms, tidy_exponentials
from .context import Context, Limits, LogEntry
from .steps import evaluate_iterated

#: Back ends tried by `integrate` after the rules, in order.
DEFAULT_BACKENDS: Tuple[str, ...] = ("maxima", "fricas", "sympy")


def _evaluate_outer(value: sp.Expr,
                    outer: Limits,
                    transform: Optional[str],
                    substitute_outer: bool,
                    ctx: Context) -> sp.Expr:
    """
    Integrate the outer transform variables.

    For a single "gaussian"/"laplace" variable (and `substitute_outer`),
    each term with a unique parameter p (see `boys_parameter`) is first
    mapped from [0, oo) onto u in [0, 1] (`boys_substitute`); the other
    terms keep the original variable.

    Parameters
    ----------
    value : sp.Expr
        Result of the inner integrals, integrand in the outer variables.
    outer : list of (Symbol, lower, upper)
        Limits of the transform variables.
    transform : str or None
        Transform used.
    substitute_outer : bool
        Whether to map the outer variable onto [0, 1].
    ctx : Context
        Options.

    Returns
    -------
    sp.Expr
        The integral, possibly partial.
    """
    # 
    if not (substitute_outer and transform in ("gaussian", "laplace")
            and len(outer) == 1):
        return evaluate_iterated(value, outer, ctx)
    # special case of Boys integrals
    var = outer[0][0]
    taken = ({str(s) for s in value.atoms(Symbol)}
             | {str(s) for s in ctx.session.symbols})
    u = fresh("u", taken, positive=True)
    pieces: List[Tuple[sp.Expr, Limits]] = []
    plain: List[sp.Expr] = []
    for term in additive_terms(tidy_exponentials(value)):
        new = boys_substitute(term, var, transform, u)
        if new is None:
            plain.append(term)
        else:
            pieces.append((new, [(u, 0, 1)]))
    if plain:
        pieces.append((sp.Add(*plain), list(outer)))
    ctx.session.extend([u], [(u, 0, 1)])
    return sp.Add(*[evaluate_iterated(f, lims, ctx) for f, lims in pieces])


def _evaluate_inner(function: sp.Expr,
                    limits: Limits,
                    separate: bool,
                    cache: Dict[Integral, sp.Expr],
                    ctx: Context) -> sp.Expr:
    """
    Integrate over the original variables, separating if possible.

    On a rectangular domain (constant limits), `separate.separate` splits
    each additive term into factors (exponentials split as well) and groups
    them by shared integration variables. Each group becomes its own
    lower-dimensional integral, e.g.

        int f(x) g(y) h(y, z) dx dy dz  ->  int f dx * int g h dy dz.

    Each distinct integral is evaluated once (`cache`) with
    `evaluate_iterated`. Terms with a denominator in the variables that
    do not split into 1-D integrals are kept together, so cancellations
    between them survive. Otherwise (or with `separate=False`) the whole
    integral is evaluated iteratively.

    Parameters
    ----------
    function : sp.Expr
        Integrand (of one transform piece).
    limits : list of (Symbol, lower, upper)
        The original limits.
    separate : bool
        Split into products of lower-dimensional integrals if the limits
        are rectangular.
    cache : dict
        Integral -> value of the separated integrals evaluated so far,
        shared across pieces; updated in place.
    ctx : Context
        Options.

    Returns
    -------
    sp.Expr
        The integral, possibly partial.
    """
    # if not separable or rectangular
    if not (separate and is_rectangular(limits)):
        return evaluate_iterated(function, limits, ctx)
    #
    total = sp.Integer(0)
    for prefactor, integrals in separate_terms(function, limits):
        value = prefactor
        for part in integrals:
            if part not in cache:
                cache[part] = evaluate_iterated(part.function, part.limits,
                                                ctx)
            value *= cache[part]
        total += value
    return total


def integrate(integral: Integral,
              backends: Sequence[str] = DEFAULT_BACKENDS,
              simplify: Optional[str] = "full",
              complete_square: bool = True,
              limit_algorithm: str = "maxima",
              separate: bool = True,
              transform: Optional[str] = None,
              substitute_outer: bool = True,
              step_timeout: Optional[float] = None,
              rules: Optional[Sequence[Rule]] = None,
              log: Optional[List[LogEntry]] = None) -> sp.Expr:
    """
    Evaluate an integral as far as possible in closed form:

    - optionally transform coupling factors Q^(-nu/2) into outer variables (`transform`) 
    - split the integrand into products of lower-dimensional integrals (`separate`)
    - try integrate each variable with the closed-form `rules`
    - if not solved, then the `backends` in order
    - finally integrate the outer variables. 
    
    Anything that fails stays an Integral, so the result may be partial (see the `symintegral.engine` 
    docstring for worked examples of each step).

    Parameters
    ----------
    integral : sympy.Integral
        The unevaluated integral, e.g. built with `chart_integral`
        or `simplex_integral`.
    backends : sequence of str, optional
        Tried in order for each variable no rule handles: "sympy" (native
        SymPy, keeps assumptions), "maxima", "fricas", "giac" (via Sage)
        and "fricas_antiderivative" (FriCAS antiderivative plus one-sided
        limits with `limit_algorithm`; only valid if the antiderivative is
        continuous on the interval -- check such results).
    simplify : {"full", "radical", None}, optional
        Sage pre-simplification of the integrand before each CAS step
        ("radical" adds canonicalize_radical: needed for e.g.
        sqrt((xi + eta)^2) -> xi + eta, but not branch-safe). Skipped if
        only "sympy" is used.
    complete_square : bool, optional
        Complete the square in Gaussian exponents (term-wise on (-oo, oo),
        with one common shift moving the bound on half-lines).
    limit_algorithm : str, optional
        Sage limit algorithm for "fricas_antiderivative".
    separate : bool, optional
        Split integrals over rectangular domains into products of
        lower-dimensional ones (cached).
    transform : {None, "gaussian", "laplace", "fourier"}, optional
        Replace factors Q^(-nu/2), Q quadratic in the integration
        variables (e.g. ``1/|r1 - r2|``), by an integral over a new variable
        that decouples them; the new variables are integrated last.
        "gaussian": 2/Gamma(nu/2) int_0^oo t^(nu-1) exp(-t^2 Q) dt;
        "laplace": s = t^2; "fourier": ``1/|l|`` = 1/(2 pi^2) int d^3k
        exp(i k.l)/k^2. None (default): no transform.
    substitute_outer : bool, optional
        For "gaussian"/"laplace", map the final t-/s-integral onto [0, 1]
        by u^2 = t^2/(p + t^2) (in quantum chemistry: the Boys-function
        form).
    step_timeout : float, optional
        Seconds per step, rule pass and back end; None = no limit.
    rules : sequence of Rule, optional
        Rules to try (default: the registry `rules.RULES`; () disables).
    log : list, optional
        Receives one dict per step: "var", "group", "method" (rule
        name or back end that succeeded, or None) and "errors".

    Returns
    -------
    sp.Expr
        Closed form, or an expression containing SymPy Integrals over the
        variables that could not be integrated.

    Raises
    ------
    TypeError
        If `integral` is not a ``sympy.Integral``.
    ValueError
        For an unknown `transform`.
    """
    if not isinstance(integral, Integral):
        raise TypeError("integrate expects a sympy.Integral")
    if transform is not None and transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; "
                         f"use one of {TRANSFORMS}")
    # limits as tuples; names already in use (for fresh variables)
    limits = [tuple(lim) for lim in integral.limits]
    taken = {str(s) for s in integral.atoms(Symbol)}
    # apply optional transforms -> pieces (integrand, outer limits)
    pieces = ([(integral.function, [])] if transform is None else
              apply_transform(integral.function, limits, transform, taken))
    all_limits = limits + [lim for _, outer in pieces for lim in outer]
    # collect all symbols, incl. new outer variables to pass to Sage incl. assumptions
    symbols = sorted(set(integral.atoms(Symbol))
                     | {lim[0] for lim in all_limits}, key=str)
    # bundle options and state for the steps
    ctx = Context(session=sage_mod.SageSession(symbols, all_limits),
                  backends=tuple(backends),
                  simplify=simplify,
                  complete_square=complete_square,
                  limit_algorithm=limit_algorithm,
                  step_timeout=step_timeout,
                  rules=rules,
                  log=log)
    # separated factors already evaluated, shared by all pieces
    cache: Dict[Integral, sp.Expr] = {}
    # inner variables, then outer ones (unless inner integrals remain)
    try:
        total = sp.Integer(0)
        for function, outer in pieces:
            value = _evaluate_inner(function, limits, separate, cache, ctx)
            if outer and value.has(Integral):
                value = Integral(value, *outer)
            elif outer:
                value = _evaluate_outer(value, outer, transform,
                                        substitute_outer, ctx)
            total += value
        return tidy_exponentials(total)
    # always forget Sage assumptions, even on errors
    finally:
        ctx.session.close()
    return
