"""
Single integration steps: closed-form rules, back ends, and the iterated
integration of one integrand over its limits.
"""

from functools import partial
from typing import List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Integral, Symbol

from .. import sage as sage_mod
from .._util import is_half_line, is_rectangular, is_whole_line
from ..rewrite import common_shift, complete_square
from ..rules import apply_rules
from .context import Context, Limits, LogEntry
from .timeout import StepTimeout


def _prepare(current: sp.Expr,
             var: Symbol,
             bounds: List,
             ctx: Context,
             sage_simplify: bool = True) -> Tuple[sp.Expr, List]:
    """
    Integrand and bounds handed to the back ends for one step.

    Simplifies with Sage (if `sage_simplify` and `ctx.simplify` are set),
    then completes the square: term-wise on the whole line, by
    one common shift (moving the bounds) on half-lines. Completing comes
    *after* simplifying, since simplify_full would re-expand the shifted
    exponent.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : list
        ``[lower, upper]``.
    ctx : Context
        Options.
    sage_simplify : bool, optional
        Whether Sage simplification is wanted (only for Sage back ends:
        its output may defeat SymPy, e.g. on the Beta integral).

    Returns
    -------
    (sp.Expr, list)
        The prepared integrand and the (possibly shifted) bounds.
    """
    expr = current
    if sage_simplify and ctx.simplify:
        ctx.session.ensure()
        expr = sage_mod.from_sage(
            sage_mod.simplify(sage_mod.to_sage(expr), ctx.simplify),
            ctx.session.symbols)
    if ctx.complete_square and is_whole_line(bounds):
        expr = complete_square(expr, var)
    elif ctx.complete_square and is_half_line(bounds):
        shift = common_shift(expr, var)
        if shift is not None:
            return (expr.subs(var, var + shift),
                    [sp.sympify(bd) - shift for bd in bounds])
    return expr, list(bounds)


def _cas_step(integrand: sp.Expr,
              var: Symbol,
              bounds: List,
              backend: str,
              ctx: Context) -> sp.Expr:
    """
    One integration with one back end.

    Parameters
    ----------
    integrand : sp.Expr
        Prepared integrand.
    var : Symbol
        Integration variable.
    bounds : list
        ``[lower, upper]``.
    backend : str
        "sympy" (native) or a Sage back end.
    ctx : Context
        Options and Sage session.

    Returns
    -------
    sp.Expr
        The integral.

    Raises
    ------
    ValueError
        "unevaluated" if the back end returns an unevaluated integral;
        any other exception of the back end is passed on.
    """
    if backend == "sympy":
        result = sp.integrate(integrand, (var, *bounds))
        if result.has(Integral):
            raise ValueError("unevaluated")
        return result
    ctx.session.ensure()
    args = [sage_mod.to_sage(var)] + [sage_mod.to_sage(b) for b in bounds]
    result = sage_mod.integrate_step(sage_mod.to_sage(integrand), args,
                                     backend, ctx.limit_algorithm)
    if sage_mod.has_unevaluated_integral(result):
        raise ValueError("unevaluated")
    return sage_mod.from_sage(result, ctx.session.symbols)


def _timeout_message(ctx: Context) -> str:
    """
    Log message of a timed-out step.

    Parameters
    ----------
    ctx : Context
        Options (for the timeout).

    Returns
    -------
    str
        "timeout after <step_timeout>s".
    """
    return f"timeout after {ctx.step_timeout}s"


def _rule_step(current: sp.Expr,
               limits: Limits,
               ctx: Context,
               entry: LogEntry) -> Optional[Tuple[int, str, sp.Expr]]:
    """
    Integrate one variable with the closed-form rules.

    The variables are tried in order, all of them if the limits are
    rectangular (the order is free), else only the innermost.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    limits : list of (Symbol, lower, upper)
        Remaining limits, innermost first.
    ctx : Context
        Options.
    entry : dict
        Log entry; rule timeouts are recorded in its "errors".

    Returns
    -------
    (int, str, sp.Expr) or None
        Index of the integrated variable in `limits`, rule name and
        result; None if no rule applied.
    """
    order = range(len(limits)) if is_rectangular(limits) else [0]
    for i in order:
        var, *bounds = limits[i]
        try:
            ruled = ctx.timed(partial(apply_rules, current, var, bounds,
                                      ctx.rules))
        except StepTimeout:
            entry["errors"][f"rules {var}"] = _timeout_message(ctx)
            continue
        if ruled is not None:
            return (i, *ruled)
    return None


def _prepare_logged(current: sp.Expr,
                    var: Symbol,
                    bounds: List,
                    ctx: Context,
                    entry: LogEntry,
                    sage_simplify: bool) -> Tuple[sp.Expr, List]:
    """
    `_prepare` under the step timeout; failures are logged and fall back
    to the unprepared integrand.

    Parameters
    ----------
    current, var, bounds, ctx, sage_simplify
        See `_prepare`.
    entry : dict
        Log entry; a failure is recorded under "prepare".

    Returns
    -------
    (sp.Expr, list)
        The prepared (or original) integrand and bounds.
    """
    try:
        return ctx.timed(partial(_prepare, current, var, bounds, ctx,
                                 sage_simplify))
    except StepTimeout:
        entry["errors"]["prepare"] = _timeout_message(ctx)
    except Exception as err:  # e.g. no Sage counterpart of a function
        entry["errors"]["prepare"] = f"{type(err).__name__}: {err}"
    return current, list(bounds)


def _backend_step(current: sp.Expr,
                    var: Symbol,
                    bounds: List,
                    ctx: Context,
                    entry: LogEntry) -> Optional[Tuple[str, sp.Expr]]:
    """
    Integrate one variable with the back ends, in order.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : list
        ``[lower, upper]``.
    ctx : Context
        Options.
    entry : dict
        Log entry; failures of the preparation and of every back end are
        recorded in its "errors".

    Returns
    -------
    (str, sp.Expr) or None
        Name of the successful back end and result; None if all failed.
    """
    # prepared lazily, once with Sage simplification (Sage back ends)
    # and once without (SymPy)
    prepared = {}
    for backend in ctx.backends:
        sage_simplify = backend != "sympy"
        if sage_simplify not in prepared:
            prepared[sage_simplify] = _prepare_logged(
                current, var, bounds, ctx, entry, sage_simplify)
        integrand, step_bounds = prepared[sage_simplify]
        try:
            result = ctx.timed(partial(_cas_step, integrand, var,
                                       step_bounds, backend, ctx))
        except StepTimeout:
            entry["errors"][backend] = _timeout_message(ctx)
            continue
        except Exception as err:  # Maxima questions, conversions, ...
            msg = str(err)
            entry["errors"][backend] = msg if msg == "unevaluated" \
                else f"{type(err).__name__}: {err}"
            continue
        return backend, result
    return None


def evaluate_iterated(current: sp.Expr,
                      limits: Sequence[Tuple],
                      ctx: Context) -> sp.Expr:
    """
    Integrate one variable at a time.

    Each step tries the rules (`_rule_step`), then the back ends on the
    innermost variable (`_backend_step`); the first variable nothing
    can integrate ends the iteration.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    limits : sequence of (Symbol, lower, upper)
        Limits, innermost first.
    ctx : Context
        Options; every step is recorded in its log.

    Returns
    -------
    sp.Expr
        The integral, or an Integral of the partial result over the
        remaining limits.
    """
    group = tuple(lim[0] for lim in limits)
    limits = list(limits)
    while limits:
        var, *bounds = limits[0]
        entry: LogEntry = {"var": var, "group": group, "method": None,
                           "errors": {}}
        ruled = _rule_step(current, limits, ctx, entry)
        if ruled is not None:
            i, entry["method"], current = ruled
            entry["var"] = limits.pop(i)[0]
            ctx.record(entry)
            continue
        done = _backend_step(current, var, bounds, ctx, entry)
        if done is not None:
            entry["method"], current = done
            limits.pop(0)
        ctx.record(entry)
        if done is None:
            break
    return Integral(current, *limits) if limits else current
