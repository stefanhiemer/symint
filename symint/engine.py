"""
engine.py -- the symint integration pipeline.

``integrate(integral, ...)``:

1. rewrite: optional integral transform of inverse distances
   (`transform`), producing pieces with extra outer variables;
2. separate: on rectangular domains, split into products of
   lower-dimensional integrals, each distinct one evaluated once;
3. per variable: closed-form rules (`rules.RULES`) first -- on any
   variable if the domain is rectangular -- then the algorithms in
   `algorithms` in order ("sympy" natively, CAS ones through Sage), each
   under `step_timeout`;
4. outer transform variables last, with the Boys substitution onto [0, 1];
5. whatever cannot be integrated is returned as SymPy Integrals over the
   remaining variables (a partial result, never an exception), and every
   step is recorded in `log`.

Sage is imported only if a CAS step is actually attempted.
"""

import signal
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Integral, Symbol

from . import sage as sage_mod
from ._util import (additive_terms, fresh, is_half_line, is_rectangular,
                    is_whole_line)
from .rewrite import (TRANSFORMS, apply_transform, boys_substitute,
                      common_shift, complete_square)
from .rules import Rule, apply_rules
from .separate import separate as separate_terms, tidy_exponentials

DEFAULT_ALGORITHMS: Tuple[str, ...] = ("maxima", "fricas")

LogEntry = Dict[str, Any]
Limits = List[Tuple]


class StepTimeout(Exception):
    """An integration step ran out of time."""


def _raise_timeout(signum: int,
                   frame: Any) -> None:
    """
    SIGALRM handler of `run_with_timeout` without Sage.

    Parameters
    ----------
    signum : int
        Signal number.
    frame : frame
        Interrupted stack frame.

    Raises
    ------
    StepTimeout
        Always.
    """
    raise StepTimeout()


def _run_with_sage_alarm(fn: Callable[[], Any],
                         seconds: float) -> Any:
    """
    Run `fn` under a cysignals alarm (Sage loaded).

    Once Sage is loaded it owns SIGALRM, and only its alarm interrupts
    Maxima and FriCAS.

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float
        Time limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If the alarm fires. Sage's pexpect interfaces (FriCAS) turn the
        alarm into a KeyboardInterrupt; one arriving clearly before the
        deadline is a genuine Ctrl-C and re-raised.
    """
    from cysignals.alarm import AlarmInterrupt, alarm, cancel_alarm
    start = time.monotonic()
    alarm(seconds)
    try:
        return fn()
    except AlarmInterrupt:
        raise StepTimeout() from None
    except KeyboardInterrupt:
        if time.monotonic() - start < 0.99 * seconds:
            raise
        raise StepTimeout() from None
    finally:
        cancel_alarm()


def _run_with_itimer(fn: Callable[[], Any],
                     seconds: float) -> Any:
    """
    Run `fn` under ``signal.setitimer`` (Sage not loaded).

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float
        Time limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If the timer fires. The previous SIGALRM handler is restored.
    """
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_with_timeout(fn: Callable[[], Any],
                     seconds: Optional[float]) -> Any:
    """
    Run a function with a wall-clock limit.

    Only in the main thread, on Unix. Uses cysignals' alarm once Sage is
    loaded, else ``signal.setitimer``.

    Parameters
    ----------
    fn : callable
        Function without arguments.
    seconds : float or None
        Time limit; None or 0 runs `fn` without a limit.

    Returns
    -------
    Any
        Result of `fn`.

    Raises
    ------
    StepTimeout
        If `fn` takes longer than `seconds`.
    """
    if not seconds:
        return fn()
    if sage_mod.sage_loaded():
        return _run_with_sage_alarm(fn, seconds)
    return _run_with_itimer(fn, seconds)


@dataclass
class _Context:
    """
    Options and state of one `integrate` call.

    Attributes
    ----------
    session : SageSession
        All symbols of the integral (``session.symbols``, from which
        assumptions are restored after Sage conversions) and all limits;
        Sage assumptions are made only when a CAS step needs them.
    algorithms, simplify, complete_square, limit_algorithm, step_timeout,
    rules, log
        As in `integrate`.
    """

    session: sage_mod.SageSession
    algorithms: Tuple[str, ...]
    simplify: Optional[str]
    complete_square: bool
    limit_algorithm: str
    step_timeout: Optional[float]
    rules: Optional[Sequence[Rule]]
    log: Optional[List[LogEntry]]

    @property
    def uses_sage(self) -> bool:
        """Whether any of the algorithms goes through Sage."""
        return any(a != "sympy" for a in self.algorithms)

    def timed(self,
              fn: Callable[[], Any]) -> Any:
        """
        Run `fn` under the step timeout.

        Parameters
        ----------
        fn : callable
            Function without arguments.

        Returns
        -------
        Any
            Result of `fn`; raises StepTimeout, see `run_with_timeout`.
        """
        return run_with_timeout(fn, self.step_timeout)

    def record(self,
               entry: LogEntry) -> None:
        """
        Append a step to the log, if one was given.

        Parameters
        ----------
        entry : dict
            Log entry, see `integrate`.

        Returns
        -------
        None
        """
        if self.log is not None:
            self.log.append(entry)


def _prepare(current: sp.Expr,
             var: Symbol,
             bounds: List,
             ctx: _Context) -> Tuple[sp.Expr, List]:
    """
    Integrand and bounds handed to the algorithms for one step.

    Simplifies with Sage (if a Sage algorithm is used and `ctx.simplify`
    is set), then completes the square: term-wise on the whole line, by
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
    ctx : _Context
        Options.

    Returns
    -------
    (sp.Expr, list)
        The prepared integrand and the (possibly shifted) bounds.
    """
    expr = current
    if ctx.uses_sage and ctx.simplify:
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
              algorithm: str,
              ctx: _Context) -> sp.Expr:
    """
    One integration with one algorithm.

    Parameters
    ----------
    integrand : sp.Expr
        Prepared integrand.
    var : Symbol
        Integration variable.
    bounds : list
        ``[lower, upper]``.
    algorithm : str
        "sympy" (native) or a Sage algorithm.
    ctx : _Context
        Options and Sage session.

    Returns
    -------
    sp.Expr
        The integral.

    Raises
    ------
    ValueError
        "unevaluated" if the algorithm returns an unevaluated integral;
        any other exception of the algorithm is passed on.
    """
    if algorithm == "sympy":
        result = sp.integrate(integrand, (var, *bounds))
        if result.has(Integral):
            raise ValueError("unevaluated")
        return result
    ctx.session.ensure()
    args = [sage_mod.to_sage(var)] + [sage_mod.to_sage(b) for b in bounds]
    result = sage_mod.integrate_step(sage_mod.to_sage(integrand), args,
                                     algorithm, ctx.limit_algorithm)
    if sage_mod.has_unevaluated_integral(result):
        raise ValueError("unevaluated")
    return sage_mod.from_sage(result, ctx.session.symbols)


def _timeout_message(ctx: _Context) -> str:
    """
    Log message of a timed-out step.

    Parameters
    ----------
    ctx : _Context
        Options (for the timeout).

    Returns
    -------
    str
        "timeout after <step_timeout>s".
    """
    return f"timeout after {ctx.step_timeout}s"


def _rule_step(current: sp.Expr,
               limits: Limits,
               ctx: _Context,
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
    ctx : _Context
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


def _algorithm_step(current: sp.Expr,
                    var: Symbol,
                    bounds: List,
                    ctx: _Context,
                    entry: LogEntry) -> Optional[Tuple[str, sp.Expr]]:
    """
    Integrate one variable with the algorithms, in order.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : list
        ``[lower, upper]``.
    ctx : _Context
        Options.
    entry : dict
        Log entry; failures of the preparation and of every algorithm are
        recorded in its "errors".

    Returns
    -------
    (str, sp.Expr) or None
        Name of the successful algorithm and result; None if all failed.
    """
    try:
        integrand, step_bounds = ctx.timed(
            partial(_prepare, current, var, bounds, ctx))
    except StepTimeout:
        entry["errors"]["prepare"] = _timeout_message(ctx)
        integrand, step_bounds = current, list(bounds)
    except Exception as err:  # e.g. no Sage counterpart of a function
        entry["errors"]["prepare"] = f"{type(err).__name__}: {err}"
        integrand, step_bounds = current, list(bounds)
    for algorithm in ctx.algorithms:
        try:
            result = ctx.timed(partial(_cas_step, integrand, var,
                                       step_bounds, algorithm, ctx))
        except StepTimeout:
            entry["errors"][algorithm] = _timeout_message(ctx)
            continue
        except Exception as err:  # Maxima questions, conversions, ...
            msg = str(err)
            entry["errors"][algorithm] = msg if msg == "unevaluated" \
                else f"{type(err).__name__}: {err}"
            continue
        return algorithm, result
    return None


def evaluate_iterated(current: sp.Expr,
                      limits: Sequence[Tuple],
                      ctx: _Context) -> sp.Expr:
    """
    Integrate one variable at a time.

    Each step tries the rules (`_rule_step`), then the algorithms on the
    innermost variable (`_algorithm_step`); the first variable nothing
    can integrate ends the iteration.

    Parameters
    ----------
    current : sp.Expr
        Integrand.
    limits : sequence of (Symbol, lower, upper)
        Limits, innermost first.
    ctx : _Context
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
        entry: LogEntry = {"var": var, "group": group, "algorithm": None,
                           "errors": {}}
        ruled = _rule_step(current, limits, ctx, entry)
        if ruled is not None:
            i, entry["algorithm"], current = ruled
            entry["var"] = limits.pop(i)[0]
            ctx.record(entry)
            continue
        done = _algorithm_step(current, var, bounds, ctx, entry)
        if done is not None:
            entry["algorithm"], current = done
            limits.pop(0)
        ctx.record(entry)
        if done is None:
            break
    return Integral(current, *limits) if limits else current


def _evaluate_outer(value: sp.Expr,
                    outer: Limits,
                    transform: Optional[str],
                    substitute_outer: bool,
                    ctx: _Context) -> sp.Expr:
    """
    Integrate the outer transform variables.

    For a single "gaussian"/"laplace" variable (and `substitute_outer`),
    each term with a unique Boys parameter is first mapped onto
    u in [0, 1] (`boys_substitute`); the other terms keep the original
    variable.

    Parameters
    ----------
    value : sp.Expr
        Result of the inner integrals, integrand in the outer variables.
    outer : list of (Symbol, lower, upper)
        Limits of the transform variables.
    transform : str or None
        Transform used.
    substitute_outer : bool
        Whether to apply the Boys substitution.
    ctx : _Context
        Options.

    Returns
    -------
    sp.Expr
        The integral, possibly partial.
    """
    if not (substitute_outer and transform in ("gaussian", "laplace")
            and len(outer) == 1):
        return evaluate_iterated(value, outer, ctx)
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
                    ctx: _Context) -> sp.Expr:
    """
    Integrate over the original variables, separating if possible.

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
    ctx : _Context
        Options.

    Returns
    -------
    sp.Expr
        The integral, possibly partial.
    """
    if not (separate and is_rectangular(limits)):
        return evaluate_iterated(function, limits, ctx)
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
              algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
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
    Evaluate an integral as far as possible in closed form.

    Parameters
    ----------
    integral : sympy.Integral
        The unevaluated integral, e.g. built with `chart_integral`,
        `simplex_integral` or `symfem_integral`.
    algorithms : sequence of str, optional
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
        Integral representation of inverse-distance factors Q^(-nu/2)
        (Q quadratic in the integration variables); the new variables
        are integrated last. "gaussian": 2/Gamma(nu/2) int_0^oo
        t^(nu-1) exp(-t^2 Q) dt (Boys); "laplace": s = t^2; "fourier":
        1/|l| = 1/(2 pi^2) int d^3k exp(i k.l)/k^2.
    substitute_outer : bool, optional
        For "gaussian"/"laplace", map the final t-/s-integral onto [0, 1]
        by u^2 = t^2/(p + t^2) (Boys-function form).
    step_timeout : float, optional
        Seconds per step, rule pass and algorithm; None = no limit.
    rules : sequence of Rule, optional
        Rules to try (default: the registry `rules.RULES`; () disables).
    log : list, optional
        Receives one dict per step: "var", "group", "algorithm" (rule
        name or algorithm that succeeded, or None) and "errors".

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
    limits = [tuple(lim) for lim in integral.limits]
    taken = {str(s) for s in integral.atoms(Symbol)}
    pieces = ([(integral.function, [])] if transform is None else
              apply_transform(integral.function, limits, transform, taken))
    all_limits = limits + [lim for _, outer in pieces for lim in outer]
    symbols = sorted(set(integral.atoms(Symbol))
                     | {lim[0] for lim in all_limits}, key=str)
    ctx = _Context(session=sage_mod.SageSession(symbols, all_limits),
                   algorithms=tuple(algorithms),
                   simplify=simplify,
                   complete_square=complete_square,
                   limit_algorithm=limit_algorithm,
                   step_timeout=step_timeout,
                   rules=rules,
                   log=log)
    cache: Dict[Integral, sp.Expr] = {}
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
    finally:
        ctx.session.close()
