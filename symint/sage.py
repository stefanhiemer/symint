"""
sage.py -- SymPy <-> Sage (passagemath) bridge for symint: conversion,
assumptions, Maxima's sign/integer questions, and single CAS
integration steps.

Optional dependency (passagemath-symbolics[maxima,fricas]); imported
lazily, so symint works without it as long as no CAS step is needed.

SymPy assumptions on the free symbols (positive, nonnegative, negative,
real, integer) and the integration ranges (e.g. xi >= 1) are passed to
Sage via ``assume`` for the duration of a `SageSession`, and restored on
the way back, since Sage's SymPy conversion returns plain Symbols.

FriCAS is started through a launcher that needs passagemath's bundled
`ecl` on PATH; `ensure_ecl_on_path` adds it before Sage is imported.
"""

import os
import re
import sys
import sysconfig
from functools import partial
from typing import Any, Callable, List, Sequence, Tuple

import sympy as sp
from sympy import Symbol

CAS_ALGORITHMS = ("maxima", "fricas", "giac", "fricas_antiderivative")
ANTIDERIVATIVE = "fricas_antiderivative"
MAX_SIGN_CASES = 3  # nested Maxima sign questions answered per step


def ensure_ecl_on_path() -> None:
    """
    Put passagemath's bundled binaries on PATH.

    Prepends ``sage_wheels/bin`` (ecl, maxima, fricas) of the first
    site-packages directory that has it, unless it is already on PATH.
    The FriCAS launcher needs ``ecl`` there.

    Returns
    -------
    None
    """
    paths = sysconfig.get_paths()
    for base in dict.fromkeys([paths["platlib"], paths["purelib"],
                               *sys.path]):
        bindir = os.path.join(base or ".", "sage_wheels", "bin")
        if os.path.isfile(os.path.join(bindir, "ecl")):
            path = os.environ.get("PATH", "").split(os.pathsep)
            if bindir not in path:
                os.environ["PATH"] = os.pathsep.join([bindir, *path])
            return


def sage() -> Any:
    """
    Import the Sage symbolics namespace.

    Returns
    -------
    module
        ``sage.all__sagemath_symbolics`` (imported on first call).

    Raises
    ------
    ImportError
        If passagemath is not installed.
    """
    ensure_ecl_on_path()
    try:
        import sage.all__sagemath_symbolics as sage_all
    except ImportError as err:
        raise ImportError(
            "CAS integration needs passagemath; install "
            "passagemath-symbolics[maxima,fricas] and passagemath-repl"
        ) from err
    return sage_all


def sage_loaded() -> bool:
    """
    Whether Sage has been imported already.

    Returns
    -------
    bool
        True once `sage` (or anything else) imported the Sage symbolics;
        decides which timeout mechanism `run_with_timeout` uses.
    """
    return "sage.all__sagemath_symbolics" in sys.modules


def to_sage(expr: sp.Expr) -> Any:
    """
    Convert a SymPy expression to Sage.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression. Functions without a Sage counterpart must be
        rewritten first.

    Returns
    -------
    sage.symbolic.expression.Expression
        The Sage symbolic expression.
    """
    return sage().SR(sp.sympify(expr))


def from_sage(expr: Any,
              symbols: Sequence[Symbol] = ()) -> sp.Expr:
    """
    Convert a Sage expression to SymPy, restoring assumptions.

    Parameters
    ----------
    expr : sage.symbolic.expression.Expression
        Sage expression.
    symbols : sequence of Symbol, optional
        Symbols with assumptions; plain symbols of the same names in the
        result are replaced by these.

    Returns
    -------
    sp.Expr
        The SymPy expression.
    """
    result = sp.sympify(expr._sympy_())
    by_name = {s.name: s for s in symbols}
    return result.xreplace({s: by_name[s.name] for s in result.free_symbols
                            if s.name in by_name and s != by_name[s.name]})


def symbol_facts(symbols: Sequence[Symbol]) -> List[Tuple]:
    """
    SymPy symbol assumptions as Sage ``assume`` arguments.

    Parameters
    ----------
    symbols : sequence of Symbol
        Symbols whose assumptions are translated (integer or real; then
        positive, nonnegative or negative).

    Returns
    -------
    list of tuple
        Argument tuples for ``assume``, e.g. ``(a, "real")``, ``(a > 0,)``.
    """
    SR = sage().SR
    facts: List[Tuple] = []
    for s in symbols:
        v = SR(s)
        if s.is_integer:
            facts.append((v, "integer"))
        elif s.is_real:
            facts.append((v, "real"))
        if s.is_positive:
            facts.append((v > 0,))
        elif s.is_nonnegative:
            facts.append((v >= 0,))
        elif s.is_negative:
            facts.append((v < 0,))
    return facts


def limit_facts(limits: Sequence[Tuple]) -> List[Tuple]:
    """
    Integration ranges as Sage ``assume`` arguments.

    Parameters
    ----------
    limits : sequence of (Symbol, lower, upper)
        Integration limits.

    Returns
    -------
    list of tuple
        ``(v >= lower,)`` and ``(v <= upper,)`` for every finite bound
        free of integration variables, e.g. (xi, 1, oo) -> xi >= 1.
    """
    SR = sage().SR
    variables = {lim[0] for lim in limits}
    facts: List[Tuple] = []
    for var, *bounds in limits:
        if len(bounds) != 2:
            continue
        v = SR(var)
        lo, hi = (sp.sympify(bd) for bd in bounds)
        if lo.is_finite and not (lo.free_symbols & variables):
            facts.append((v >= to_sage(lo),))
        if hi.is_finite and not (hi.free_symbols & variables):
            facts.append((v <= to_sage(hi),))
    return facts


class SageSession:
    """
    Sage assumptions for one integration.

    The assumptions are made only when a CAS step first needs them
    (`ensure`), so that integrations done entirely by rules or SymPy
    never import Sage, and are all forgotten on `close`. Further symbols
    and limits (e.g. substitution variables) can be added at any time
    with `extend`.

    Parameters
    ----------
    symbols : sequence of Symbol, optional
        Symbols whose assumptions are passed to Sage.
    limits : sequence of (Symbol, lower, upper), optional
        Integration ranges passed to Sage as inequalities.

    Attributes
    ----------
    symbols : list of Symbol
        Symbols of the integration, kept even while no assumptions are
        made; the engine also restores assumptions after conversions
        (`from_sage`) from them.
    limits : list of (Symbol, lower, upper)
        Integration ranges.
    facts : list of tuple
        Assumptions made so far (what `close` forgets).
    active : bool
        Whether `ensure` has been called since the last `close`.
    """

    def __init__(self,
                 symbols: Sequence[Symbol] = (),
                 limits: Sequence[Tuple] = ()) -> None:
        self.symbols = list(symbols)
        self.limits = list(limits)
        self.facts: List[Tuple] = []
        self.active = False

    def _assume(self,
                facts: Sequence[Tuple]) -> None:
        """
        Make assumptions in Sage and remember them.

        Parameters
        ----------
        facts : sequence of tuple
            ``assume`` argument tuples. Facts that Sage rejects as
            redundant or inconsistent with existing ones are skipped.

        Returns
        -------
        None
        """
        sage_all = sage()
        for fact in facts:
            try:
                sage_all.assume(*fact)
                self.facts.append(fact)
            except ValueError:  # redundant/inconsistent with existing fact
                pass

    def ensure(self) -> None:
        """
        Make the session's assumptions, once.

        Returns
        -------
        None
        """
        if not self.active:
            self.active = True
            self._assume(symbol_facts(self.symbols)
                         + limit_facts(self.limits))

    def extend(self,
               symbols: Sequence[Symbol] = (),
               limits: Sequence[Tuple] = ()) -> None:
        """
        Add symbols and limits (assumed at once if the session is active).

        Parameters
        ----------
        symbols : sequence of Symbol, optional
            Further symbols with assumptions.
        limits : sequence of (Symbol, lower, upper), optional
            Further integration ranges.

        Returns
        -------
        None
        """
        self.symbols += [s for s in symbols if s not in self.symbols]
        self.limits += list(limits)
        if self.active:
            self._assume(symbol_facts(symbols) + limit_facts(limits))

    def close(self) -> None:
        """
        Forget all assumptions made by this session.

        Returns
        -------
        None
        """
        if not self.active:
            return
        sage_all = sage()
        for fact in reversed(self.facts):
            try:
                sage_all.forget(*fact)
            except ValueError:
                pass
        self.facts, self.active = [], False


def simplify(expr: Any,
             mode: str) -> Any:
    """
    Pre-simplify a Sage integrand.

    Parameters
    ----------
    expr : sage.symbolic.expression.Expression
        Integrand.
    mode : {None, "full", "radical"}
        None: unchanged; "full": ``simplify_full``; "radical": also
        ``canonicalize_radical`` (e.g. sqrt((xi + eta)^2) -> xi + eta;
        not branch-safe).

    Returns
    -------
    sage.symbolic.expression.Expression
        The simplified integrand.
    """
    if mode is None:
        return expr
    if mode == "full":
        return expr.simplify_full()
    if mode == "radical":
        return expr.simplify_full().canonicalize_radical()
    raise ValueError(f"unknown simplify mode {mode!r}; "
                     "use None, 'full' or 'radical'")


_SIGN_QUESTION = re.compile(
    r"Is\s+(?P<expr>.+?)\s+(?P<kind>positive, negative or zero|"
    r"positive or negative|positive or zero|negative or zero)\?", re.S)
_INTEGER_QUESTION = re.compile(r"Is\s+(?P<expr>.+?)\s+an integer\?", re.S)


def _question(err: Exception,
              pattern: re.Pattern) -> Any:
    """
    The expression Maxima asks about in an error message.

    Parameters
    ----------
    err : Exception
        Error raised by a Sage integrate/limit call.
    pattern : re.Pattern
        `_SIGN_QUESTION` or `_INTEGER_QUESTION`.

    Returns
    -------
    sage.symbolic.expression.Expression or None
        The expression q of "Is q positive ...?" / "Is q an integer?", or
        None if the message is no such question or q cannot be parsed.
    """
    match = pattern.search(str(err))
    if match is None:
        return None
    try:
        return sage().SR(" ".join(match.group("expr").split()))
    except Exception:
        return None


def _equal(u: Any,
           v: Any) -> bool:
    """
    Whether two Sage expressions are equal.

    Parameters
    ----------
    u, v : sage.symbolic.expression.Expression
        Expressions to compare.

    Returns
    -------
    bool
        True if ``simplify_full`` of u - v is 0, or ``simplify_full`` plus
        ``canonicalize_radical`` of u / v is 1 (merges split
        exponentials). False means "not shown equal".
    """
    if (u - v).simplify_full() == 0:
        return True
    if v == 0:
        return False
    try:
        return bool((u / v).simplify_full().canonicalize_radical() == 1)
    except Exception:
        return False


def _noninteger_variables(err: Exception) -> List[Any]:
    """
    Variables of an "Is q an integer?" question that may be declared
    non-integer.

    Parameters
    ----------
    err : Exception
        Error raised by a Sage integrate/limit call.

    Returns
    -------
    list
        The variables of q not known to be integers; empty if `err` is
        no integer question.
    """
    q = _question(err, _INTEGER_QUESTION)
    if q is None:
        return []
    return [v for v in q.variables() if not v.is_integer()]


def _with_noninteger(compute: Callable[[], Any],
                     variables: Sequence[Any],
                     depth: int) -> Any:
    """
    Run `compute` with `variables` assumed non-integer (generic case).

    Parameters
    ----------
    compute : callable
        Sage integrate/limit call without arguments.
    variables : sequence
        Sage variables to declare non-integer for the call.
    depth : int
        Current number of nested questions.

    Returns
    -------
    Any
        Result of `compute`, see `with_sign_cases`.
    """
    sage_all = sage()
    declared = []
    try:
        for v in variables:
            try:
                sage_all.assume(v, "noninteger")
                declared.append(v)
            except ValueError:
                pass
        return with_sign_cases(compute, depth + 1)
    finally:
        for v in declared:
            sage_all.forget(v, "noninteger")


def _with_sign_split(compute: Callable[[], Any],
                     q: Any,
                     depth: int) -> Any:
    """
    Run `compute` under q > 0 and q < 0; accept if both agree.

    The case q = 0 follows by continuity.

    Parameters
    ----------
    compute : callable
        Sage integrate/limit call without arguments.
    q : sage.symbolic.expression.Expression
        Expression Maxima asked the sign of.
    depth : int
        Current number of nested questions.

    Returns
    -------
    Any
        The common result.

    Raises
    ------
    ValueError
        If no case is consistent with the current assumptions, or the
        results of the two cases differ.
    """
    sage_all = sage()
    results = []
    for relation in (q > 0, q < 0):
        try:
            sage_all.assume(relation)
        except ValueError:  # contradicts the current assumptions
            continue
        try:
            results.append(with_sign_cases(compute, depth + 1))
        finally:
            sage_all.forget(relation)
    if not results:
        raise ValueError(f"no consistent sign case for {q}")
    if not all(_equal(r, results[0]) for r in results[1:]):
        raise ValueError(f"result depends on the sign of {q}: "
                         f"{q} > 0 -> {results[0]}, {q} < 0 -> {results[1]}")
    return results[0]


def with_sign_cases(compute: Callable[[], Any],
                    depth: int = 0) -> Any:
    """
    Run a Sage computation, answering Maxima's questions.

    "Is q an integer?" is answered by declaring the symbols of q
    non-integer (the generic case, `_with_noninteger`); sign questions by
    computing under q > 0 and q < 0 and accepting the result if both
    agree (`_with_sign_split`). Nested questions are answered up to
    `MAX_SIGN_CASES` deep.

    Parameters
    ----------
    compute : callable
        Sage integrate/limit call without arguments.
    depth : int, optional
        Current number of nested questions.

    Returns
    -------
    Any
        Result of `compute`.

    Raises
    ------
    ValueError
        The original error if it is no answerable question or the depth
        limit is reached; see also `_with_sign_split`.
    """
    try:
        return compute()
    except ValueError as err:
        if depth >= MAX_SIGN_CASES:
            raise
        generic = _noninteger_variables(err)
        q = None if generic else _question(err, _SIGN_QUESTION)
        if not generic and q is None:
            raise
    if generic:
        return _with_noninteger(compute, generic, depth)
    return _with_sign_split(compute, q, depth)


_BAD_LIMIT = re.compile(r"\b(und|ind|infinity)\b|\blimit\(")


def _endpoint(F: Any,
              var: Any,
              point: Any,
              side: str,
              limit_algorithm: str) -> Any:
    """
    One-sided limit of an antiderivative at an interval end.

    Parameters
    ----------
    F : sage.symbolic.expression.Expression
        Antiderivative.
    var : sage.symbolic.expression.Expression
        Integration variable.
    point : Any
        Interval end (may be infinite).
    side : {"+", "-"}
        Direction, from inside the interval.
    limit_algorithm : str
        Sage limit algorithm.

    Returns
    -------
    sage.symbolic.expression.Expression
        The limit.

    Raises
    ------
    ValueError
        If the limit is infinite, undefined or unevaluated.
    """
    point = sage().SR(point)
    kwargs = {} if point.is_infinity() else {"dir": side}
    value = with_sign_cases(partial(F.limit, **{str(var): point},
                                    algorithm=limit_algorithm, **kwargs))
    if value.is_infinity() or _BAD_LIMIT.search(str(value)):
        raise ValueError(f"limit of antiderivative at {var} -> {point} "
                         f"is {value}")
    return value


def has_unevaluated_integral(expr: Any) -> bool:
    """
    Whether a Sage expression still contains an integral.

    Parameters
    ----------
    expr : sage.symbolic.expression.Expression
        Result of a Sage integration.

    Returns
    -------
    bool
        True if any node is a definite or indefinite ``integrate``.
    """
    from sage.symbolic.integration.integral import (DefiniteIntegral,
                                                    IndefiniteIntegral)
    stack = [sage().SR(expr)]
    while stack:
        e = stack.pop()
        if isinstance(e.operator(), (DefiniteIntegral, IndefiniteIntegral)):
            return True
        stack.extend(e.operands())
    return False


def integrate_step(integrand: Any,
                   args: Sequence[Any],
                   algorithm: str,
                   limit_algorithm: str = "maxima") -> Any:
    """
    One Sage integration step.

    Parameters
    ----------
    integrand : sage.symbolic.expression.Expression
        Integrand.
    args : sequence
        ``(var,)`` or ``(var, lower, upper)`` as Sage expressions.
    algorithm : str
        One of `CAS_ALGORITHMS`. `ANTIDERIVATIVE` asks FriCAS for an
        antiderivative F and returns F(upper^-) - F(lower^+) via
        one-sided limits -- only valid if F is continuous on the interval.
    limit_algorithm : str, optional
        Sage limit algorithm for `ANTIDERIVATIVE`.

    Returns
    -------
    sage.symbolic.expression.Expression
        The (possibly unevaluated, see `has_unevaluated_integral`)
        integral.
    """
    sage_all = sage()
    if algorithm != ANTIDERIVATIVE:
        return with_sign_cases(partial(sage_all.integrate, integrand, *args,
                                       algorithm=algorithm))
    var, *bounds = args
    F = sage_all.integrate(integrand, var, algorithm="fricas")
    if has_unevaluated_integral(F):
        raise ValueError("FriCAS found no antiderivative")
    if not bounds:
        return F
    lo, hi = bounds
    return (_endpoint(F, var, hi, "-", limit_algorithm)
            - _endpoint(F, var, lo, "+", limit_algorithm))
