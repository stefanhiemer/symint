"""
Answering Maxima's sign and integer questions.

Maxima stops with "Is q positive, negative or zero?" or "Is q an
integer?" when it cannot decide a case. `with_sign_cases` answers these
by case splits and accepts a result only if all cases agree.
"""

import re
from typing import Any, Callable, List, Sequence

from .loader import sage

MAX_SIGN_CASES = 3  # nested Maxima sign questions answered per step


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
