"""
Single CAS integration steps through Sage.
"""

import re
from functools import partial
from typing import Any, Sequence

from .loader import sage
from .questions import with_sign_cases

CAS_BACKENDS = ("maxima", "fricas", "giac", "fricas_antiderivative")
ANTIDERIVATIVE = "fricas_antiderivative"


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
    elif mode == "full":
        return expr.simplify_full()
    elif mode == "radical":
        return expr.simplify_full().canonicalize_radical()
    else:
        raise ValueError(f"unknown simplify mode {mode!r}; "
                         "use None, 'full' or 'radical'")

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
                   backend: str,
                   limit_algorithm: str = "maxima") -> Any:
    """
    One Sage integration step.

    Parameters
    ----------
    integrand : sage.symbolic.expression.Expression
        Integrand.
    args : sequence
        ``(var,)`` or ``(var, lower, upper)`` as Sage expressions.
    backend : str
        One of `CAS_BACKENDS`. `ANTIDERIVATIVE` asks FriCAS for an
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
    if backend != ANTIDERIVATIVE:
        return with_sign_cases(partial(sage_all.integrate, integrand, *args,
                                       algorithm=backend))
    var, *bounds = args
    F = sage_all.integrate(integrand, var, algorithm="fricas")
    if has_unevaluated_integral(F):
        raise ValueError("FriCAS found no antiderivative")
    if not bounds:
        return F
    lo, hi = bounds
    return (_endpoint(F, var, hi, "-", limit_algorithm)
            - _endpoint(F, var, lo, "+", limit_algorithm))
