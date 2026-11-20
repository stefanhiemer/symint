"""
SymPy <-> Sage expression conversion.

Sage's SymPy conversion returns plain Symbols; `from_sage` restores the
assumptions from the original symbols.
"""

from typing import Any, Sequence

import sympy as sp
from sympy import Symbol

from .loader import sage


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
