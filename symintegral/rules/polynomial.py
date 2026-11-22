"""
"polynomial" rule: polynomials in the variable over finite ranges (FEM
integrands on affine or Duffy-collapsed cells), via exact
antiderivatives.
"""

from typing import Optional

import sympy as sp
from sympy import Symbol

from .base import Bounds


def polynomial_rule(expr: sp.Expr,
                    var: Symbol,
                    bounds: Bounds) -> Optional[sp.Expr]:
    """
    Exact integral of a polynomial over a finite range.

    Parameters
    ----------
    expr : sp.Expr
        Integrand, polynomial in `var` with arbitrary coefficients.
    var : Symbol
        Integration variable.
    bounds : sequence
        Finite ``(lower, upper)``.

    Returns
    -------
    sp.Expr or None
        F(upper) - F(lower) with F the antiderivative, or None if `expr`
        is not polynomial in `var`.

    Examples
    --------
    >>> import sympy as sp
    >>> x, a = sp.symbols("x a")
    >>> polynomial_rule(a * x**2, x, (0, 1))
    a/3
    """
    if not expr.has(var):
        return expr * (bounds[1] - bounds[0])
    elif not expr.is_polynomial(var):
        return None
    else:
        F = sp.Poly(sp.expand(expr), var).integrate().as_expr()
        return F.subs(var, bounds[1]) - F.subs(var, bounds[0])
