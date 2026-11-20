"""
domain.py -- build SymPy Integrals over common integration domains.

`integrate` takes a plain ``sympy.Integral``; these functions build one
from an integrand and a description of where it is integrated:

- `chart_integral`: a coordinate-chart mapping with keys "limits" and
  "volume_element" (as used by symbolic_qc.coordinates),
- `simplex_integral`: the reference simplex, Duffy-collapsed onto the
  unit cube,
- `symfem_integral`: a symfem reference cell (duck-typed, symfem is not
  imported), with simplices Duffy-collapsed by default.

The Duffy collapse makes triangles and tetrahedra rectangular, so that
the engine can separate the integral and choose the integration order
freely.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import sympy as sp
from sympy import Integral, Symbol

from ._util import Limit


def chart_integral(expr: sp.Expr,
                   chart: Mapping[str, Any]) -> Integral:
    """
    Integral of `expr` over a coordinate chart.

    Parameters
    ----------
    expr : sp.Expr
        Integrand in the chart's coordinates.
    chart : mapping
        With "limits" (list of (Symbol, lower, upper), innermost first)
        and optionally "volume_element" (default 1), e.g. the charts of
        symbolic_qc.coordinates.

    Returns
    -------
    sympy.Integral
        Integral of ``expr * volume_element`` over the limits.

    Examples
    --------
    >>> import sympy as sp
    >>> r, th, ph = sp.symbols("r theta phi", positive=True)
    >>> sph = {"limits": [(r, 0, sp.oo), (th, 0, sp.pi), (ph, 0, 2 * sp.pi)],
    ...        "volume_element": r**2 * sp.sin(th)}
    >>> chart_integral(sp.exp(-r), sph)
    Integral(r**2*exp(-r)*sin(theta), (r, 0, oo), (theta, 0, pi), (phi, 0, 2*pi))
    """
    volume = sp.sympify(chart.get("volume_element", 1))
    return Integral(sp.sympify(expr) * volume, *chart["limits"])


def _duffy(variables: Sequence[Symbol],
           new: Sequence[Symbol] = None
           ) -> Tuple[List[Limit], sp.Expr, Dict[Symbol, sp.Expr]]:
    """
    Duffy map of the reference simplex onto the unit cube.

    Maps {x_i >= 0, x_1 + ... + x_d <= 1} onto [0, 1]^d by

        x_1 = u_1,  x_2 = (1 - u_1) u_2,  x_3 = (1 - u_1)(1 - u_2) u_3, ...

    with Jacobian prod_k (1 - u_k)^(d - k). Polynomials stay polynomials.

    Parameters
    ----------
    variables : sequence of Symbol
        Simplex coordinates x_1, ..., x_d.
    new : sequence of Symbol, optional
        Cube coordinates; default real ``u_<x>`` symbols.

    Returns
    -------
    limits : list of (Symbol, 0, 1)
        Limits of the cube coordinates.
    jacobian : sp.Expr
        Jacobian determinant of the map.
    substitution : dict
        x_i -> expression in the cube coordinates.
    """
    d = len(variables)
    new = list(new) if new else [Symbol(f"u_{v}", real=True)
                                 for v in variables]
    substitution, scale, jacobian = {}, sp.Integer(1), sp.Integer(1)
    for k, (x, u) in enumerate(zip(variables, new)):
        substitution[x] = scale * u
        jacobian *= (1 - u) ** (d - 1 - k)
        scale *= (1 - u)
    return [(u, 0, 1) for u in new], jacobian, substitution


def simplex_integral(expr: sp.Expr,
                     variables: Sequence[Symbol],
                     new: Sequence[Symbol] = None) -> Integral:
    """
    Integral of `expr` over the reference simplex, on the unit cube.

    The simplex {x_i >= 0, x_1 + ... + x_d <= 1} is Duffy-collapsed onto
    [0, 1]^d (see `_duffy`), so the resulting Integral is rectangular.

    Parameters
    ----------
    expr : sp.Expr
        Integrand in the simplex coordinates.
    variables : sequence of Symbol
        Simplex coordinates x_1, ..., x_d (d = 1, 2, 3, ...).
    new : sequence of Symbol, optional
        Cube coordinates; default real ``u_<x>`` symbols.

    Returns
    -------
    sympy.Integral
        Integral over the cube coordinates, Jacobian included.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols("x y")
    >>> simplex_integral(x, (x, y))
    Integral(u_x*(1 - u_x), (u_x, 0, 1), (u_y, 0, 1))
    """
    limits, jacobian, substitution = _duffy(variables, new)
    integrand = sp.sympify(expr).subs(substitution, simultaneous=True)
    return Integral(integrand * jacobian, *limits)


def symfem_integral(expr: sp.Expr,
                    reference: Any,
                    physical: Sequence[Symbol] = None,
                    dummy: Sequence[Symbol] = None,
                    collapse: bool = True) -> Integral:
    """
    Integral of `expr` over a symfem reference cell.

    Mirrors symfem's own ``ScalarFunction.integral``: the physical point
    is origin + sum_i t_i axes_i, the limits are
    ``reference.integration_limits(t)`` and the volume element is
    ``reference.jacobian()``. Works for intervals, triangles,
    quadrilaterals, tetrahedra, hexahedra, ...

    Parameters
    ----------
    expr : sp.Expr
        Integrand in the physical coordinates.
    reference : symfem.references.Reference
        The reference cell (only its public attributes are used).
    physical : sequence of Symbol, optional
        Physical coordinates appearing in `expr`; default symfem's
        ``x, y, z`` (plain Symbols "x", "y", "z").
    dummy : sequence of Symbol, optional
        Integration variables; default symfem's ``t0, t1, t2``.
    collapse : bool, optional
        For triangles and tetrahedra, Duffy-collapse onto the unit square
        / cube so the Integral is rectangular.

    Returns
    -------
    sympy.Integral
        Integral over the dummy (or, collapsed, the cube) variables,
        volume element included.
    """
    tdim = reference.tdim
    physical = list(physical or sp.symbols("x y z"))[:reference.gdim]
    dummy = list(dummy or sp.symbols("t0 t1 t2"))[:tdim]
    point = [sp.sympify(o) for o in reference.origin]
    for ti, axis in zip(dummy, reference.axes):
        point = [p + ti * sp.sympify(a) for p, a in zip(point, axis)]
    substitution = dict(zip(physical, point))
    volume = sp.sympify(reference.jacobian())
    if collapse and getattr(reference, "name", None) in ("triangle",
                                                         "tetrahedron"):
        limits, jacobian, duffy = _duffy(dummy)
        substitution = {x: e.subs(duffy, simultaneous=True)
                        for x, e in substitution.items()}
        volume *= jacobian
    else:
        limits = [tuple(lim) for lim in reference.integration_limits(dummy)]
    integrand = sp.sympify(expr).subs(substitution, simultaneous=True)
    return Integral(integrand * volume, *limits)
