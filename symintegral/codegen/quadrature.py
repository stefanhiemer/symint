"""
Unevaluated Integrals in generated code: each becomes a nested integrand
function and a ``scipy.integrate.nquad`` call over the remaining
variables.
"""

from typing import List, Tuple

import sympy as sp
from sympy import Integral, Symbol
from sympy.printing.numpy import SciPyPrinter


def _quadrature_lines(integral: Integral,
                      k: int,
                      printer: SciPyPrinter,
                      indent: str) -> List[str]:
    """
    Source lines computing one Integral with ``scipy.integrate.nquad``.

    Defines a local ``_integrand<k>`` taking the integration variables
    (innermost first, the order nquad expects) and assigns the result to
    ``_integral<k>``. Each variable gets a range callable receiving the
    variables outside it, so that bounds may depend on them.

    Parameters
    ----------
    integral : sympy.Integral
        Integral whose integrand contains no further Integrals.
    k : int
        Index distinguishing this integral's names from the others'.
    printer : SciPyPrinter
        Printer for the integrand and bounds.
    indent : str
        Indentation of the generated function body.

    Returns
    -------
    list of str
        The lines, to be placed before any use of ``_integral<k>``.

    Examples
    --------
    ``Integral(exp(-a*x*y), (x, 0, y), (y, 0, 1))`` with k = 0 gives::

        def _integrand0(x, y):
            return numpy.exp(-a*x*y)
        _integral0 = scipy.integrate.nquad(
            _integrand0, [lambda y, *_: (0, y), lambda *_: (0, 1)])[0]

    (shown wrapped; generated as one line). Free symbols such as ``a``
    are arguments of the enclosing generated function, which the nested
    ``_integrand0`` sees as a closure.
    """
    # collect/write integration limits
    variables = [lim[0] for lim in integral.limits]
    fargs = ", ".join(printer.doprint(v) for v in variables)
    ranges = []
    for i, (_, lo, hi) in enumerate(integral.limits):
        outer = ", ".join(printer.doprint(w) for w in variables[i + 1:])
        ranges.append(f"lambda {outer + ', ' if outer else ''}*_: "
                      f"({printer.doprint(sp.sympify(lo))}, "
                      f"{printer.doprint(sp.sympify(hi))})")
    # nested integrand function, then the nquad call; [0] drops the error
    # estimate nquad returns alongside the value
    return [f"{indent}def _integrand{k}({fargs}):",
            f"{indent}    return {printer.doprint(integral.function)}",
            f"{indent}_integral{k} = scipy.integrate.nquad("
            f"_integrand{k}, [{', '.join(ranges)}])[0]"]


def _integral_functions(exprs: List[sp.Expr],
                        printer: SciPyPrinter,
                        indent: str) -> Tuple[List[sp.Expr], List[str]]:
    r"""
    Replace unevaluated Integrals by numerical quadrature.

    Each Integral becomes a symbol ``_integral<k>`` computed by the lines
    from `_quadrature_lines`. Nested Integrals are replaced innermost
    first.

    Parameters
    ----------
    exprs : list of sp.Expr
        Flattened outputs, possibly containing Integrals.
    printer : SciPyPrinter
        Printer for the integrands and bounds.
    indent : str
        Indentation of the generated function body.

    Returns
    -------
    exprs : list of sp.Expr
        `exprs` with every Integral replaced by its symbol.
    lines : list of str
        Source lines defining the integrands and computing the symbols;
        they go before the expressions that use them.

    Notes
    -----
    A partial result 2 int_0^1 int_0^y exp(-a x^4) dx dy becomes
    ``2*_integral0``, with the inner bound passed to nquad as a function
    of the outer variable y:

    >>> import sympy as sp
    >>> from sympy.printing.numpy import SciPyPrinter
    >>> a, x, y = sp.symbols("a x y")
    >>> printer = SciPyPrinter({"fully_qualified_modules": True})
    >>> partial = 2 * sp.Integral(sp.exp(-a * x**4), (x, 0, y), (y, 0, 1))
    >>> exprs, lines = _integral_functions([partial], printer, "")
    >>> exprs
    [2*_integral0]
    >>> print("\n".join(lines))
    def _integrand0(x, y):
        return numpy.exp(-a*x**4)
    _integral0 = scipy.integrate.nquad(_integrand0, [lambda y, *_: (0, y), lambda *_: (0, 1)])[0]
    """
    lines: List[str] = []
    out = []
    k = 0
    for e in exprs:
        while e.has(Integral):
            innermost = [i for i in e.atoms(Integral)
                         if not i.function.has(Integral)]
            replacements = {}
            for integral in innermost:
                lines += _quadrature_lines(integral, k, printer, indent)
                replacements[integral] = Symbol(f"_integral{k}")
                k += 1
            e = e.xreplace(replacements)
        out.append(e)
    return out, lines
