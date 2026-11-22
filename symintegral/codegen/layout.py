"""
Function layout: flattening the return value, the signature with array
arguments unpacked into Symbols, and the return statement.
"""

from typing import Any, List, Sequence, Tuple, Union

import sympy as sp
from sympy import Symbol
from sympy.printing.numpy import SciPyPrinter


ArgSpec = Union[Symbol, Tuple[str, Any]]


def _flatten(exprs: Any) -> Tuple[List[sp.Expr], Tuple[str, Any]]:
    """
    Flatten the output of a generated function into a list.

    Parameters
    ----------
    exprs : sp.Expr, sp.Matrix, list, tuple or object with ``as_sympy``
        What the generated function returns. Objects with an
        ``as_sympy`` method are converted to a Matrix.

    Returns
    -------
    flat : list of sp.Expr
        The expressions, matrices in row-major order.
    kind : (str, Any)
        ``("matrix", shape)``, ``("list", length)`` or
        ``("scalar", None)``: how `numpy_function` rebuilds the return
        value from `flat`.
    """
    if isinstance(exprs, sp.MatrixBase):
        return list(exprs), ("matrix", exprs.shape)
    if hasattr(exprs, "as_sympy") and not isinstance(exprs, sp.Basic):
        return _flatten(sp.Matrix(exprs.as_sympy()))
    if isinstance(exprs, (list, tuple)):
        return [sp.sympify(e) for e in exprs], ("list", len(exprs))
    return [sp.sympify(exprs)], ("scalar", None)


def _signature(args: Sequence[ArgSpec],
               printer: SciPyPrinter,
               indent: str) -> Tuple[List[str], List[str]]:
    """
    Argument names of a generated function and the unpacking lines.

    Parameters
    ----------
    args : sequence
        Symbols, or ``(argname, symbols)`` for array arguments, see
        `numpy_function`.
    printer : SciPyPrinter
        Printer for the symbol names.
    indent : str
        Indentation of the function body.

    Returns
    -------
    signature : list of str
        Argument names, in order.
    unpack : list of str
        Lines ``c11 = c[0, 0]`` ... for every Symbol in an array
        argument.
    """
    signature, unpack = [], []
    for arg in args:
        if isinstance(arg, Symbol):
            signature.append(printer.doprint(arg))
            continue
        argname, symbols = arg
        signature.append(argname)
        array = sp.Array(symbols)
        for index in sp.utilities.iterables.cartes(
                *[range(n) for n in array.shape]):
            sym = array[index]
            if isinstance(sym, Symbol):
                idx = ", ".join(map(str, index))
                unpack.append(f"{indent}{printer.doprint(sym)} = "
                              f"{argname}[{idx}]")
    return signature, unpack


def _return_lines(values: Sequence[str],
                  kind: str,
                  shape: Any,
                  indent: str) -> List[str]:
    """
    The return statement of a generated function.

    Parameters
    ----------
    values : sequence of str
        Printed output expressions, matrices in row-major order.
    kind : {"scalar", "list", "matrix"}
        Output type, from `_flatten`.
    shape : (int, int) or None
        Matrix shape for "matrix".
    indent : str
        Indentation of the function body.

    Returns
    -------
    list of str
        One line for a scalar or list; a ``numpy.array`` with one line
        per row for a matrix.
    """
    if kind == "scalar":
        return [f"{indent}return {values[0]}"]
    if kind == "list":
        return [f"{indent}return [{', '.join(values)}]"]
    rows, cols = shape
    body = ",\n".join(
        f"{indent}    [{', '.join(values[r * cols:(r + 1) * cols])}]"
        for r in range(rows))
    return [f"{indent}return numpy.array([\n{body}])"]
