"""
codegen.py -- NumPy/SciPy source code from symint results.

`numpy_function` turns expressions (scalars, matrices, lists) into the
source of a Python function with common-subexpression elimination.
Unevaluated SymPy Integrals left in a partial result become calls to
``scipy.integrate.nquad`` over just the remaining variables, so a
reduced integral (e.g. a Boys-like 1-D remainder) is usable right away.
`module_source` assembles several functions into one module and
`compile_function` executes a source string and returns the function.
Special functions map to scipy.special; the `special_funcs` argument of
`numpy_function` lists the mappings added to those SymPy's SciPyPrinter
already knows.
"""

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import sympy as sp
from sympy import Integral, Symbol
from sympy.printing.numpy import SciPyPrinter

ArgSpec = Union[Symbol, Tuple[str, Any]]

HEADER = ("import numpy\n"
          "import scipy.special\n"
          "import scipy.integrate\n")


def _one_argument(*args: Any) -> bool:
    """
    Whether a function call has one argument (printer condition).

    Parameters
    ----------
    *args
        Arguments of the SymPy function being printed.

    Returns
    -------
    bool
        True for exactly one argument, e.g. the complete elliptic_e(m).
    """
    return len(args) == 1


def _two_arguments(*args: Any) -> bool:
    """
    Whether a function call has two arguments (printer condition).

    Parameters
    ----------
    *args
        Arguments of the SymPy function being printed.

    Returns
    -------
    bool
        True for exactly two arguments, e.g. the incomplete
        elliptic_e(z, m).
    """
    return len(args) == 2


def _flatten(exprs: Any) -> Tuple[List[sp.Expr], Tuple[str, Any]]:
    """
    Flatten the output of a generated function into a list.

    Parameters
    ----------
    exprs : sp.Expr, sp.Matrix, list, tuple or symfem Function
        What the generated function returns. Objects with an
        ``as_sympy`` method (symfem functions) are converted to a Matrix.

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
        return _flatten(sp.Matrix(exprs.as_sympy()))  # symfem functions
    if isinstance(exprs, (list, tuple)):
        return [sp.sympify(e) for e in exprs], ("list", len(exprs))
    return [sp.sympify(exprs)], ("scalar", None)


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
    """
    variables = [lim[0] for lim in integral.limits]
    fargs = ", ".join(printer.doprint(v) for v in variables)
    ranges = []
    for i, (_, lo, hi) in enumerate(integral.limits):
        outer = ", ".join(printer.doprint(w) for w in variables[i + 1:])
        ranges.append(f"lambda {outer + ', ' if outer else ''}*_: "
                      f"({printer.doprint(sp.sympify(lo))}, "
                      f"{printer.doprint(sp.sympify(hi))})")
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


def numpy_function(name: str,
                   exprs: Any,
                   args: Sequence[ArgSpec],
                   cse: bool = True,
                   docstring: str = None,
                   indent: str = "    ",
                   special_funcs: Dict[str, Any] = {
                       "elliptic_k": "scipy.special.ellipk",
                       "elliptic_e": [
                           (_one_argument, "scipy.special.ellipe"),
                           (_two_arguments, "scipy.special.ellipeinc")],
                       "elliptic_f": "scipy.special.ellipkinc",
                       "erfi": "scipy.special.erfi"}) -> str:
    """
    Source code of a NumPy function evaluating `exprs`.

    Parameters
    ----------
    name : str
        Function name.
    exprs : sp.Expr, sp.Matrix, list, or symfem Function
        What to return: a scalar, a numpy array of the matrix's shape, or
        a list.
    args : sequence
        Function arguments, each a Symbol, or ``(argname, symbols)`` with
        `symbols` a sympy Matrix / nested list of Symbols unpacked from
        an array argument (e.g. ``("c", Matrix([[c11, c12], ...]))``
        gives ``c11 = c[0, 0]`` ..., like topoptlab's ``matrices``).
    cse : bool, optional
        Common-subexpression elimination across all outputs.
    docstring : str, optional
        Docstring of the generated function.
    indent : str, optional
        Indentation of the function body.
    special_funcs : dict, optional
        SymPy functions SciPyPrinter does not know, mapped to
        scipy.special (the printer's ``user_functions``; a value may be a
        list of (condition on the arguments, name)). Elliptic integrals
        use the parameter convention m = k^2 in both libraries:
        elliptic_e(z, m) <-> ellipeinc(phi, m), likewise for F.

    Returns
    -------
    str
        Source of one function, without imports: combine with
        `module_source` (or `HEADER`) or run it with `compile_function`.
        Unevaluated Integrals in `exprs` are computed with
        ``scipy.integrate.nquad`` inside the function.

    Raises
    ------
    sympy.printing.codeprinter.PrintMethodNotImplementedError
        If `exprs` contains a function with no NumPy/SciPy counterpart
        (extend `special_funcs` defaults to add one).

    Examples
    --------
    >>> import sympy as sp
    >>> a, l1, l2 = sp.symbols("a l1 l2")
    >>> print(numpy_function("f", sp.Matrix([[l1 * sp.exp(a), l2]]),
    ...                      [a, ("l", [l1, l2])]))
    def f(a, l):
        l1 = l[0]
        l2 = l[1]
        return numpy.array([
            [l1*numpy.exp(a), l2]])
    <BLANKLINE>
    """
    # printer with fully qualified names; strict, so that unsupported
    # functions raise instead of producing code that fails at run time
    printer = SciPyPrinter({"fully_qualified_modules": True,
                            "strict": True,
                            "user_functions": special_funcs})
    flat, (kind, shape) = _flatten(exprs)
    # function signature: scalar Symbols are passed as they are; array
    # arguments (argname, symbols) are passed under argname and unpacked
    # into their entries (c11 = c[0, 0], ...) at the top of the body
    signature, unpack = _signature(args, printer, indent)
    # define function
    lines = [f"def {name}({', '.join(signature)}):"]
    # add doc string
    if docstring:
        lines.append(f'{indent}"""{docstring}"""')
    lines += unpack
    flat, integral_lines = _integral_functions(flat, printer, indent)
    lines += integral_lines
    # common substring extraction
    if cse:
        replacements, flat = sp.cse(flat,
                                    symbols=sp.numbered_symbols("_x"))
        lines += [f"{indent}{printer.doprint(s)} = {printer.doprint(e)}"
                  for s, e in replacements]
    values = [printer.doprint(e) for e in flat]
    lines += _return_lines(values, kind, shape, indent)
    return "\n".join(lines) + "\n"


def module_source(functions: Sequence[str],
                  header: str = HEADER) -> str:
    """
    Join generated functions into the source of one module.

    Parameters
    ----------
    functions : sequence of str
        Function sources, e.g. from `numpy_function`.
    header : str, optional
        Module preamble; the default imports numpy, scipy.special and
        scipy.integrate, which generated code refers to by full name.

    Returns
    -------
    str
        Module source, ready to be written to a ``.py`` file.
    """
    return header + "\n\n" + "\n\n".join(functions)


def compile_function(source: str,
                     name: str = None) -> Callable:
    """
    Execute generated source and return one of its functions.

    Parameters
    ----------
    source : str
        Source of one or more functions, without imports (`HEADER` is
        prepended). Executed with ``exec``: pass only source you
        generated.
    name : str, optional
        Function to return; default is the first one defined in `source`.

    Returns
    -------
    callable
        The compiled function.
    """
    namespace: Dict[str, Any] = {}
    exec(HEADER + source, namespace)
    if name is None:
        name = source.split("def ", 1)[1].split("(", 1)[0]
    return namespace[name]
