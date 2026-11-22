"""
`numpy_function`, `module_source` and `compile_function`.
"""

from typing import Any, Callable, Dict, Sequence

import sympy as sp
from sympy.printing.numpy import SciPyPrinter

from .layout import ArgSpec, _flatten, _return_lines, _signature
from .printing import HEADER, SPECIAL_FUNCS
from .quadrature import _integral_functions


def numpy_function(name: str,
                   exprs: Any,
                   args: Sequence[ArgSpec],
                   cse: bool = True,
                   docstring: str = None,
                   indent: str = "    ",
                   special_funcs: Dict[str, Any] = SPECIAL_FUNCS) -> str:
    """
    Generate code for a function using Numpy to evaluate `exprs`.

    Parameters
    ----------
    name : str
        Function name.
    exprs : sp.Expr, sp.Matrix, list, or object with ``as_sympy``
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
        (extend `SPECIAL_FUNCS` to add one).

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
    # printer with names; strict (unsupported functions raise error)
    printer = SciPyPrinter({"fully_qualified_modules": True,
                            "strict": True,
                            "user_functions": special_funcs})
    flat, (kind, shape) = _flatten(exprs)
    # generate function signature
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
