"""
Printer settings: the module header generated code relies on, and the
special functions SymPy's SciPyPrinter does not map by itself.
"""

from typing import Any, Dict


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


# SymPy functions SciPyPrinter does not know -> scipy.special; a value may
# be a list of (condition on the arguments, name)
SPECIAL_FUNCS: Dict[str, Any] = {
    "elliptic_k": "scipy.special.ellipk",
    "elliptic_e": [
        (_one_argument, "scipy.special.ellipe"),
        (_two_arguments, "scipy.special.ellipeinc")],
    "elliptic_f": "scipy.special.ellipkinc",
    "erfi": "scipy.special.erfi"}
