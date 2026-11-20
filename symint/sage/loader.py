"""
Lazy loading of Sage (passagemath).

FriCAS is started through a launcher that needs passagemath's bundled
`ecl` on PATH; `ensure_ecl_on_path` adds it before Sage is imported.
"""

import os
import sys
import sysconfig
from typing import Any


def ensure_ecl_on_path() -> None:
    """
    Put passagemath's bundled binaries on PATH.

    Prepends ``sage_wheels/bin`` (ecl, maxima, fricas) of the first
    site-packages directory that has it, unless it is already on PATH.
    The FriCAS launcher needs ``ecl`` there.

    Returns
    -------
    None
    """
    paths = sysconfig.get_paths()
    for base in dict.fromkeys([paths["platlib"], paths["purelib"],
                               *sys.path]):
        bindir = os.path.join(base or ".", "sage_wheels", "bin")
        if os.path.isfile(os.path.join(bindir, "ecl")):
            path = os.environ.get("PATH", "").split(os.pathsep)
            if bindir not in path:
                os.environ["PATH"] = os.pathsep.join([bindir, *path])
            return


def sage() -> Any:
    """
    Import the Sage symbolics namespace.

    Returns
    -------
    module
        ``sage.all__sagemath_symbolics`` (imported on first call).

    Raises
    ------
    ImportError
        If passagemath is not installed.
    """
    ensure_ecl_on_path()
    try:
        import sage.all__sagemath_symbolics as sage_all
    except ImportError as err:
        raise ImportError("CAS integration needs passagemath; install "
                          "passagemath-symbolics[maxima,fricas] and passagemath-repl") from err
    return sage_all


def sage_loaded() -> bool:
    """
    Whether Sage has been imported already.

    Returns
    -------
    bool
        True once `sage` (or anything else) imported the Sage symbolics;
        decides which timeout mechanism `run_with_timeout` uses.
    """
    return "sage.all__sagemath_symbolics" in sys.modules
