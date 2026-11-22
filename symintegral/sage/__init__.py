"""
SymPy <-> Sage (passagemath) bridge for symintegral.

Submodules
----------
loader
    Lazy import of Sage (`sage`, `sage_loaded`, `ensure_ecl_on_path`).
convert
    Expression conversion (`to_sage`, `from_sage`).
assumptions
    SymPy assumptions and integration ranges as Sage facts
    (`symbol_facts`, `limit_facts`, `SageSession`).
questions
    Case splits answering Maxima's sign/integer questions
    (`with_sign_cases`).
integration
    Single CAS integration steps (`integrate_step`, `simplify`,
    `has_unevaluated_integral`).

Sage is imported only when a CAS step is actually attempted, so
integrations done entirely by rules or SymPy never load it.
"""

from .assumptions import SageSession, limit_facts, symbol_facts
from .convert import from_sage, to_sage
from .integration import (ANTIDERIVATIVE, CAS_BACKENDS,
                          has_unevaluated_integral, integrate_step, simplify)
from .loader import ensure_ecl_on_path, sage, sage_loaded
from .questions import MAX_SIGN_CASES, with_sign_cases

__all__ = [
    "sage", "sage_loaded", "ensure_ecl_on_path",
    "to_sage", "from_sage",
    "symbol_facts", "limit_facts", "SageSession",
    "MAX_SIGN_CASES", "with_sign_cases",
    "CAS_BACKENDS", "ANTIDERIVATIVE", "simplify",
    "has_unevaluated_integral", "integrate_step",
]
