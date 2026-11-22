"""
Options and state shared by the steps of one `integrate` call.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import sage as sage_mod
from ..rules import Rule
from .timeout import run_with_timeout

LogEntry = Dict[str, Any]
Limits = List[Tuple]


@dataclass
class Context:
    """
    Options and state of one `integrate` call.

    Attributes
    ----------
    session : SageSession
        All symbols of the integral (``session.symbols``, from which
        assumptions are restored after Sage conversions) and all limits;
        Sage assumptions are made only when a CAS step needs them.
    backends, simplify, complete_square, limit_algorithm, step_timeout,
    rules, log
        As in `integrate`.
    """

    session: sage_mod.SageSession
    backends: Tuple[str, ...]
    simplify: Optional[str]
    complete_square: bool
    limit_algorithm: str
    step_timeout: Optional[float]
    rules: Optional[Sequence[Rule]]
    log: Optional[List[LogEntry]]

    @property
    def uses_sage(self) -> bool:
        """Whether any of the back ends goes through Sage."""
        return any(a != "sympy" for a in self.backends)

    def timed(self,
              fn: Callable[[], Any]) -> Any:
        """
        Run `fn` under the step timeout.

        Parameters
        ----------
        fn : callable
            Function without arguments.

        Returns
        -------
        Any
            Result of `fn`; raises StepTimeout, see `run_with_timeout`.
        """
        return run_with_timeout(fn, self.step_timeout)

    def record(self,
               entry: LogEntry) -> None:
        """
        Append a step to the log, if one was given.

        Parameters
        ----------
        entry : dict
            Log entry, see `integrate`.

        Returns
        -------
        None
        """
        if self.log is not None:
            self.log.append(entry)
