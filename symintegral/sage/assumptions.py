"""
SymPy assumptions as Sage ``assume`` facts.

SymPy assumptions on the free symbols (positive, nonnegative, negative,
real, integer) and the integration ranges (e.g. xi >= 1) are passed to
Sage via ``assume`` for the duration of a `SageSession`, and forgotten
afterwards, since Sage's assumptions are global.
"""

from typing import List, Sequence, Tuple

import sympy as sp
from sympy import Symbol

from .convert import to_sage
from .loader import sage


def symbol_facts(symbols: Sequence[Symbol]) -> List[Tuple]:
    """
    SymPy symbol assumptions as Sage ``assume`` arguments.

    Parameters
    ----------
    symbols : sequence of Symbol
        Symbols whose assumptions are translated (integer or real; then
        positive, nonnegative or negative).

    Returns
    -------
    list of tuple
        Argument tuples for ``assume``, e.g. ``(a, "real")``, ``(a > 0,)``.
    """
    SR = sage().SR
    facts: List[Tuple] = []
    for s in symbols:
        v = SR(s)
        if s.is_integer:
            facts.append((v, "integer"))
        elif s.is_real:
            facts.append((v, "real"))
        if s.is_positive:
            facts.append((v > 0,))
        elif s.is_nonnegative:
            facts.append((v >= 0,))
        elif s.is_negative:
            facts.append((v < 0,))
    return facts


def limit_facts(limits: Sequence[Tuple]) -> List[Tuple]:
    """
    Integration ranges as Sage ``assume`` arguments.

    Parameters
    ----------
    limits : sequence of (Symbol, lower, upper)
        Integration limits.

    Returns
    -------
    list of tuple
        ``(v >= lower,)`` and ``(v <= upper,)`` for every finite bound
        free of integration variables, e.g. (xi, 1, oo) -> xi >= 1.
    """
    SR = sage().SR
    variables = {lim[0] for lim in limits}
    facts: List[Tuple] = []
    for var, *bounds in limits:
        if len(bounds) != 2:
            continue
        v = SR(var)
        lo, hi = (sp.sympify(bd) for bd in bounds)
        if lo.is_finite and not (lo.free_symbols & variables):
            facts.append((v >= to_sage(lo),))
        if hi.is_finite and not (hi.free_symbols & variables):
            facts.append((v <= to_sage(hi),))
    return facts


class SageSession:
    """
    Sage assumptions for one integration.

    The assumptions are made only when a CAS step first needs them
    (`ensure`), so that integrations done entirely by rules or SymPy
    never import Sage, and are all forgotten on `close`. Further symbols
    and limits (e.g. substitution variables) can be added at any time
    with `extend`.

    Parameters
    ----------
    symbols : sequence of Symbol, optional
        Symbols whose assumptions are passed to Sage.
    limits : sequence of (Symbol, lower, upper), optional
        Integration ranges passed to Sage as inequalities.

    Attributes
    ----------
    symbols : list of Symbol
        Symbols of the integration, kept even while no assumptions are
        made; the engine also restores assumptions after conversions
        (`from_sage`) from them.
    limits : list of (Symbol, lower, upper)
        Integration ranges.
    facts : list of tuple
        Assumptions made so far (what `close` forgets).
    active : bool
        Whether `ensure` has been called since the last `close`.
    """

    def __init__(self,
                 symbols: Sequence[Symbol] = (),
                 limits: Sequence[Tuple] = ()) -> None:
        self.symbols = list(symbols)
        self.limits = list(limits)
        self.facts: List[Tuple] = []
        self.active = False

    def _assume(self,
                facts: Sequence[Tuple]) -> None:
        """
        Make assumptions in Sage and remember them.

        Parameters
        ----------
        facts : sequence of tuple
            ``assume`` argument tuples. Facts that Sage rejects as
            redundant or inconsistent with existing ones are skipped.

        Returns
        -------
        None
        """
        sage_all = sage()
        for fact in facts:
            try:
                sage_all.assume(*fact)
                self.facts.append(fact)
            except ValueError:  # redundant/inconsistent with existing fact
                pass

    def ensure(self) -> None:
        """
        Make the session's assumptions, once.

        Returns
        -------
        None
        """
        if not self.active:
            self.active = True
            self._assume(symbol_facts(self.symbols)
                         + limit_facts(self.limits))

    def extend(self,
               symbols: Sequence[Symbol] = (),
               limits: Sequence[Tuple] = ()) -> None:
        """
        Add symbols and limits (assumed at once if the session is active).

        Parameters
        ----------
        symbols : sequence of Symbol, optional
            Further symbols with assumptions.
        limits : sequence of (Symbol, lower, upper), optional
            Further integration ranges.

        Returns
        -------
        None
        """
        self.symbols += [s for s in symbols if s not in self.symbols]
        self.limits += list(limits)
        if self.active:
            self._assume(symbol_facts(symbols) + limit_facts(limits))

    def close(self) -> None:
        """
        Forget all assumptions made by this session.

        Returns
        -------
        None
        """
        if not self.active:
            return
        sage_all = sage()
        for fact in reversed(self.facts):
            try:
                sage_all.forget(*fact)
            except ValueError:
                pass
        self.facts, self.active = [], False
