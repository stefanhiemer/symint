"""
The rule framework: `Rule`, the registry `RULES` and `apply_rules`.

A `Rule` integrates one variable over one range in closed form, or
declines (returns None) so that the next rule or the CAS takes over.
The engine consults `RULES` in order; add your own with `register_rule`.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Symbol

Bounds = Sequence[Any]


@dataclass
class Rule:
    """
    A closed-form integration rule.

    Attributes
    ----------
    name : str
        Shown in the engine's log.
    applies : callable(bounds) -> bool
        Cheap test on the integration range.
    evaluate : callable(expr, var, bounds) -> sp.Expr or None
        The definite integral of `expr` over `var` in `bounds`, or None
        if the rule does not apply to this integrand.
    """

    name: str
    applies: Callable[[Bounds], bool]
    evaluate: Callable[[sp.Expr, Symbol, Bounds], Optional[sp.Expr]]


RULES: List[Rule] = []


def register_rule(rule: Rule,
                  first: bool = False) -> Rule:
    """
    Add a rule to the registry.

    Parameters
    ----------
    rule : Rule
        The rule; a registered rule of the same name is replaced.
    first : bool, optional
        Insert at the front (tried before the built-in rules) instead of
        appending.

    Returns
    -------
    Rule
        `rule`, so that the call can be used as an expression.
    """
    unregister_rule(rule.name)
    if first:
        RULES.insert(0, rule)
    else:
        RULES.append(rule)
    return rule


def unregister_rule(name: str) -> None:
    """
    Remove a rule from the registry.

    Parameters
    ----------
    name : str
        Name of the rule; nothing happens if no rule has this name.

    Returns
    -------
    None
    """
    RULES[:] = [r for r in RULES if r.name != name]


def apply_rules(expr: sp.Expr,
                var: Symbol,
                bounds: Bounds,
                rules: Sequence[Rule] = None) -> Optional[Tuple[str, sp.Expr]]:
    """
    Integrate with the first rule that can.

    Parameters
    ----------
    expr : sp.Expr
        Integrand.
    var : Symbol
        Integration variable.
    bounds : sequence
        ``(lower, upper)``.
    rules : sequence of Rule, optional
        Rules to try in order; default the registry `RULES`.

    Returns
    -------
    (str, sp.Expr) or None
        Name of the successful rule and the integral, or None if every
        rule declined.
    """
    for rule in RULES if rules is None else rules:
        if rule.applies(bounds):
            result = rule.evaluate(expr, var, bounds)
            if result is not None:
                return rule.name, result
    return None
