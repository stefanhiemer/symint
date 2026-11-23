"""A user rule: Slater-type radial moments int_0^oo x^n exp(-c x) dx.

The built-in "gaussian" rule covers these integrals too; this example
shows how such a rule is written and registered. Registered with
first=True, it is tried before the built-in rules, so the log names it.
"""

import sympy as sp

from symintegral import Rule, integrate, register_rule, unregister_rule

x = sp.Symbol("x", positive=True)
c = sp.Symbol("c", positive=True)
n = sp.Symbol("n", nonnegative=True)


def slater_rule(expr, var, bounds):
    if list(bounds) != [0, sp.oo]:
        return None
    total = 0
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, power, rate = sp.Integer(1), sp.Integer(0), None
        for f in sp.Mul.make_args(sp.powsimp(term, combine="exp")):
            base, exp_ = f.as_base_exp()
            if isinstance(f, sp.exp):
                lin = sp.Poly(f.args[0], var) if f.args[0].is_polynomial(var) \
                    else None
                if lin is None or lin.degree() != 1 or lin.coeff_monomial(1):
                    return None
                rate = -lin.coeff_monomial(var)
            elif base == var:
                power += exp_
            elif not f.has(var):
                coeff *= f
            else:
                return None
        if rate is None or rate.is_positive is not True:
            return None
        total += coeff * sp.gamma(power + 1) / rate**(power + 1)
    return total


register_rule(Rule("slater", lambda bounds: len(bounds) == 2, slater_rule),
              first=True)

for f in (x**n * sp.exp(-c * x), x**2 * sp.exp(-c * x) * (1 + x)):
    log = []
    res = integrate(sp.Integral(f, (x, 0, sp.oo)), backends=(), log=log)
    print(f, "->", sp.simplify(res), " by", log[0]["method"])

unregister_rule("slater")
