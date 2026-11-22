"""Gaussian moments with the built-in closed-form rule (no CAS needed)."""

import sympy as sp

from symintegral import integrate

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
b = sp.Symbol("b", real=True)

# whole line, shifted Gaussian, polynomial prefactor
I1 = sp.Integral(x**4 * sp.exp(-a * x**2 + b * x), (x, -sp.oo, sp.oo))
print(I1, "=", sp.simplify(integrate(I1, backends=())))

# half line
I2 = sp.Integral(x**3 * sp.exp(-a * x**2), (x, 0, sp.oo))
print(I2, "=", integrate(I2, backends=()))

# finite interval: the Boys-type integral int_0^1 u^2 exp(-T u^2) du
T = sp.Symbol("T", positive=True)
u = sp.Symbol("u", positive=True)
I3 = sp.Integral(u**2 * sp.exp(-T * u**2), (u, 0, 1))
print(I3, "=", sp.simplify(integrate(I3, backends=())))

# which rule did what
log = []
integrate(sp.Integral(x**2 * sp.exp(-a * x**2) * sp.exp(-a * b**2),
                      (x, -sp.oo, sp.oo), (b, -sp.oo, sp.oo)),
          backends=(), log=log)
for entry in log:
    print(entry["var"], "->", entry["method"])
