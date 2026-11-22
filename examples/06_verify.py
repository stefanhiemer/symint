"""Checking results: antiderivatives by term grouping, definite
integrals against quadrature."""

import sympy as sp

from symintegral import check_antiderivative, check_definite, integrate

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)

checks = {
    "x exp(-a x)": (-(a * x + 1) * sp.exp(-a * x) / a**2, x * sp.exp(-a * x)),
    "sin(x)^2": (x / 2 - sp.sin(2 * x) / 4, sp.sin(x)**2),
    "x e^x/(1+e^x)^2": (x * sp.exp(x) / (1 + sp.exp(x))
                        - sp.log(1 + sp.exp(x)),
                        x * sp.exp(x) / (1 + sp.exp(x))**2),
    "wrong": (x**2 * sp.exp(x), x * sp.exp(x)),
}
for name, (F, f) in checks.items():
    c = check_antiderivative(F, f, x)
    print(f"{name:18s} ok={c.ok!s:5s} decided by: {c.method}")

I = sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo))
print("definite:", check_definite(integrate(I, backends=()), I))
