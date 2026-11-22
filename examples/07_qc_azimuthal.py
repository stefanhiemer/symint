"""Quantum-chemistry rules: import symintegral.qc to enable them."""

import sympy as sp

import symintegral.qc  # noqa: F401  (registers the "azimuthal" rule)
from symintegral import RULES, integrate

phi = sp.Symbol("phi", real=True)
k, m = sp.symbols("k m", positive=True)

print("rules:", [r.name for r in RULES])
I = sp.Integral(sp.exp(k * sp.cos(phi) + m * sp.sin(phi)), (phi, 0, 2 * sp.pi))
print(I, "=", integrate(I, backends=()))
