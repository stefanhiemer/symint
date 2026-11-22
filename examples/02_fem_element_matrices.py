"""Element matrices on the reference triangle, then NumPy code for them.

The integrals are written in general form (as e.g. topoptlab would) and
handed to symintegral; `simplex_integral` Duffy-collapses the triangle onto
the unit square, so the polynomial rule and separation apply.
"""

import numpy as np
import sympy as sp

from symintegral import (compile_function, integrate, numpy_function,
                    simplex_integral)

x, y = sp.symbols("x y", real=True)

# P1 and P2 Lagrange basis on the reference triangle (0,0), (1,0), (0,1)
l0, l1, l2 = 1 - x - y, x, y
P1 = [l0, l1, l2]
P2 = [l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
      4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0]


def mass(basis, coeff=1):
    return sp.Matrix(len(basis), len(basis), lambda i, j: integrate(
        simplex_integral(coeff * basis[i] * basis[j], (x, y)),
        backends=()))


M1 = mass(P1)
print("P1 mass matrix * 24 =", (24 * M1).tolist())  # [[2,1,1],[1,2,1],[1,1,2]]
M2 = mass(P2)
print("P2 mass matrix * 360 =", (360 * M2).tolist())

# variable coefficient c(x, y) = c0 + c1 x + c2 y, passed as an array
c = sp.symbols("c0 c1 c2")
Mc = mass(P1, c[0] + c[1] * x + c[2] * y)
src = numpy_function("p1_mass", Mc, [("c", list(c))],
                     docstring="P1 mass matrix with linear coefficient.")
print(src)
f = compile_function(src)
print(f(np.array([1.0, 0.0, 0.0])) * 24)
