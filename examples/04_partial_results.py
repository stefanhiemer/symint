"""Partial results: what symintegral cannot do stays an Integral, and the
generated NumPy code evaluates the rest by quadrature."""

import numpy as np
import sympy as sp
from scipy.integrate import dblquad

from symintegral import compile_function, integrate, numpy_function

x, y = sp.symbols("x y", real=True)
a = sp.Symbol("a", positive=True)

# y separates and is done exactly; exp(-a x^4) has no rule
I = sp.Integral(sp.exp(-a * x**4) * (y**2 + x * y), (x, 0, 1), (y, 0, 1))
res = integrate(I, backends=())
print("partial result:", res)

f = compile_function(numpy_function("f", res, [a]))
ref = dblquad(lambda yy, xx: np.exp(-2.0 * xx**4) * (yy**2 + xx * yy),
              0, 1, 0, 1)[0]
print("generated code:", f(2.0), " scipy dblquad:", ref)
