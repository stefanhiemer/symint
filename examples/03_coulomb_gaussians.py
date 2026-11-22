"""Coulomb interaction of two Gaussians via the integral transform.

1/|r1 - r2| couples all six variables; transform="gaussian" replaces it
by (2/sqrt(pi)) int_0^oo exp(-t^2 |r1 - r2|^2) dt, after which every
spatial integral is Gaussian. The leftover t-integral is mapped onto
u in [0, 1] (the Boys-function form).
"""

import sympy as sp

from symintegral import integrate

a, b, R = sp.symbols("a b R", positive=True)
x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2", real=True)
r12 = sp.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
limits = [(v, -sp.oo, sp.oo) for v in (x1, y1, z1, x2, y2, z2)]

# both Gaussians at the origin
I0 = sp.Integral(sp.exp(-a * (x1**2 + y1**2 + z1**2))
                 * sp.exp(-b * (x2**2 + y2**2 + z2**2)) / r12, *limits)
log = []
res0 = integrate(I0, transform="gaussian", backends=("sympy",), log=log)
print("centred:", sp.simplify(res0))
print("steps:  ", [e["method"] for e in log])

# second Gaussian moved by R along z: the result involves erf (Boys F_0)
IR = sp.Integral(sp.exp(-a * (x1**2 + y1**2 + z1**2))
                 * sp.exp(-b * (x2**2 + y2**2 + (z2 - R)**2)) / r12, *limits)
resR = integrate(IR, transform="gaussian", backends=("sympy",))
print("shifted:", sp.simplify(resR))

# R -> oo: two point charges, (pi/a)^(3/2) (pi/b)^(3/2) / R
print("R -> oo:", sp.simplify(sp.limit(resR * R, R, sp.oo)))

