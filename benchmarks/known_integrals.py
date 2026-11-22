"""
Run symintegral on integrals with known closed forms and report the outcome.

    python benchmarks/known_integrals.py [--timeout SECONDS] [--only NAME]

For each integral: "exact" (result simplifies to the known value),
"numeric" (equal at random parameter values only), "partial" (an
Integral remains), "WRONG" (a closed form that disagrees), or "error".
"""

import argparse
import random
import time

import sympy as sp

from symintegral import integrate

x, y, z, t = sp.symbols("x y z t", real=True)
a, b, c, p, q = sp.symbols("a b c p q", positive=True)
m = sp.Symbol("m", positive=True)  # used with m < 1 below
pi, oo, E = sp.pi, sp.oo, sp.E

# (name, integral, known value, parameter values for the numeric check)
CASES = [
    # --- Gaussians and relatives
    ("gauss moment x^4 shifted",
     sp.Integral(x**4 * sp.exp(-a * x**2 + b * x), (x, -oo, oo)),
     sp.sqrt(pi) * (12 * a**2 + 12 * a * b**2 + b**4)
     * sp.exp(b**2 / (4 * a)) / (16 * a**sp.Rational(9, 2)), {}),
    ("gauss * cos", sp.Integral(sp.exp(-x**2) * sp.cos(b * x), (x, 0, oo)),
     sp.sqrt(pi) / 2 * sp.exp(-b**2 / 4), {}),
    ("gauss with 1/x^2 term",
     sp.Integral(sp.exp(-a * x**2 - b / x**2), (x, 0, oo)),
     sp.sqrt(pi / a) / 2 * sp.exp(-2 * sp.sqrt(a * b)), {}),
    ("x erf(bx) gauss",
     sp.Integral(x * sp.exp(-a * x**2) * sp.erf(b * x), (x, -oo, oo)),
     b / (a * sp.sqrt(a + b**2)), {}),
    ("erfc on half line", sp.Integral(sp.erfc(x), (x, 0, oo)),
     1 / sp.sqrt(pi), {}),
    ("2-D coupled quadratic form",
     sp.Integral(sp.exp(-(x**2 + x * y + y**2)), (x, -oo, oo), (y, -oo, oo)),
     2 * pi / sp.sqrt(3), {}),
    ("3-D gauss with linear term",
     sp.Integral(sp.exp(-(x**2 + y**2 + z**2) - (x + y + z)),
                 (x, -oo, oo), (y, -oo, oo), (z, -oo, oo)),
     pi**sp.Rational(3, 2) * sp.exp(sp.Rational(3, 4)), {}),
    # --- Gamma / Beta / Laplace type
    ("Gamma, symbolic power",
     sp.Integral(x**p * sp.exp(-a * x), (x, 0, oo)),
     sp.gamma(p + 1) / a**(p + 1), {}),
    ("Gamma(1/2)", sp.Integral(sp.exp(-x) / sp.sqrt(x), (x, 0, oo)),
     sp.sqrt(pi), {}),
    ("Beta(p, q)",
     sp.Integral(x**(p - 1) * (1 - x)**(q - 1), (x, 0, 1)),
     sp.beta(p, q), {}),
    ("Laplace of sin", sp.Integral(sp.exp(-a * x) * sp.sin(b * x), (x, 0, oo)),
     b / (a**2 + b**2), {}),
    ("Laplace of J0", sp.Integral(sp.besselj(0, x) * sp.exp(-a * x),
                                  (x, 0, oo)),
     1 / sp.sqrt(1 + a**2), {}),
    ("Frullani", sp.Integral((sp.exp(-a * x) - sp.exp(-b * x)) / x,
                             (x, 0, oo)),
     sp.log(b / a), {}),
    ("(x^a - 1)/log x", sp.Integral((x**a - 1) / sp.log(x), (x, 0, 1)),
     sp.log(a + 1), {}),
    ("Euler gamma", sp.Integral(sp.exp(-x) * sp.log(x), (x, 0, oo)),
     -sp.EulerGamma, {}),
    # --- Bose/Fermi, zeta
    ("x/(e^x - 1)", sp.Integral(x / (sp.exp(x) - 1), (x, 0, oo)),
     pi**2 / 6, {}),
    ("x^3/(e^x - 1)", sp.Integral(x**3 / (sp.exp(x) - 1), (x, 0, oo)),
     pi**4 / 15, {}),
    ("1/(1 - xy) on unit square",
     sp.Integral(1 / (1 - x * y), (x, 0, 1), (y, 0, 1)), pi**2 / 6, {}),
    # --- oscillatory / Dirichlet
    ("Dirichlet sin x/x", sp.Integral(sp.sin(x) / x, (x, 0, oo)), pi / 2, {}),
    ("sin^2 x/x^2", sp.Integral(sp.sin(x)**2 / x**2, (x, 0, oo)), pi / 2, {}),
    ("cos x/(1 + x^2)", sp.Integral(sp.cos(x) / (1 + x**2), (x, 0, oo)),
     pi / (2 * E), {}),
    # --- rational / algebraic
    ("1/(1 + x^4)", sp.Integral(1 / (1 + x**4), (x, -oo, oo)),
     pi / sp.sqrt(2), {}),
    ("x^(-2/3)/(1 + x)",
     sp.Integral(x**sp.Rational(-2, 3) / (1 + x), (x, 0, oo)),
     2 * pi / sp.sqrt(3), {}),
    ("22/7 - pi", sp.Integral(x**4 * (1 - x)**4 / (1 + x**2), (x, 0, 1)),
     sp.Rational(22, 7) - pi, {}),
    ("quarter circle", sp.Integral(sp.sqrt(1 - x**2), (x, 0, 1)), pi / 4, {}),
    ("1/(2 + cos t) full period",
     sp.Integral(1 / (2 + sp.cos(t)), (t, 0, 2 * pi)), 2 * pi / sp.sqrt(3), {}),
    # --- logarithms
    ("log sin", sp.Integral(sp.log(sp.sin(x)), (x, 0, pi / 2)),
     -pi / 2 * sp.log(2), {}),
    ("log(1+x)/(1+x^2)", sp.Integral(sp.log(1 + x) / (1 + x**2), (x, 0, 1)),
     pi * sp.log(2) / 8, {}),
    ("log x log(1-x)", sp.Integral(sp.log(x) * sp.log(1 - x), (x, 0, 1)),
     2 - pi**2 / 6, {}),
    ("log x/(1 + x^2)", sp.Integral(sp.log(x) / (1 + x**2), (x, 0, oo)),
     sp.Integer(0), {}),
    ("Catalan", sp.Integral(sp.atan(x) / x, (x, 0, 1)), sp.Catalan, {}),
    ("x sin x/(1 + cos^2 x)",
     sp.Integral(x * sp.sin(x) / (1 + sp.cos(x)**2), (x, 0, pi)),
     pi**2 / 4, {}),
    # --- elliptic
    ("complete elliptic K",
     sp.Integral(1 / sp.sqrt(1 - m * sp.sin(t)**2), (t, 0, pi / 2)),
     sp.elliptic_k(m), {m: sp.Rational(1, 3)}),
]


def _numeric_equal(expr, known, fixed, samples=3, tol=1e-8):
    rng = random.Random(1)
    syms = (expr.free_symbols | known.free_symbols) - set(fixed)
    for _ in range(samples):
        vals = {s: sp.Rational(rng.randint(30, 170), 100) for s in syms}
        vals.update(fixed)
        try:
            d = complex(sp.N((expr - known).subs(vals), 25))
            k = complex(sp.N(known.subs(vals), 25))
        except (TypeError, ValueError):
            return False
        if not abs(d) <= tol * max(1.0, abs(k)):  # also rejects nan
            return False
    return True


def classify(res, known, fixed):
    if res.has(sp.Integral):
        return "partial"
    try:
        if sp.simplify(res - known) == 0:
            return "exact"
    except Exception:
        pass
    return "numeric" if _numeric_equal(res, known, fixed) else "WRONG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--only")
    args = ap.parse_args()
    counts = {}
    for name, integral, known, fixed in CASES:
        if args.only and args.only not in name:
            continue
        log = []
        t0 = time.time()
        try:
            res = integrate(integral, step_timeout=args.timeout, log=log)
            status = classify(res, known, fixed)
        except Exception as err:  # report, do not stop the run
            res, status = f"{type(err).__name__}: {err}", "error"
        used = sorted({str(e["method"]) for e in log})
        counts[status] = counts.get(status, 0) + 1
        print(f"{status:8s} {time.time() - t0:6.1f}s  {name:28s} "
              f"[{', '.join(used)}]", flush=True)
        if status in ("WRONG", "error"):
            print(f"         got {res}")
    print("summary:", counts)


if __name__ == "__main__":
    main()
