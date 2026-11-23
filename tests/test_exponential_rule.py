"""The "gaussian" rule with linear exponents: P(x) exp(B x + C)."""

import pytest
import sympy as sp

from symintegral import gaussian_rule, integrate

x = sp.Symbol("x", real=True)
a, k = sp.symbols("a k", positive=True)
b = sp.Symbol("b", real=True)
n = sp.Symbol("n", nonnegative=True)
oo = sp.oo


def _close(value, integrand, lo, hi, params):
    """value equals the numerical integral at the given parameter values."""
    got = complex(sp.N(sp.sympify(value).subs(params), 20))
    want = complex(sp.N(sp.Integral(integrand.subs(params), (x, lo, hi)), 20))
    return abs(got - want) <= 1e-10 * max(1.0, abs(want))


@pytest.mark.parametrize("integrand, lo, hi", [
    (x**3 * sp.exp(-k * x), 0, oo),                      # Slater, integer n
    ((x**2 - 3 * x + 1) * sp.exp(-k * x + 2), 0, oo),    # polynomial, constant C
    (x**2 * sp.exp(-k * x), 1, oo),                      # shifted half-line
    (x * sp.exp(-k * x), -2, oo),                        # half-line through 0
    ((x + 1)**2 * sp.exp(k * x), -oo, 1),                # mirrored half-line
    (x**2 * sp.exp(b * x), 0, 1),                        # finite, any sign of B
    (x**3 * sp.exp(-2 * x), -1, 3),                      # finite, numeric B
    (sp.exp(-k * x), sp.Rational(1, 2), 2),              # finite, P = 1
])
def test_linear_exponent_numerically(integrand, lo, hi):
    value = gaussian_rule(integrand, x, (lo, hi))
    assert value is not None and not value.has(sp.Integral)
    for params in ({k: sp.Rational(7, 5), b: sp.Rational(-3, 4)},
                   {k: sp.Rational(1, 3), b: sp.Rational(5, 2)}):
        assert _close(value, integrand, lo, hi, params)


def test_slater_symbolic_power():
    value = gaussian_rule(x**n * sp.exp(-k * x), x, (0, oo))
    assert sp.simplify(value - sp.gamma(n + 1) / k**(n + 1)) == 0


def test_slater_symbolic_power_times_polynomial():
    integrand = x**n * (x + 2) * sp.exp(-k * x)
    value = gaussian_rule(integrand, x, (0, oo))
    params = {n: sp.Rational(3, 2), k: sp.Rational(4, 3)}
    assert _close(value, integrand, 0, oo, params)


def test_fractional_power():
    # int_0^oo x^(-1/2) exp(-k x) dx = sqrt(pi/k)
    value = gaussian_rule(sp.exp(-k * x) / sp.sqrt(x), x, (0, oo))
    assert sp.simplify(value - sp.sqrt(sp.pi / k)) == 0


@pytest.mark.parametrize("integrand, lo, hi", [
    (x * sp.exp(-b * x), 0, oo),          # sign of the rate not provable
    (x * sp.exp(k * x), 0, oo),           # growing exponential
    (sp.exp(-k * x), -oo, oo),            # whole line diverges
    (sp.exp(-k * x) / x, 0, oo),          # x^(-1): divergent at 0
    (x**n * sp.exp(-k * x), 1, oo),       # symbolic power off [0, oo)
    (x**n * sp.exp(-k * x), 0, 1),        # symbolic power, finite range
    (sp.exp(-k * x) * sp.sin(x), 0, oo),  # not polynomial times exponential
])
def test_declines(integrand, lo, hi):
    assert gaussian_rule(integrand, x, (lo, hi)) is None


def test_gaussian_with_power_still_declines():
    # the extra var^s factor is only for linear exponents
    assert gaussian_rule(x**n * sp.exp(-k * x**2), x, (0, oo)) is None


def test_rules_only_integrate():
    # the hydrogen-like 1s normalisation, rules only, no CAS
    r = sp.Symbol("r", positive=True)
    zeta = sp.Symbol("zeta", positive=True)
    psi2 = zeta**3 / sp.pi * sp.exp(-2 * zeta * r)
    integral = sp.Integral(psi2 * r**2 * 4 * sp.pi, (r, 0, oo))
    log = []
    assert integrate(integral, backends=(), log=log) == 1
    assert log[0]["method"] == "gaussian"
