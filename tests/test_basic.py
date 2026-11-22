import pytest
import sympy as sp

import symintegral
from symintegral import (Rule, integrate, register_rule, simplex_integral,
                    unregister_rule)

x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
a = sp.Symbol("a", positive=True)


def test_version():
    assert isinstance(symintegral.__version__, str)


def test_gaussian_moment_rules_only():
    res = integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)),
                    backends=())
    assert sp.simplify(res - sp.sqrt(sp.pi) / (2 * a**sp.Rational(3, 2))) == 0


def test_simplex_moment():
    assert integrate(simplex_integral(x**2 * y, (x, y)),
                     backends=()) == sp.Rational(1, 60)


def test_polynomial_rule():
    assert integrate(sp.Integral(x**3 + 2 * x, (x, 0, 2)),
                     backends=()) == 8


def test_unevaluated_stays_integral():
    expr = sp.Integral(sp.exp(-a * x**4), (x, 0, 1))
    assert integrate(expr, backends=()) == expr


def test_custom_rule():
    g = sp.Function("g")

    def g_rule(expr, var, bounds):
        return sp.Integer(1) if expr == g(var) and bounds == [0, 1] else None

    register_rule(Rule("g", lambda bounds: True, g_rule), first=True)
    try:
        assert integrate(sp.Integral(3 * g(x), (x, 0, 1)), backends=()) == 3
    finally:
        unregister_rule("g")


def test_codegen_quadrature():
    pytest.importorskip("scipy")
    from symintegral import compile_function, numpy_function
    f = compile_function(numpy_function(
        "f", sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), [a]))
    assert float(f(1.0)) == pytest.approx(0.84483859, abs=1e-8)
