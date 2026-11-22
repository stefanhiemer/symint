"""CAS back ends through Sage: Maxima, FriCAS, assumptions, sign questions."""

import pytest
import sympy as sp

from symintegral import integrate

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
b = sp.Symbol("b", real=True)


def _no_integral(expr):
    return not expr.has(sp.Integral)


def test_maxima_gaussian_cosine():
    log = []
    res = integrate(sp.Integral(sp.exp(-x**2) * sp.cos(x), (x, 0, sp.oo)),
                    backends=("maxima",), rules=(), log=log)
    assert sp.simplify(res - sp.sqrt(sp.pi) * sp.exp(-sp.Rational(1, 4)) / 2) == 0
    assert log[0]["method"] == "maxima"


def test_fricas_rational():
    res = integrate(sp.Integral(1 / (1 + x**3), (x, 0, 1)),
                    backends=("fricas",), rules=())
    ref = sp.log(2) / 3 + sp.sqrt(3) * sp.pi / 9
    assert abs(sp.N(res - ref)) < 1e-12


def test_fricas_antiderivative():
    res = integrate(sp.Integral(1 / (1 + x**2), (x, 0, 1)),
                    backends=("fricas_antiderivative",), rules=())
    assert sp.simplify(res - sp.pi / 4) == 0


@pytest.mark.parametrize("mode", ["full", "radical"])
def test_simplify_modes(mode):
    res = integrate(sp.Integral(1 / (1 + x**2), (x, 0, 1)),
                    backends=("maxima",), rules=(), simplify=mode)
    assert sp.simplify(res - sp.pi / 4) == 0


def test_unknown_simplify_mode_is_logged_and_skipped():
    # a failing preparation is logged; the step runs on the raw integrand
    log = []
    res = integrate(sp.Integral(1 / (1 + x**2), (x, 0, 1)),
                    backends=("maxima",), rules=(), simplify="nonsense",
                    log=log)
    assert sp.simplify(res - sp.pi / 4) == 0
    assert "ValueError" in log[0]["errors"]["prepare"]


def test_sign_question_answered_by_case_split():
    # Maxima asks about the sign / integrality of b; symintegral splits cases
    res = integrate(sp.Integral(x**b * sp.exp(-x), (x, 0, 1)),
                    backends=("maxima",), rules=())
    assert _no_integral(res)
    val = res.subs(b, sp.Rational(1, 2)).evalf()
    ref = sp.Integral(sp.sqrt(x) * sp.exp(-x), (x, 0, 1)).evalf()
    assert abs(val - ref) < 1e-10


def test_assumptions_restored_on_result():
    res = integrate(sp.Integral(sp.exp(-a * x), (x, 0, sp.oo)),
                    backends=("maxima",), rules=())
    assert res == 1 / a
    assert all(s.is_positive for s in res.free_symbols)


def test_limit_assumption_passed_to_sage():
    # needs y >= 1 (from the outer range) to drop an Abs / sign case
    y = sp.Symbol("y", real=True)
    res = integrate(sp.Integral(1 / (x**2 + y**2), (x, 0, sp.oo), (y, 1, 2)),
                    backends=("maxima",), rules=(), separate=False)
    assert _no_integral(res)
    assert abs(sp.N(res) - sp.N(sp.pi / 2 * sp.log(2))) < 1e-12


def test_failing_backend_leaves_integral():
    expr = sp.Integral(sp.exp(-x**4 - sp.sin(x)), (x, 0, 1))
    log = []
    res = integrate(expr, backends=("maxima",), rules=(), log=log)
    assert res.has(sp.Integral)
    assert log[-1]["method"] is None


def test_to_from_sage_roundtrip():
    from symintegral import from_sage, to_sage
    e = sp.exp(-a * x**2) + sp.sin(x)
    assert from_sage(to_sage(e), [a, x]) == e


def test_sign_split_cases_agree():
    # Maxima asks for the sign of b; b > 0 and b < 0 give the same result
    res = integrate(sp.Integral(sp.exp(-x**2) * sp.cos(b * x), (x, 0, sp.oo)),
                    backends=("maxima",), rules=())
    assert sp.simplify(res - sp.sqrt(sp.pi) * sp.exp(-b**2 / 4) / 2) == 0


def test_sign_split_cases_disagree():
    # log(b + 1) vs log(-b - 1): the result depends on the sign -> refused
    log = []
    res = integrate(sp.Integral(x / (x**2 + b), (x, 0, 1)),
                    backends=("maxima",), rules=(), log=log)
    assert res.has(sp.Integral)
    assert "depends on the sign" in log[0]["errors"]["maxima"]


def test_unanswerable_question_passed_on():
    # "Is b zero or nonzero?" is no sign question symintegral answers
    log = []
    res = integrate(sp.Integral(1 / (x**2 + b**2), (x, 0, sp.oo)),
                    backends=("maxima",), rules=(), log=log)
    assert res.has(sp.Integral)
    assert "zero or nonzero" in log[0]["errors"]["maxima"]


def test_sympy_gets_unsimplified_integrand():
    # Sage's pre-simplification defeats SymPy on the Beta integral; SymPy
    # must receive the original integrand when it follows Sage back ends
    y = sp.Symbol("y", positive=True)
    p, q = sp.symbols("p q", positive=True)
    log = []
    res = integrate(sp.Integral(y**(p - 1) * (1 - y)**(q - 1), (y, 0, 1)),
                    backends=("maxima", "fricas", "sympy"), log=log)
    assert log[0]["method"] == "sympy"
    assert sp.simplify(sp.gammasimp(res - sp.beta(p, q).rewrite(sp.gamma))) == 0
