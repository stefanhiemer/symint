import sympy as sp

from symintegral import check_antiderivative, check_definite, integrate
from symintegral.verify import group_terms

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", positive=True)
b = sp.Symbol("b", real=True)


def test_group_terms_merges_split_exponentials():
    expr = x * sp.exp(-(x - b)**2) - x * sp.exp(-x**2) * sp.exp(2 * b * x) \
        * sp.exp(-b**2)
    assert group_terms(expr, x) == {}


def test_group_terms_merges_radical_powers():
    r = sp.sqrt(x**2 + 1)
    assert group_terms((x**2 + 1)**sp.Rational(3, 2) - (x**2 + 1) * r, x) == {}


def test_proven_by_grouping():
    F = sp.erf(sp.sqrt(a) * (x - b)) * sp.sqrt(sp.pi) / (2 * sp.sqrt(a))
    f = sp.exp(-a * (x - b)**2)
    assert check_antiderivative(F, f, x).method == "groups"


def test_transcendental_denominator():
    F = x * sp.exp(x) / (1 + sp.exp(x)) - sp.log(1 + sp.exp(x))
    f = x * sp.exp(x) / (1 + sp.exp(x))**2
    assert check_antiderivative(F, f, x).method == "groups"


def test_needs_simplify():
    F = x / 2 - sp.sin(2 * x) / 4
    c = check_antiderivative(F, sp.sin(x)**2, x)
    assert c.ok and c.method == "simplify"


def test_constant_of_integration_irrelevant():
    assert check_antiderivative(sp.atan(x) + a**2 + 7, 1 / (1 + x**2), x)


def test_wrong_antiderivative_rejected():
    c = check_antiderivative(x * sp.exp(x), x * sp.exp(x), x)
    assert not c.ok and c.method is None and c.residual != 0


def test_numeric_only():
    # piecewise-equivalent forms: only numerically zero for x > 0
    c = check_antiderivative(sp.log(x**2) / 2, 1 / x, x, simplify=False)
    assert c.ok


def test_check_definite_against_integrate():
    I = sp.Integral(x**4 * sp.exp(-a * x**2 + b * x), (x, -sp.oo, sp.oo))
    assert check_definite(integrate(I, backends=()), I)
    assert not check_definite(sp.sqrt(sp.pi / a), I)
