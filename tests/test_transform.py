"""Integral transforms of Q^(-nu/2) factors and the outer-variable step."""

import pytest
import sympy as sp

from symintegral import integrate

a, b = sp.symbols("a b", positive=True)
r1 = sp.symbols("x1 y1 z1", real=True)
r2 = sp.symbols("x2 y2 z2", real=True)
r12 = sp.sqrt(sum((u - v)**2 for u, v in zip(r1, r2)))
COULOMB = sp.Integral(sp.exp(-a * sum(u**2 for u in r1))
                      * sp.exp(-b * sum(v**2 for v in r2)) / r12,
                      *[(v, -sp.oo, sp.oo) for v in (*r1, *r2)])
EXACT = 2 * sp.pi**sp.Rational(5, 2) / (a * b * sp.sqrt(a + b))


@pytest.mark.parametrize("transform", ["gaussian", "laplace"])
def test_coulomb_two_gaussians(transform):
    log = []
    res = integrate(COULOMB, transform=transform, backends=("sympy",),
                    log=log)
    assert sp.simplify(res - EXACT) == 0
    assert [e["method"] for e in log].count("gaussian") == 6


def test_without_substitution_onto_unit_interval():
    res = integrate(COULOMB, transform="gaussian", substitute_outer=False,
                    backends=("sympy",))
    assert sp.simplify(res - EXACT) == 0


def test_shifted_centres_give_erf():
    # Gaussians at different centres: the outer u-integral is the Boys
    # function F_0(T) = int_0^1 exp(-T u^2) du, an erf
    R = sp.Symbol("R", positive=True)
    x2, y2, z2 = r2
    shifted = (sp.exp(-a * sum(u**2 for u in r1))
               * sp.exp(-b * (x2**2 + y2**2 + (z2 - R)**2)) / r12)
    res = integrate(sp.Integral(shifted, *COULOMB.limits),
                    transform="gaussian", backends=("sympy",))
    assert not res.has(sp.Integral)
    assert res.has(sp.erf)
    vals = {a: 1, b: 2, R: sp.Rational(3, 2)}
    p = sp.Rational(2, 3)                      # ab/(a+b)
    T = p * vals[R]**2
    ref = (2 * sp.pi**sp.Rational(5, 2) / (1 * 2 * sp.sqrt(3))
           * sp.sqrt(sp.pi) / (2 * sp.sqrt(T)) * sp.erf(sp.sqrt(T)))
    assert abs(sp.N(res.subs(vals) - ref)) < 1e-12


def test_one_dimensional_example_from_docs():
    x = sp.Symbol("x", real=True)
    res = integrate(sp.Integral(sp.exp(-x**2) / sp.sqrt(1 + x**2),
                                (x, -sp.oo, sp.oo)),
                    transform="gaussian", backends=())
    # outer integral has no rule: stays, but numerically right
    ref = sp.sqrt(sp.E) * sp.besselk(0, sp.Rational(1, 2))
    assert abs(sp.N(res) - sp.N(ref)) < 1e-8


def test_terms_without_coupling_factor_untouched():
    x = sp.Symbol("x", real=True)
    res = integrate(sp.Integral(sp.exp(-x**2) * (1 + 1 / sp.sqrt(1 + x**2)),
                                (x, -sp.oo, sp.oo)),
                    transform="gaussian", backends=())
    ref = sp.sqrt(sp.pi) + sp.sqrt(sp.E) * sp.besselk(0, sp.Rational(1, 2))
    assert abs(sp.N(res) - sp.N(ref)) < 1e-8


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        integrate(COULOMB, transform="nonsense")
