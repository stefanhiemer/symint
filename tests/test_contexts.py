import sympy as sp

from symintegral import RULES, integrate


def test_qc_registers_azimuthal_once():
    import importlib

    import symintegral.qc
    importlib.reload(symintegral.qc)
    assert [r.name for r in RULES].count("azimuthal") == 1


def test_qc_azimuthal_elliptic():
    import symintegral.qc  # noqa: F401
    phi = sp.Symbol("phi", real=True)
    A = sp.Symbol("A", positive=True)
    res = integrate(sp.Integral(1 / sp.sqrt(A + sp.cos(phi)),
                                (phi, 0, 2 * sp.pi)), backends=())
    assert not res.has(sp.Integral)
    val = res.subs(A, 3).evalf()
    ref = sp.Integral(1 / sp.sqrt(3 + sp.cos(phi)), (phi, 0, 2 * sp.pi)).evalf()
    assert abs(val - ref) < 1e-10
