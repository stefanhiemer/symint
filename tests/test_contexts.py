import sympy as sp

from symint import RULES, integrate


def test_qc_registers_azimuthal_once():
    import importlib

    import symint.qc
    importlib.reload(symint.qc)
    assert [r.name for r in RULES].count("azimuthal") == 1


def test_qc_azimuthal_elliptic():
    import symint.qc  # noqa: F401
    phi = sp.Symbol("phi", real=True)
    A = sp.Symbol("A", positive=True)
    res = integrate(sp.Integral(1 / sp.sqrt(A + sp.cos(phi)),
                                (phi, 0, 2 * sp.pi)), algorithms=())
    assert not res.has(sp.Integral)
    val = res.subs(A, 3).evalf()
    ref = sp.Integral(1 / sp.sqrt(3 + sp.cos(phi)), (phi, 0, 2 * sp.pi)).evalf()
    assert abs(val - ref) < 1e-10
