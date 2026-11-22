"""Step timeouts, with setitimer (no Sage) and cysignals (Sage loaded)."""

import subprocess
import sys
import textwrap

import pytest

from symintegral import StepTimeout, run_with_timeout


def test_run_with_timeout_passes_result():
    assert run_with_timeout(lambda: 42, 1.0) == 42
    assert run_with_timeout(lambda: 42, None) == 42


def test_run_with_timeout_raises():
    def spin():
        while True:
            pass
    with pytest.raises(StepTimeout):
        run_with_timeout(spin, 0.2)


SLOW = textwrap.dedent("""
    import sys, time, sympy as sp
    from symintegral import integrate
    if PRELOAD:
        from symintegral.sage import sage
        sage()
    x = sp.Symbol("x", real=True)
    expr = sp.Integral(sp.exp(-x**4 - sp.sin(x)) / (1 + x**3), (x, 0, 1))
    log = []
    t = time.time()
    res = integrate(expr, backends=("sympy",), rules=(), step_timeout=1.0,
                    log=log)
    assert res.has(sp.Integral), res
    assert time.time() - t < 30
    assert any("timeout" in str(e).lower()
               for e in log[-1]["errors"].values()), log
    print("sage loaded:", "sage.all__sagemath_symbolics" in sys.modules)
""")


@pytest.mark.parametrize("sage_loaded", [False, True])
def test_integrate_step_timeout(sage_loaded):
    # fresh interpreter: whether Sage is loaded decides the mechanism
    # (setitimer without Sage, cysignals alarm with it)
    out = subprocess.run([sys.executable, "-c",
                          SLOW.replace("PRELOAD", repr(sage_loaded))],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert f"sage loaded: {sage_loaded}" in out.stdout


SAGE_LOADED_DURING_STEP = textwrap.dedent("""
    import sympy as sp
    from symintegral import integrate, run_with_timeout, StepTimeout
    from symintegral.sage import sage
    x = sp.Symbol("x", real=True)
    # the first timed step imports Sage while the setitimer handler is set
    integrate(sp.Integral(1 / (1 + x**2), (x, 0, 1)),
              backends=("maxima",), rules=(), step_timeout=20)
    # a later Sage step running into its limit must raise, not kill
    try:
        run_with_timeout(lambda: sage().maxima.eval(
            "for i:1 thru 10^9 do i;"), 1.0)
    except StepTimeout:
        print("timeout ok")
""")


def test_sage_imported_inside_timed_step():
    # regression: restoring the default SIGALRM handler after Sage was
    # imported made later Sage alarms terminate the process
    out = subprocess.run([sys.executable, "-c", SAGE_LOADED_DURING_STEP],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, (out.returncode, out.stderr)
    assert "timeout ok" in out.stdout
