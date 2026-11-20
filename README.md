# symint

Closed-form evaluation of multidimensional SymPy integrals.

symint takes an unevaluated `sympy.Integral` and evaluates it as far as
possible in closed form. It never hands the whole integral to a computer
algebra system. Instead it rewrites the integral and splits it into small
one-variable problems. Fast closed-form rules handle these where they can,
and SymPy, Maxima or FriCAS handle the rest. Anything it cannot evaluate
stays a SymPy `Integral`, so you get a partial result instead of an
exception. `numpy_function` can turn that partial result into NumPy/SciPy
code that uses quadrature for the remaining integrals.

The core package is general-purpose. Code tied to a particular
application lives in context subpackages, which you enable by importing
them:

| Subpackage | Adds |
|---|---|
| `symint.qc` | Quantum chemistry: the `"azimuthal"` rule, closed forms for the azimuthal part of 1/r12 kernels (Bessel I₀, elliptic K). |

```python
import symint.qc   # registers the "azimuthal" rule for all later integrate() calls
```

## Installation

```bash
pip install .                 # SymPy + passagemath (Maxima/FriCAS)
pip install ".[numeric]"      # + NumPy/SciPy, to run generated code
pip install -e ".[test]"      # development install with pytest
```

passagemath provides the default Maxima and FriCAS back ends. Sage is
imported only when a CAS step actually runs, so pure-rule calls
(`algorithms=()`) stay fast to start.

## Quick start

```python
import sympy as sp
from symint import integrate, simplex_integral, numpy_function, compile_function

x, y = sp.symbols("x y", real=True)
a = sp.Symbol("a", positive=True)

integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)),
          algorithms=("sympy",))
# sqrt(pi)/(2*a**(3/2))

integrate(simplex_integral(x**2 * y, (x, y)), algorithms=())   # reference triangle
# 1/60

leftover = integrate(sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), algorithms=())
f = compile_function(numpy_function("f", leftover, [a]))         # needs [numeric]
f(1.0)  # 0.8448385947...
```

Pass `log=[]` to `integrate` to see which rule or back end handled each
step. You can add your own closed-form rules with `register_rule`. The
module docstring (`help(symint)`) documents the full pipeline, the domains
and the back ends.

## Running tests

```bash
pytest
```

This runs the tests in `tests/` and the doctests in the package.
