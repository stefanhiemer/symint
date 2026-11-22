# symintegral

Closed-form evaluation of multidimensional SymPy integrals.

symintegral takes an unevaluated `sympy.Integral` and evaluates it as far as
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
| `symintegral.qc` | Quantum chemistry: the `"azimuthal"` rule, closed forms for the azimuthal part of 1/r12 kernels (Bessel I₀, elliptic K). |

```python
import symintegral.qc   # registers the "azimuthal" rule for all later integrate() calls
```

## Installation

Install from PyPI:

```bash
pip install symintegral               # SymPy + passagemath (Maxima/FriCAS)
pip install "symintegral[numeric]"    # + NumPy/SciPy, to run generated code
```

From a clone of the repository:

```bash
pip install -e ".[test]"      # development install with pytest
```

passagemath provides the default Maxima and FriCAS back ends. Sage is
imported only when a CAS step actually runs, so pure-rule calls
(`backends=()`) stay fast to start.

## Quick start

```python
import sympy as sp
from symintegral import integrate, simplex_integral, numpy_function, compile_function

x, y = sp.symbols("x y", real=True)
a = sp.Symbol("a", positive=True)

integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)),
          backends=("sympy",))
# sqrt(pi)/(2*a**(3/2))

integrate(simplex_integral(x**2 * y, (x, y)), backends=())   # reference triangle
# 1/60

leftover = integrate(sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), backends=())
f = compile_function(numpy_function("f", leftover, [a]))         # needs [numeric]
f(1.0)  # 0.8448385947...
```

Pass `log=[]` to `integrate` to see which rule or back end handled each
step. You can add your own closed-form rules with `register_rule`. The
documentation (see below, or `help(symintegral)`) describes the full pipeline,
the domains and the back ends.

## Checking results

`check_antiderivative(F, f, x)` checks dF/dx = f. It sorts the residual into
groups by its transcendental factors (exponentials split, radicals reduced)
and cancels each group's rational coefficient separately, so exact
cancellations are found without a full `simplify`. Only what survives is
simplified, and as a last resort tested numerically. The result says which
stage decided (`"groups"`, `"simplify"`, `"numeric"`).
`check_definite(result, integral)` compares a definite result with numerical
quadrature.

## Examples and benchmark

`examples/` contains runnable scripts: Gaussian moments, FEM element
matrices with code generation, the Coulomb integral of two Gaussians,
partial results evaluated by quadrature, a user-defined rule, result
checking, and the quantum-chemistry rules.

`benchmarks/known_integrals.py` runs symintegral on integrals with known closed
forms and reports for each whether the result is exact, only numerically
equal, partial, or wrong.

## Running tests

```bash
pytest
```

This runs the tests in `tests/` and the doctests in the package.

## Documentation

The documentation is built with Sphinx:

```bash
pip install -e ".[docs]"
cd docs
sphinx-apidoc -o source/ ../symintegral --force --no-toc --separate
make html
xdg-open build/html/index.html
```

## Building and uploading to PyPI

```bash
pip install -e ".[pypi]"
rm -rf dist/
python -m build
python -m twine check dist/*
```

Upload to TestPyPI first and test the install from there:

```bash
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ symintegral
```

then to PyPI:

```bash
python -m twine upload dist/*
```

## License

GPL-3.0-or-later, see `LICENSE`.
