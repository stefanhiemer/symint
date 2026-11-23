# symintegral

[![PyPI](https://img.shields.io/pypi/v/symintegral)](https://pypi.org/project/symintegral/)
[![Documentation](https://readthedocs.org/projects/symintegral/badge/?version=latest)](https://symintegral.readthedocs.io/en/latest/)
[![CI](https://github.com/stefanhiemer/symint/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/stefanhiemer/symint/actions/workflows/ci.yaml)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](#)

This is my personal library for symbolic integration. SymPy is my go-to 
python package for symbolic computing, but unfortunately slow on
multidimensional integrals and often fails to evaluate them. I started 
looking for lightweight packages to operate on sympy integrals, but found
only large general purpose symbolic packages like SAGE. Thus I created  
this package for my personal use as fusion of routines created in other 
projects, such as [topoptlab](https://github.com/stefanhiemer/topoptlab). 

A word of caution: this is not meant to be a complete or bullet-proof
integration library. It covers the integrals I need in my own work, and
it is developed as far as that requires. Expect gaps, and check results
that matter to you (e.g. with `check_definite`). Known failures are
collected in `benchmarks/LIBRARY_OF_SHAME.md`.

`symintegral` tries to find closed-form evaluation of multidimensional 
SymPy integrals by taking an unevaluated `sympy.Integral` and evaluating 
it as far as possible in closed form. Several simplification strategies 
are in place:

- rewrite the integral and splits it into small one-variable problems
- closed-form rules for specific integrals for speed
- use SymPy, Maxima or FriCAS to handle complex problems. 

Anything that cannot be solved stays a SymPy `Integral` as a partial 
result. `numpy_function` can turn that partial result into NumPy/SciPy 
code that uses quadrature for the remaining integrals.

The core package is general-purpose, but code tied to particular
applications live in context subpackages, which you enable by importing
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
documentation (see below, or `help(symintegral)`) describes the full 
pipeline, the domains and the back ends.

## Checking results

`check_antiderivative(F, f, x)` checks dF/dx = f. It sorts the residual 
into groups by its transcendental factors (exponentials split, radicals 
reduced) and cancels each group's rational coefficient separately, so 
exact cancellations are found without a full `simplify`. Only what 
survives is simplified, and as a last resort tested numerically. The 
result says which stage decided (`"groups"`, `"simplify"`, `"numeric"`).
`check_definite(result, integral)` compares a definite result with 
numerical quadrature.

## Examples and benchmark

`examples/` contains runnable scripts: 

- Gaussian moments
- FEM element matrices with code generation
- the Coulomb integral of two Gaussians,
- partial results evaluated by quadrature 
- creation/application of a user-defined rule
- result checking 
- quantum-chemistry specific rules.

`benchmarks/known_integrals.py` runs symintegral on integrals with known closed
forms and reports for each whether the result is exact, only numerically
equal, partial, or wrong.

## Running tests

```bash
pytest
```

This runs the tests in `tests/` and the doctests in the package.

## Documentation

The documentation is online at https://symintegral.readthedocs.io. To
build it locally with Sphinx:

```bash
pip install -e ".[docs]"
cd docs
sphinx-apidoc -o source/ ../symintegral --force --no-toc --separate
make html
xdg-open build/html/index.html
```

## Building and uploading to PyPI

`scripts/release.sh` does all of the steps below: it checks for
uncommitted changes and already-uploaded versions, runs the tests,
builds, uploads to TestPyPI, tests the install from there in a fresh
environment, and asks before the final PyPI upload. Bump the version
first and commit:

```bash
scripts/bump_version.py patch   # 0.1.0 -> 0.1.1 (or minor, major, X.Y.Z)
git commit -am "Release 0.1.1"
pip install -e ".[pypi,test]"
scripts/release.sh --dry-run    # checks and build only, no upload
scripts/release.sh              # TestPyPI, install test, then PyPI
```

The manual steps:

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
