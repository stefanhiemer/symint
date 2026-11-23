"""
symintegral -- closed-form evaluation of multidimensional SymPy integrals.

symintegral takes an unevaluated ``sympy.Integral`` and evaluates it as far
as possible in closed form. Its
strategy is to never hand the whole integral to a computer-algebra
system: the integral is first rewritten and split into small one-variable
problems, which are solved by fast closed-form rules where possible and
by SymPy, Maxima or FriCAS otherwise. Whatever cannot be done stays a
SymPy ``Integral`` -- a partial result, never an exception -- and can be
turned into NumPy/SciPy code (with quadrature for the leftover integrals)
by `numpy_function`.

The core package is general-purpose. Code tied to one application
lives in context subpackages, which register their rules when imported:

`symintegral.qc`
    Quantum chemistry: the "azimuthal" rule for the azimuthal part of
    1/r12 kernels (Bessel I_0 and elliptic K closed forms).

Pipeline
--------
`integrate` runs, for each integral:

1. **Transform** (optional, ``transform=``): factors Q^(-nu/2), Q
   quadratic in the integration variables (e.g. ``1/|r1 - r2|`` in Coulomb
   or gravitational kernels), couple the variables. They are replaced
   by an integral representation over a new outer variable --
   "gaussian" (2/Gamma(nu/2) int_0^oo t^(nu-1) exp(-t^2 Q) dt),
   "laplace" (s = t^2) or "fourier" -- after which the original
   variables appear only in Gaussians and decouple.
2. **Separation** (``separate=True``): on rectangular domains the
   integrand is split into products of lower-dimensional integrals; each
   distinct factor is evaluated once.
3. **Per-variable steps**: the closed-form rules in `RULES` are tried
   first (on any variable, if the domain is rectangular), then the
   back ends in ``backends`` in order, each step optionally under
   ``step_timeout`` seconds. Gaussian exponents are square-completed
   before each step.
4. **Outer variables** introduced by the transform are integrated last,
   after u^2 = t^2/(p + t^2) maps them from [0, oo) onto [0, 1] (in
   quantum chemistry: the Boys-function form).
5. **Result**: a closed form, or an expression containing Integrals over
   the variables that could not be done. Pass ``log=[]`` to see which
   rule or back end handled each step and why others failed.

Domains
-------
`integrate` takes a plain ``sympy.Integral``. For common domains these
functions build one from an integrand, volume element included:

`chart_integral`
    Over a coordinate-chart mapping with keys "limits" and
    "volume_element" (e.g. spherical, prolate-spheroidal).
`simplex_integral`
    Over the unit d-simplex, Duffy-collapsed onto [0, 1]^d, which makes
    triangles and tetrahedra rectangular, so that separation and free
    choice of integration order apply.

Closed-form rules
-----------------
A `Rule` integrates one variable over one range in closed form or
declines (returns None). The registry `RULES` is consulted in order;
`register_rule` / `unregister_rule` modify it, and ``integrate(...,
rules=...)`` overrides it for one call. Built in (context subpackages
such as `symintegral.qc` add more):

"gaussian" (`gaussian_rule`)
    int P(x) exp(-A x^2 + B x + C) dx over the whole line and half-lines
    (A provably positive) and over finite intervals (any A != 0; erfi-type
    results for A < 0), via erf plus a moment recurrence. This covers
    the Boys integral int_0^1 u^(2n) exp(-T u^2) du. Also linear
    exponents: int P(x) exp(B x + C) dx over half-lines (decaying) and
    finite intervals, and the Slater-type int_0^oo x^n exp(-k x) dx =
    Gamma(n + 1)/k^(n + 1) with symbolic n.
"polynomial" (`polynomial_rule`)
    Polynomials over finite ranges (FEM integrands on affine or collapsed
    cells), by exact antiderivative.

Back ends
---------
``backends`` lists what is tried after the rules, in order:

"sympy"
    ``sympy.integrate``, natively; keeps SymPy assumptions.
"maxima", "fricas", "giac"
    Through passagemath (Sage). Assumptions on symbols and limits are
    passed on; Maxima's sign and integer questions are answered by case
    splits.
"fricas_antiderivative"
    FriCAS antiderivative plus one-sided limits; only valid if the
    antiderivative is continuous on the interval, so check such results.

The default is ``DEFAULT_BACKENDS = ("maxima", "fricas", "sympy")``; pass
``backends=()`` to use the rules only. Sage is imported only when a
CAS step is actually attempted, so the rules, "sympy" and the code
generator work without it. `to_sage` and `from_sage` convert
expressions.

Timeouts
--------
`run_with_timeout` runs a callable under a wall-clock limit and raises
`StepTimeout` (cysignals alarms once Sage is loaded, ``setitimer``
otherwise). `integrate` uses it for every step when ``step_timeout`` is
given.

Code generation
---------------
`numpy_function`
    Source of a NumPy function returning a scalar, list or matrix of
    expressions, with common-subexpression elimination and array
    arguments unpacked into symbols. Special functions map to
    ``scipy.special`` (ellipk, iv, erf, erfi, ...); remaining Integrals
    become ``scipy.integrate.nquad`` calls.
`module_source`
    Joins generated functions into one module.
`compile_function`
    Executes generated source and returns the function.

Verification
------------
`check_antiderivative`
    Checks dF/dx = f: the residual is sorted into groups by its
    transcendental factors (exponentials split, radicals reduced), each
    group's rational coefficient is cancelled separately, and only what
    survives is simplified or, last, tested numerically. The returned
    `verify.Check` says which stage decided ("groups", "simplify",
    "numeric").
`check_definite`
    Compares a definite result with numerical quadrature at random
    parameter values.

Examples
--------
A Gaussian moment, with SymPy as the only back end:

>>> import sympy as sp
>>> from symintegral import integrate
>>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
>>> integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)),
...           backends=("sympy",))
sqrt(pi)/(2*a**(3/2))

A moment on the reference triangle, with rules only:

>>> from symintegral import simplex_integral
>>> y = sp.Symbol("y", real=True)
>>> integrate(simplex_integral(x**2 * y, (x, y)), backends=())
1/60

What no rule or back end can do stays an Integral:

>>> integrate(sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), backends=())
Integral(exp(-a*x**4), (x, 0, 1))

and can still be compiled, with quadrature for the remaining integral:

>>> from symintegral import compile_function, numpy_function
>>> f = compile_function(numpy_function(
...     "f", sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), [a]))
>>> round(float(f(1.0)), 8)
0.84483859

A user-supplied rule, here for an otherwise unknown function g:

>>> from symintegral import Rule, register_rule, unregister_rule
>>> g = sp.Function("g")
>>> def g_rule(expr, var, bounds):
...     return sp.Integer(1) if expr == g(var) and bounds == [0, 1] else None
>>> _ = register_rule(Rule("g", lambda bounds: True, g_rule), first=True)
>>> integrate(sp.Integral(3 * g(x), (x, 0, 1)), backends=())
3
>>> unregister_rule("g")

Dependencies
------------
Required: SymPy and passagemath with Maxima/FriCAS (CAS back ends).
Code generation and the timeout fallback need NumPy and SciPy.
"""

__version__ = "0.1.0"

from .codegen import compile_function, module_source, numpy_function
from .domain import chart_integral, simplex_integral
from .engine import (DEFAULT_BACKENDS, StepTimeout, integrate,
                     run_with_timeout)
from .rewrite import TRANSFORMS
from .rules import (RULES, Rule, apply_rules, gaussian_rule, polynomial_rule,
                    register_rule, unregister_rule)
from .sage import from_sage, to_sage
from .verify import check_antiderivative, check_definite

__all__ = [
    "__version__",
    "chart_integral", "simplex_integral",
    "integrate", "DEFAULT_BACKENDS", "TRANSFORMS", "StepTimeout",
    "run_with_timeout",
    "Rule", "RULES", "register_rule", "unregister_rule", "apply_rules",
    "polynomial_rule", "gaussian_rule",
    "to_sage", "from_sage",
    "numpy_function", "module_source", "compile_function",
    "check_antiderivative", "check_definite",
]
