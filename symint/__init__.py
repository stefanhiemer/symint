"""
symint -- closed-form evaluation of multidimensional SymPy integrals.

symint takes an unevaluated ``sympy.Integral`` and evaluates it as far
as possible in closed form. Its
strategy is to never hand the whole integral to a computer-algebra
system: the integral is first rewritten and split into small one-variable
problems, which are solved by fast closed-form rules where possible and
by SymPy, Maxima or FriCAS otherwise. Whatever cannot be done stays a
SymPy ``Integral`` -- a partial result, never an exception -- and can be
turned into NumPy/SciPy code (with quadrature for the leftover integrals)
by `numpy_function`.

It was written for integrals over quantum-chemistry basis functions
(Gaussians, Slater-type and elliptic-coordinate functions, 1/r12 kernels)
and finite-element integrands (polynomials and rationals on reference
cells), but nothing in it is specific to either.

Pipeline
--------
`integrate` runs, for each integral:

1. **Transform** (optional, ``transform=``): inverse-distance factors
   Q^(-nu/2), Q quadratic in the integration variables, are replaced by
   an integral representation -- "gaussian" (Boys:
   2/Gamma(nu/2) int_0^oo t^(nu-1) exp(-t^2 Q) dt), "laplace" (s = t^2)
   or "fourier" -- so that the spatial integrals become Gaussian.
2. **Separation** (``separate=True``): on rectangular domains the
   integrand is split into products of lower-dimensional integrals; each
   distinct factor is evaluated once.
3. **Per-variable steps**: the closed-form rules in `RULES` are tried
   first (on any variable, if the domain is rectangular), then the
   back ends in ``algorithms`` in order, each step optionally under
   ``step_timeout`` seconds. Gaussian exponents are square-completed
   before each step.
4. **Outer variables** introduced by the transform are integrated last,
   after the Boys substitution u^2 = t^2/(p + t^2) maps them onto [0, 1].
5. **Result**: a closed form, or an expression containing Integrals over
   the variables that could not be done. Pass ``log=[]`` to see which
   rule or algorithm handled each step and why others failed.

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
`symfem_integral`
    Over a symfem reference cell (duck-typed; symfem is not imported).
    Simplices are Duffy-collapsed by default.

Closed-form rules
-----------------
A `Rule` integrates one variable over one range in closed form or
declines (returns None). The registry `RULES` is consulted in order;
`register_rule` / `unregister_rule` modify it, and ``integrate(...,
rules=...)`` overrides it for one call. Built in:

"gaussian" (`gaussian_rule`)
    int P(x) exp(-A x^2 + B x + C) dx over the whole line and half-lines
    (A provably positive) and over finite intervals (any A != 0; erfi-type
    results for A < 0), via erf plus a moment recurrence. This covers
    the Boys integral int_0^1 u^(2n) exp(-T u^2) du.
"azimuthal" (`azimuthal_rule`)
    Full periods of exp(A + k cos(phi) + m sin(phi)) -> 2 pi e^A I_0(M)
    and of (A + k cos(phi) + m sin(phi))^(-1/2) -> 4 K(2M/(A+M))/sqrt(A+M),
    the azimuthal part of 1/r12; M = sqrt(k^2 + m^2).
"polynomial" (`polynomial_rule`)
    Polynomials over finite ranges (FEM integrands on affine or collapsed
    cells), by exact antiderivative.

Back ends
---------
``algorithms`` lists what is tried after the rules, in order:

"sympy"
    ``sympy.integrate``, natively; keeps SymPy assumptions.
"maxima", "fricas", "giac"
    Through passagemath (Sage). Assumptions on symbols and limits are
    passed on; Maxima's sign and integer questions are answered by case
    splits.
"fricas_antiderivative"
    FriCAS antiderivative plus one-sided limits; only valid if the
    antiderivative is continuous on the interval, so check such results.

The default is ``DEFAULT_ALGORITHMS = ("maxima", "fricas")``; pass
``algorithms=()`` to use the rules only. Sage is imported only when a
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

Examples
--------
A Gaussian moment, with SymPy as the only back end:

>>> import sympy as sp
>>> from symbolic_qc.symint import integrate
>>> x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
>>> integrate(sp.Integral(x**2 * sp.exp(-a * x**2), (x, -sp.oo, sp.oo)),
...           algorithms=("sympy",))
sqrt(pi)/(2*a**(3/2))

A moment on the reference triangle, with rules only:

>>> from symbolic_qc.symint import simplex_integral
>>> y = sp.Symbol("y", real=True)
>>> integrate(simplex_integral(x**2 * y, (x, y)), algorithms=())
1/60

What no rule or back end can do stays an Integral:

>>> integrate(sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), algorithms=())
Integral(exp(-a*x**4), (x, 0, 1))

and can still be compiled, with quadrature for the remaining integral:

>>> from symbolic_qc.symint import compile_function, numpy_function
>>> f = compile_function(numpy_function(
...     "f", sp.Integral(sp.exp(-a * x**4), (x, 0, 1)), [a]))
>>> round(float(f(1.0)), 8)
0.84483859

A user-supplied rule, here for an otherwise unknown function g:

>>> from symbolic_qc.symint import Rule, register_rule, unregister_rule
>>> g = sp.Function("g")
>>> def g_rule(expr, var, bounds):
...     return sp.Integer(1) if expr == g(var) and bounds == [0, 1] else None
>>> _ = register_rule(Rule("g", lambda bounds: True, g_rule), first=True)
>>> integrate(sp.Integral(3 * g(x), (x, 0, 1)), algorithms=())
3
>>> unregister_rule("g")

Dependencies
------------
Required: SymPy. Code generation and the timeout fallback need NumPy and
SciPy. Optional: passagemath with Maxima/FriCAS (CAS back ends), symfem
(`symfem_integral` takes its reference cells, but symint does not import it).

symint does not import the surrounding symbolic_qc package and is meant
to be extracted into a package of its own.
"""

from .codegen import compile_function, module_source, numpy_function
from .domain import chart_integral, simplex_integral, symfem_integral
from .engine import (DEFAULT_ALGORITHMS, StepTimeout, integrate,
                     run_with_timeout)
from .rewrite import TRANSFORMS
from .rules import (RULES, Rule, apply_rules, azimuthal_rule, gaussian_rule,
                    polynomial_rule, register_rule, unregister_rule)
from .sage import from_sage, to_sage

__all__ = [
    "chart_integral", "simplex_integral", "symfem_integral",
    "integrate", "DEFAULT_ALGORITHMS", "TRANSFORMS", "StepTimeout",
    "run_with_timeout",
    "Rule", "RULES", "register_rule", "unregister_rule", "apply_rules",
    "polynomial_rule", "gaussian_rule", "azimuthal_rule",
    "to_sage", "from_sage",
    "numpy_function", "module_source", "compile_function",
]
