# Library of shame

Integrals symintegral gets wrong, cannot do, or does badly, worst first.
Each entry says what goes wrong, how to reproduce it, and what is known
about the cause. Entries from `known_integrals.py` can be rerun with

    python benchmarks/known_integrals.py --only "<name>"

State: 2026-11-22, default back ends `("maxima", "fricas", "sympy")`,
step timeout 20 s. Benchmark: 27 exact, 1 numeric, 3 partial, 2 WRONG.

## Wrong results

### 1. Separation splits convergent integrals into divergent pieces → `nan`

| Benchmark name | Integral | Correct | Returned |
|---|---|---|---|
| `Frullani` | ∫₀^∞ (e^(−ax) − e^(−bx))/x dx | log(b/a) | `nan` |
| `(x^a - 1)/log x` | ∫₀¹ (xᵃ − 1)/log x dx | log(a + 1) | `nan` |

Separation expands the integrand into terms and integrates each term
separately. Here every term diverges: SymPy returns ∞ for each, and the
sum is ∞ − ∞ = `nan`. With `separate=False` both are correct and fast
(0.1 s and 0.4 s).

This is a silent wrong answer, the worst kind. It appeared when "sympy"
became the third default back end. Before that the pieces stayed
unevaluated and the result was partial (correct, if useless).

Fix ideas: if a separated result contains `nan`, `oo` or `zoo`, or a back
end reports a piece as divergent, discard the separation and integrate
that integral unseparated. At least never return `nan` silently.

### 2. A Maxima timeout leaves Maxima broken for the rest of the process

```python
import sympy as sp
from symintegral import integrate
x = sp.Symbol("x", real=True); a = sp.Symbol("a", positive=True)
bose = sp.Integral(x / (sp.exp(x) - 1), (x, 0, sp.oo))
integrate(bose, step_timeout=20)            # pi**2/6
integrate(sp.Integral((x**a - 1) / sp.log(x), (x, 0, 1)), step_timeout=5)
integrate(bose, step_timeout=20)            # partial: "Integral is divergent"
```

After the step timeout interrupts Maxima, Maxima answers later,
unrelated integrals with "Integral is divergent". In the full benchmark
run this made `x/(e^x - 1)` and `x^3/(e^x - 1)` partial, and Maxima also
failed `x^(-2/3)/(1 + x)` (FriCAS rescued that one). Run alone, all
three are solved by Maxima in about 1 s.

What is known: Sage's `assumptions()` and Maxima's `facts()` are empty
afterwards, `context` is unchanged, and `forget(facts())` does not
help. The broken state is probably at the Lisp level, inside Maxima's
`defint` (e.g. special variables left bound by the interrupt).

Fix ideas: find and reset that state after a Maxima timeout; or restart
Maxima; or run Maxima steps with a timeout in a subprocess.

## Not solved

### 3. `x sin x/(1 + cos^2 x)`

∫₀^π x sin x/(1 + cos² x) dx = π²/4. No back end solves it. The textbook
route is the symmetry x → π − x ("King's rule"), which turns it into
(π/2) ∫₀^π sin x/(1 + cos² x) dx. A rule could try
I = ½ ∫ₐᵇ (f(x) + f(a + b − x)) dx when that sum is simpler.

### 4. Bose integrals with symbolic exponent

`x/(e^x - 1)` and `x^3/(e^x - 1)` are solved by Maxima when it is healthy
(see 2). A rule ∫₀^∞ x^(s−1)/(e^x − 1) dx = Γ(s) ζ(s) (and the Fermi
variant with η(s)) would make them independent of Maxima and might
also cover symbolic s. Not tested yet.

## Correct but ugly

### 5. `log(1+x)/(1+x^2)`

∫₀¹ log(1 + x)/(1 + x²) dx = π log 2 / 8. Maxima returns

    -Catalan + pi*log(2)/4 - I*polylog(2, 1/2 + I/2)/2
      + I*(-log(2)**2/8 + 5*pi**2/96 - Catalan*I + I*pi*log(2)/8)/2

which is equal only numerically; `simplify` cannot reduce it.

### 6. `Beta(p, q)`

SymPy returns `gamma(p)*hyper((p, 1 - q), (p + 1,), 1)/gamma(p + 1)`
instead of `beta(p, q)`. `simplify` proves them equal (the benchmark
says "exact"), but the form is hard to use. Options: a Beta rule, or
post-processing with `hyperexpand`/`gammasimp`.

### 7. Maxima's Gaussian antiderivative

Maxima's antiderivative of the Gaussian is written with `Abs`, `sign`
and `erfc` and is only piecewise equal to the usual form.
`check_antiderivative` confirms it numerically only, and plain `simplify`
wrongly calls the residual nonzero.

## Slow

### 8. `(x^a - 1)/log x` takes 20 s

Maxima runs into the step timeout on one of the separated pieces before
SymPy is tried. This timeout then triggers entry 2. Unseparated, SymPy
solves it in 0.4 s.

### 9. Quantum-chemistry integrals in symbolic-quantum-chemistry

From `examples/demo_sage_gaussian.py` there: the nuclear-attraction (V)
and ERI integrals stay partial without `transform=`, and the core
Hamiltonian (Hcore) hits its 30 s timeout. Not investigated.

## Engine gaps

10. The separation cache does not reuse integrals that differ only in
    the name of the variable (∫₀¹ x² dx vs ∫₀¹ y² dy).
11. Non-rectangular domains are not separated at all. See `ROADMAP.md`.
12. An unknown `simplify=` mode is only logged at every step, not
    rejected up front the way an unknown `transform=` is.
13. Untested code: the integer-question path in `sage/questions.py`,
    parts of `qc/rules.py`, and the timeout code that only runs in
    subprocesses (total coverage 87%).

## Fixed

Kept for reference; each has a regression test.

- SIGALRM killed the process ("Alarm clock") when Sage was first
  imported inside a timed step (`engine/timeout.py`).
- Sage's pre-simplified integrand was also given to SymPy, which then
  failed on Beta and elliptic K (`engine/steps.py`).
- The benchmark counted `nan` results as "numeric" because `nan > tol`
  is False. This hid entry 1 (`known_integrals.py`, no test).
