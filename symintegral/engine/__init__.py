"""
The symintegral integration pipeline.

``integrate(integral, ...)``:

1. rewrite (optional, `transform`): factors Q^(-nu/2), Q quadratic in the
   integration variables (e.g. ``1/|r1 - r2|``), are replaced by an integral
   over a new outer variable t with a Gaussian/exponential integrand in
   Q, which decouples the original variables. E.g. with Q = 1 + x^2,

       int exp(-x^2) / sqrt(1 + x^2) dx
         = 2/sqrt(pi) int_0^oo [ int exp(-x^2 - t^2 (1 + x^2)) dx ] dt,

   where the inner integral is now a plain Gaussian in x;
2. separate: on rectangular domains, split into products of
   lower-dimensional integrals, each distinct one evaluated once. E.g.

       int_0^1 int_0^1 (x^2 y^2 + x^2) dx dy
         = (int_0^1 x^2 dx) (int_0^1 y^2 dy) + (int_0^1 x^2 dx) (int_0^1 dy),

   so one 2-D integral becomes 1-D ones, and int_0^1 x^2 dx, which
   occurs in both terms, is evaluated once;
3. per variable: closed-form rules (`rules.RULES`) first -- on any
   variable if the domain is rectangular -- then the back ends in
   `backends` in order ("sympy" natively, CAS ones through Sage), each
   under `step_timeout`;
4. outer variables t last, after mapping t in [0, oo) onto u in [0, 1]
   (in quantum chemistry: the Boys-function form);
5. whatever cannot be integrated is returned as SymPy Integrals over the
   remaining variables (a partial result, never an exception), and every
   step is recorded in `log`.

Sage is imported only if a CAS step is actually attempted.

Submodules
----------
pipeline
    `integrate` and `DEFAULT_BACKENDS`.
steps
    Single steps: closed-form rules, back ends, `evaluate_iterated`.
context
    `Context`: options and state of one `integrate` call.
timeout
    `run_with_timeout` and `StepTimeout`.
"""

from .context import Context, LogEntry
from .pipeline import DEFAULT_BACKENDS, integrate
from .steps import evaluate_iterated
from .timeout import StepTimeout, run_with_timeout

__all__ = [
    "integrate", "DEFAULT_BACKENDS", "evaluate_iterated",
    "Context", "LogEntry", "StepTimeout", "run_with_timeout",
]
