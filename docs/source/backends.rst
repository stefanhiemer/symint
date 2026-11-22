Back ends
=========

Whatever no rule handles goes to the back ends listed in ``backends``,
tried in order until one returns a result without an unevaluated
integral:

``"maxima"``, ``"fricas"``, ``"giac"``
   Through passagemath (Sage).
``"sympy"``
   ``sympy.integrate``, natively.
``"fricas_antiderivative"``
   A FriCAS antiderivative plus one-sided limits at the bounds (Sage's
   ``limit`` with ``limit_algorithm``). Only valid if the antiderivative
   is continuous on the interval, so check such results.

The default, `~symintegral.engine.pipeline.DEFAULT_BACKENDS`, is
``("maxima", "fricas", "sympy")``. ``backends=()`` uses the rules only.

Choosing back ends
------------------

On a benchmark of 33 integrals with known closed forms
(``benchmarks/known_integrals.py``), Maxima solves most, FriCAS some of
the rest (e.g. the Dirichlet integral), and SymPy several that both miss
(Beta function, Frullani, complete elliptic K). Giac added nothing on
that benchmark. Run it yourself with

.. code-block:: bash

   python benchmarks/known_integrals.py --timeout 20

Each back end gets its own copy of the prepared integrand: the Sage back
ends receive the Sage-simplified one (``simplify="full"`` or
``"radical"``), SymPy the unsimplified one, since Sage's simplified form
can defeat SymPy.

Assumptions
-----------

SymPy assumptions on the symbols (``positive``, ``nonnegative``,
``negative``, ``real``, ``integer``) and the integration ranges are passed
to Sage with ``assume`` for the duration of a step, and the results come
back with the original SymPy symbols, assumptions included.

Maxima sometimes asks questions during integration ("Is a positive or
negative?"). symintegral answers them by splitting into cases: it integrates
once per answer and returns the result if all cases agree, otherwise the
step fails and the next back end is tried.

Timeouts
--------

``step_timeout=<seconds>`` limits every single step. A step that runs
out of time is logged and the next back end is tried. The limit uses
Unix signals: cysignals' alarm once Sage is loaded (the only way to
interrupt Maxima and FriCAS), ``signal.setitimer`` before. It only works
in the main thread.

`~symintegral.engine.timeout.run_with_timeout` exposes the same mechanism for any callable
and raises `~symintegral.engine.timeout.StepTimeout`.

Converting expressions
----------------------

`~symintegral.sage.convert.to_sage` and `~symintegral.sage.convert.from_sage` convert between SymPy and Sage
expressions; ``from_sage(expr, symbols)`` restores the given SymPy
symbols (with their assumptions) in the result.
