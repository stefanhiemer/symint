# Roadmap

Planned improvements to symintegral. Each item says what is limited today and
how it could be extended.

## Separation on non-rectangular domains

**Today.** `_evaluate_inner` (`symintegral/engine/pipeline.py`) separates an
integral into products of lower-dimensional integrals only if the whole
domain is rectangular (no limit depends on an integration variable).
Otherwise the full integral is evaluated iteratively.

**Why the restriction exists.** On a non-rectangular domain the limits
couple variables even when the integrand factorises:

    ∫₀¹ f(x) ∫₀ˣ g(y) dy dx = ∫₀¹ f(x) G(x) dx,   G(x) = ∫₀ˣ g(y) dy,

so the result is not (∫ f dx)(∫ g dy).

**Proposal.** Treat limit dependencies as couplings, like shared factors:

1. Build one graph over the integration variables with an edge between
   two variables if they share a factor of the integrand *or* the limits
   of one depend on the other.
2. Separate along the connected components of this graph; they are
   independent integrals. Example:

       ∫₀¹ dx ∫₀ˣ dy ∫₀¹ dz f(x) g(y) h(z)
         = [∫₀¹ ∫₀ˣ f(x) g(y) dy dx] · [∫₀¹ h(z) dz]

   z separates, and x and y stay coupled.
3. Within a coupled component, integrate in a topological order of the
   limit dependencies: a variable whose limits depend on another must be
   integrated first (inside). Here y before x.

**Out of scope for now.** Reordering inside a coupled component (e.g. x
before y on the triangle) requires rewriting the limits
(∫₀¹ dy ∫_y¹ dx). This is feasible for simple polytopes but not in
general. For simplices, `simplex_integral` already avoids the problem
with the Duffy collapse onto the unit cube.

**Gain.** Variables with constant limits that no other limit depends on
can be split off even when the rest of the domain is not rectangular.
