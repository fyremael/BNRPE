# Future Extensions

This roadmap is research-first and evidence-gated.

## Short Horizon

1. Continue low-width (`d=64/128`) rank-8 dual-axis optimization.
2. Expand matrix runs to more seeds and report tail behavior.
3. Tighten gate thresholds only after repeated stable GREEN outcomes.

## Medium Horizon

1. Add fused kernels where they improve median and tail overhead.
2. Extend experiment scripts to default to non-degenerate two-axis profiles.
3. Add richer diagnostics for commutator budget vs runtime tradeoff.

## Long Horizon

1. Validate transfer to task-level training loops beyond synthetic benchmarks.
2. Explore adaptive rank selection by width/sequence regime.
3. Package optional backend-specific paths under a stable API contract.

## Extension Acceptance Rule

Any optimization or extension proposal should be accepted only if:

1. Matrix-vs-matrix comparison returns `ACCEPT`.
2. No correctness regressions are introduced in rotor invariants.
3. Gate report remains GREEN in CI mode at minimum.
