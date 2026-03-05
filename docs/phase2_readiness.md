# Phase 2 Readiness Memo

Date: March 5, 2026

## Current Status
- CI gate: **GREEN** (`bnrpe_jax/artifacts_ci/governance/phase2_gate_report.md`)
- Full gate: **GREEN** (`bnrpe_jax/artifacts_full/governance/phase2_gate_report.md`)
- Key pass signals:
  - benchmark rank-4 median overhead: pass
  - benchmark rank-8 median overhead: pass
  - benchmark rank-4 median overhead (`dual_axis_non_degenerate`): pass
  - benchmark rank-8 median overhead (`dual_axis_non_degenerate`): pass
  - norm preservation: pass
  - fusion tradeoff gate: pass
  - fusion sweep stable-candidate gate: pass

## What Changed This Iteration
- Added exact single-axis fast-path execution in `apply_bnrpe` when one axis stream is zero across the batch.
- Switched exact two-axis path to direct batched `(2r x 2r)` solves after matrix-level comparison showed broad non-regressing wins over the Schur-form variant.
- Stabilized fusion gate policy so speed-guard checks are applied only when full-path overhead exceeds rank-8 pass budget.
- Made non-degenerate two-axis benchmarking mandatory in validation output and gate evaluation.
- Added regression coverage validating one-axis and two-axis fast paths against the generic reference path.

## Remaining Risk
1. Two-axis rank-8 overhead is still higher than single-axis, especially in lower-width regimes.
2. Experiment/fusion scripts still use predominantly single-axis coordinates and should be expanded to dual-axis profiles.

## Recommended Next Technical Work
1. Continue low-width (`d=64/128`) rank-8-focused runtime tuning with matrix comparison acceptance gating.
2. Extend dual-axis experiments to multi-seed matrix runs and track worst-tail stability.
3. Tighten two-axis gate thresholds incrementally after repeated stable GREEN runs.
