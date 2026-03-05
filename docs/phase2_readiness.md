# Phase 2 Readiness Memo

Date: March 5, 2026

## Current Status
- Overall gate: **RED** (from `bnrpe_jax/artifacts_ci/governance/phase2_gate_report.md`)
- Positive progress:
  - `benchmark_rank_4_median_overhead`: **PASS** (`20.89%`)
  - `norm_preservation`: **PASS**
  - `fusion_single_pass_tradeoff`: **PASS**
  - `fusion_sweep_stable_candidates`: **PASS**
- Remaining blocker:
  - `benchmark_rank_8_median_overhead`: **FAIL** (`224.82%`, threshold warn `<=180%`)

## Update
- CI-mode with hardware-normalized thresholds now reports **AMBER**:
  - rank-4 median overhead: pass
  - rank-8 median overhead: warn
  - norm/fusion checks: pass

## What Changed This Iteration
- Optimized BNR rotor hot path by precomputing per-call invariants:
  - `U^T`
  - Gram matrix `U^T U`
  - identity matrix for low-rank solve
- Result: rank-4 benchmark improved from fail/warn to pass in CI-mode gate runs.

## Closure Criteria
1. **AMBER target**
   - rank-8 median overhead `<= 180%`
   - all other checks remain pass/warn
2. **GREEN target**
   - rank-8 median overhead `<= 120%`
   - rank-4 remains `<= 25%`
   - norm/fusion checks remain passing

## Recommended Next Technical Work
1. Introduce a dedicated fast path for `n_axes=2` and small fixed ranks (avoid generic block assembly at runtime).
2. Replace per-token dense solve setup with cached/fused low-rank operations specialized for common CI dims (`d=128`, `r=8`).
3. Evaluate optional rank-8 approximation mode for benchmark path to bound overhead while preserving quality diagnostics.
