# BNR-PE Engineering Process

This repository is Git-backed as of March 4, 2026.

## Daily flow
1. Create a branch from `main`.
2. Run validation locally:
   - `python scripts/run_validation_suite.py`
   - or `make validate-all`
3. Commit code and generated reports together when they are part of a decision package.

## Artifact contract
- Benchmarks: `artifacts/benchmarks/benchmark_overhead.{csv,json}`
- Experiment tables: `artifacts/experiments/{summary.md,results.tex,metrics.json}`
- Fusion prototype: `artifacts/fusion/fusion_prototype.{csv,json}`

## Release gate signals
1. Correctness:
   - all tests passing
   - no shape regressions in `apply_bnrpe`
2. Runtime:
   - benchmark overhead tracked by rank and sequence length
3. Model quality proxies:
   - norm preservation error
   - relative-offset drift diagnostic
