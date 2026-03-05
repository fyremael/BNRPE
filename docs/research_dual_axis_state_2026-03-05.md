# Dual-Axis Research State (March 5, 2026)

## Question
Is the current BNR-PE dual-axis rank-8 path a dead end, or does it show recoverable performance regimes worth further research?

## Method
We ran a reproducible dual-axis benchmark matrix:

```bash
python bnrpe_jax/scripts/research_benchmark_matrix.py --output-dir artifacts_ci/research_matrix_dual_axis --position-profile dual_axis_non_degenerate --lengths 128,256,512,1024 --dims 64,128,256,512 --ranks 4,8 --alphas 0.2 --iters 30 --seeds 0
```

Primary outputs:
- `bnrpe_jax/artifacts_ci/research_matrix_dual_axis/matrix_results.csv`
- `bnrpe_jax/artifacts_ci/research_matrix_dual_axis/summary.json`
- `bnrpe_jax/artifacts_ci/research_matrix_dual_axis/summary.md`

## Results Snapshot
From `summary.md`:

- Rank-4 overall median overhead: `79.37%` (min `-3.09%`, max `324.84%`)
- Rank-8 overall median overhead: `215.50%` (min `32.04%`, max `624.78%`)

Rank-8 by model width (`d`):
- `d=64`: median `573.35%`
- `d=128`: median `300.93%`
- `d=256`: median `148.17%`
- `d=512`: median `66.51%`

Rank-4 by model width (`d`):
- `d=64`: median `221.27%`
- `d=128`: median `118.64%`
- `d=256`: median `50.44%`
- `d=512`: median `21.96%`

## Interpretation
This is **not** a uniform failure mode. The runtime burden is strongly regime-dependent:
- Narrow widths are currently expensive.
- Wider widths are substantially better and include low-overhead points.

That pattern supports a viable research hypothesis: current cost is dominated by fixed/low-rank solve overhead in low-`d` regimes, not by a fundamental incompatibility of dual-axis BNR-PE.

## Immediate Research Direction
1. Focus kernel-level optimization on low-`d`/rank-8 regimes (`d=64,128`) where overhead is worst.
2. Keep reporting split by `(d, L, r)` to avoid averaging away regime behavior.
3. Re-run the same matrix command after each optimization to quantify true movement in medians and worst-case tails.

## Evaluation Protocol (Level-Up)
Use matrix-vs-matrix comparison as a hard acceptance filter for optimization attempts:

```bash
python bnrpe_jax/scripts/compare_research_matrices.py --baseline-json bnrpe_jax/artifacts_ci/research_matrix_dual_axis/summary.json --candidate-json bnrpe_jax/artifacts_ci/research_matrix_dual_axis_opt3/summary.json --output-dir bnrpe_jax/artifacts_ci/research_compare_opt3
```

Current comparison result:
- Verdict: `REJECT`
- Reason: low-`d` rank-8 improved, but overall rank medians regressed.
- Evidence: `bnrpe_jax/artifacts_ci/research_compare_opt3/comparison.md`
