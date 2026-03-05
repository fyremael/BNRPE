# CI vs Full Delta Report

- CI overall: **GREEN**
- Full overall: **GREEN**

## Gate Checks

| Check | CI | Full |
|---|---|---|
| benchmark_rank_4_median_overhead | PASS | PASS |
| benchmark_rank_4_median_overhead_dual_axis_non_degenerate | PASS | PASS |
| benchmark_rank_8_median_overhead | PASS | PASS |
| benchmark_rank_8_median_overhead_dual_axis_non_degenerate | PASS | PASS |
| fusion_single_pass_tradeoff | PASS | PASS |
| fusion_sweep_stable_candidates | PASS | PASS |
| norm_preservation | PASS | PASS |

## Benchmark Median Overhead (%)

| Profile | Rank | CI | Full | Delta (Full-CI) |
|---|---:|---:|---:|---:|
| single_axis | 4 | -29.63 | -31.44 | -1.81 |
| single_axis | 8 | 22.59 | 6.36 | -16.23 |
| dual_axis_non_degenerate | 4 | 59.01 | 19.31 | -39.71 |
| dual_axis_non_degenerate | 8 | 198.15 | 191.41 | -6.74 |

## Fusion Prototype (overhead vs RoPE %)

| Mode | CI | Full | Delta (Full-CI) |
|---|---:|---:|---:|
| full_prerotate_bnr | 14.98 | -10.00 | -24.98 |
| hybrid_two_pass_rope_plus_lowrank_corr | 37.26 | 33.35 | -3.90 |
| hybrid_single_pass_neumann1 | 23.82 | 21.56 | -2.26 |
