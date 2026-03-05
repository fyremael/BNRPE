# BNR-PE JAX Reference Implementation

## Install
You need JAX installed for your platform.

Minimal deps:
- jax
- jaxlib

## Run demo
```bash
python demo.py
```

## Validation
```bash
python -m pytest -q
```

## Benchmark and experiment pipelines
```bash
python scripts/run_validation_suite.py --mode full --with-sweep
```

`Makefile` alternative:
```bash
make validate-all
```

Individual commands:
```bash
python scripts/benchmark_overhead.py --output-dir artifacts/benchmarks
python scripts/run_experiment_tables.py --output-dir artifacts/experiments
python scripts/prototype_fused_paths.py --output-dir artifacts/fusion
python scripts/sweep_fusion_hybrid.py --output-dir artifacts/fusion_sweep
python scripts/build_gate_report.py --output-dir artifacts/governance
```

## CI mode (fast)
```bash
python scripts/run_validation_suite.py --mode ci --output-root artifacts_ci
```

Outputs include a gate report at:
- `artifacts/governance/phase2_gate_report.md` (or `artifacts_ci/governance/...` in CI mode)

## Key ideas
- Generators per axis: A_a = U_a * skew(M_a) * U_a^T (low rank r)
- Rotor map: Cayley(G(P)) for exact orthogonality (norm-preserving)
- Efficient apply: Woodbury-style reduction solves an r×r system per token
- Regularizer: commutator budget surrogate using low-rank trace algebra
