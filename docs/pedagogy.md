# Pedagogy

The teaching path is intentionally executable.

## Primary Notebook

- `bnrpe_jax/notebooks/BNRPE_DualAxis_Pedagogy.ipynb`
- Colab URL:
  `https://colab.research.google.com/github/fyremael/BNRPE/blob/main/bnrpe_jax/notebooks/BNRPE_DualAxis_Pedagogy.ipynb`

## Learning Sequence

1. Confirm rotor invariants (norm preservation checks).
2. Compare RoPE-like baseline overhead against BNR-PE at fixed ranks.
3. Sweep `(L, d)` to expose regime structure.
4. Interpret where optimization effort has highest payoff.

## Team Teaching Use

- Use the notebook live in technical reviews.
- Pair each plot with a gate or matrix result from `artifacts_ci`.
- Keep one reproducible command block per claim.

## Refresh Pedagogy Assets

```bash
cd bnrpe_jax
python scripts/build_research_showcase.py --output-dir artifacts/showcase_latest
```
