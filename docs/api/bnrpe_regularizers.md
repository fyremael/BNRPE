# `bnrpe.regularizers`

Regularizers and budget terms for BNR-PE training experiments.

## Functions

### `commutator_budget_penalty`

```python
def commutator_budget_penalty(params: 'BNRPEParams', weight: 'float' = 1.0) -> 'jnp.ndarray'
```

Penalty: sum_{a<b} ||[A_a, A_b]||_F^2 for low-rank A_a = U_a skew(M_a) U_a^T.

This is a practical surrogate for a commutator budget.

Args:
  params: BNRPE parameter bundle.
  weight: Scalar multiplier for the summed penalty.

Returns:
  Scalar commutator penalty.
