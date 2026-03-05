# `bnrpe.rotors`

Rotor construction and application routines for BNR-PE.

## Functions

### `apply_bnrpe`

```python
def apply_bnrpe(x: 'jnp.ndarray', P: 'jnp.ndarray', params: 'BNRPEParams') -> 'jnp.ndarray'
```

Apply BNR-PE rotor to x at position P.

Args:
  x: (..., d)
  P: (..., n_axes) coordinates
  params: BNRPEParams

Returns:
  x_rot: (..., d)

### `apply_bnrpe_batch`

```python
def apply_bnrpe_batch(X: 'jnp.ndarray', P: 'jnp.ndarray', params: 'BNRPEParams') -> 'jnp.ndarray'
```

Convenience wrapper for sequence-shaped tensors.

Args:
  X: ``(L, d)`` token features.
  P: ``(L, n_axes)`` positions.
  params: BNRPE parameter bundle.

### `attention_logits`

```python
def attention_logits(Q: 'jnp.ndarray', K: 'jnp.ndarray') -> 'jnp.ndarray'
```

Compute attention logits from rotated query/key matrices.

Args:
  Q: Query matrix of shape ``(L, d)``.
  K: Key matrix of shape ``(L, d)``.

Returns:
  Matrix of shape ``(L, L)``.

### `generator_lowrank`

```python
def generator_lowrank(P: 'jnp.ndarray', params: 'BNRPEParams') -> 'Tuple[jnp.ndarray, jnp.ndarray]'
```

Build concatenated low-rank factorization of G(P) = sum_a P_a * alpha * scale_a * U_a M_a U_a^T
as Ucat, Acat where G(P) = Ucat Acat Ucat^T, with Acat block-diagonal (n_axes*r × n_axes*r).

Returns:
  Ucat: ``(d, n_axes*r)`` concatenated low-rank basis.
  Acat: ``(n_axes*r, n_axes*r)`` skew block matrix.
