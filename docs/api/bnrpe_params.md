# `bnrpe.params`

Parameter structures and initialization utilities for BNR-PE.

## Classes

### `BNRPEParams`

```python
class BNRPEParams(U: 'jnp.ndarray', M_raw: 'jnp.ndarray', axis_scale: 'jnp.ndarray', alpha: 'jnp.ndarray') -> None
```

Low-rank BNR-PE parameter bundle.

Attributes:
  U: Per-axis low-rank basis of shape ``(n_axes, d, r)``.
  M_raw: Per-axis unconstrained generator cores of shape ``(n_axes, r, r)``.
  axis_scale: Per-axis scalar multipliers of shape ``(n_axes,)``.
  alpha: Global coupling scalar (shape ``()``).

### Dataclass Fields

- `U`: `jnp.ndarray`
- `M_raw`: `jnp.ndarray`
- `axis_scale`: `jnp.ndarray`
- `alpha`: `jnp.ndarray`

## Functions

### `init_params`

```python
def init_params(key: 'jax.random.KeyArray', d: 'int', r: 'int', n_axes: 'int', alpha: 'float' = 0.0) -> 'BNRPEParams'
```

Initialize low-rank axis generators.

Args:
  d: embedding dimension
  r: low-rank (per axis)
  n_axes: number of coordinate axes
  alpha: coupling strength. ``alpha=0`` produces an identity rotor.

Returns:
  BNRPEParams

### `skew`

```python
def skew(M: 'jnp.ndarray') -> 'jnp.ndarray'
```

Return the skew-symmetric component of a square matrix.
