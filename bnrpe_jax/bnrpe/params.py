"""
BNR-PE JAX reference implementation.

Design goals:
- Minimal dependencies: jax, jax.numpy
- Structured low-rank skew generators per axis: A_a = U_a M_a U_a^T with M_a skew
- Exactly-orthogonal rotor map via Cayley transform, applied efficiently using Woodbury
- Optional differential rotor accumulation for 1D sequences
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

def skew(M: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (M - M.T)

@dataclass
class BNRPEParams:
    # U: (n_axes, d, r)
    U: jnp.ndarray
    # M_raw: (n_axes, r, r)  (skewed on use)
    M_raw: jnp.ndarray
    # scale per axis (n_axes,)
    axis_scale: jnp.ndarray
    # global mixing strength alpha (scalar)
    alpha: jnp.ndarray

def init_params(key: jax.random.KeyArray, d: int, r: int, n_axes: int,
                alpha: float = 0.0) -> BNRPEParams:
    """
    Initialize low-rank axis generators.

    Args:
      d: embedding dimension
      r: low-rank (per axis)
      n_axes: number of coordinate axes
      alpha: coupling strength; alpha=0 should behave close to Abelian if U blocks are disjoint.

    Returns:
      BNRPEParams
    """
    k1, k2, k3 = jax.random.split(key, 3)
    U = jax.random.normal(k1, (n_axes, d, r)) / jnp.sqrt(d)
    M_raw = jax.random.normal(k2, (n_axes, r, r)) / jnp.sqrt(r)
    axis_scale = jnp.ones((n_axes,), dtype=jnp.float32)
    return BNRPEParams(U=U, M_raw=M_raw, axis_scale=axis_scale, alpha=jnp.array(alpha, dtype=jnp.float32))
