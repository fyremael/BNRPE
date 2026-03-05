"""Parameter structures and initialization utilities for BNR-PE."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

def skew(M: jnp.ndarray) -> jnp.ndarray:
    """Return the skew-symmetric component of a square matrix."""
    return 0.5 * (M - M.T)

@dataclass
class BNRPEParams:
    """Low-rank BNR-PE parameter bundle.

    Attributes:
      U: Per-axis low-rank basis of shape ``(n_axes, d, r)``.
      M_raw: Per-axis unconstrained generator cores of shape ``(n_axes, r, r)``.
      axis_scale: Per-axis scalar multipliers of shape ``(n_axes,)``.
      alpha: Global coupling scalar (shape ``()``).
    """
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
      alpha: coupling strength. ``alpha=0`` produces an identity rotor.

    Returns:
      BNRPEParams
    """
    k1, k2, k3 = jax.random.split(key, 3)
    U = jax.random.normal(k1, (n_axes, d, r)) / jnp.sqrt(d)
    M_raw = jax.random.normal(k2, (n_axes, r, r)) / jnp.sqrt(r)
    axis_scale = jnp.ones((n_axes,), dtype=jnp.float32)
    return BNRPEParams(U=U, M_raw=M_raw, axis_scale=axis_scale, alpha=jnp.array(alpha, dtype=jnp.float32))
