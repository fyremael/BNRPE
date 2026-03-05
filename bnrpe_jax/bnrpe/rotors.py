from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .params import BNRPEParams, skew

def _apply_cayley_lowrank(
    v: jnp.ndarray,
    U: jnp.ndarray,
    A_skew: jnp.ndarray,
    Ut: jnp.ndarray | None = None,
    Gram: jnp.ndarray | None = None,
    eye_r: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Apply Cayley(S) to vector v where S = U A U^T and A is skew (r×r).
    Cayley(S) = (I - 0.5 S)^{-1} (I + 0.5 S).

    Complexity: O(d r + r^3).
    """
    # C = 0.5 A
    C = 0.5 * A_skew  # (r,r)

    # y = (I + U C U^T) v
    if Ut is None:
        Ut = U.T
    Utv = Ut @ v                  # (r,)
    y = v + U @ (C @ Utv)         # (d,)

    # Solve (I - U C U^T) x = y using Woodbury-style reduction.
    # Let x = y + U z. Then:
    # (I - U C U^T)(y + U z) = y  => (I - C (U^T U)) z = C U^T y
    if Gram is None:
        Gram = Ut @ U             # (r,r)
    rhs = C @ (Ut @ y)            # (r,)
    if eye_r is None:
        eye_r = jnp.eye(Gram.shape[0], dtype=Gram.dtype)
    M = eye_r - (C @ Gram)        # (r,r)
    z = jnp.linalg.solve(M, rhs)  # (r,)
    x = y + U @ z                 # (d,)
    return x

def generator_lowrank(P: jnp.ndarray, params: BNRPEParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build concatenated low-rank factorization of G(P) = sum_a P_a * alpha * scale_a * U_a M_a U_a^T
    as Ucat, Acat where G(P) = Ucat Acat Ucat^T, with Acat block-diagonal (n_axes*r × n_axes*r).

    Returns:
      Ucat: (d, n_axes*r)
      Acat: (n_axes*r, n_axes*r) skew
    """
    n_axes, d, r = params.U.shape
    assert P.shape[-1] == n_axes
    # scale each axis
    coeff = params.alpha * params.axis_scale * P  # (n_axes,)
    # Concatenate U blocks: [U_0, U_1, ...]
    Ucat = jnp.concatenate([params.U[a] for a in range(n_axes)], axis=1)  # (d, n_axes*r)
    # Block-diagonal A with scaled skew(M_raw[a])
    blocks = []
    for a in range(n_axes):
        A = skew(params.M_raw[a]) * coeff[a]
        blocks.append(A)
    # Build block diagonal
    Acat = jnp.block([[blocks[i] if i == j else jnp.zeros((r, r), dtype=blocks[0].dtype)
                       for j in range(n_axes)]
                      for i in range(n_axes)])
    # Ensure skew numerically
    Acat = skew(Acat)
    return Ucat, Acat

def apply_bnrpe(x: jnp.ndarray, P: jnp.ndarray, params: BNRPEParams) -> jnp.ndarray:
    """
    Apply BNR-PE rotor to x at position P.

    Args:
      x: (..., d)
      P: (..., n_axes) coordinates
      params: BNRPEParams

    Returns:
      x_rot: (..., d)
    """
    n_axes, d, _ = params.U.shape
    if x.shape[-1] != d:
        raise ValueError(f"x last dim must be {d}, got {x.shape[-1]}")
    if P.shape[-1] != n_axes:
        raise ValueError(f"P last dim must be {n_axes}, got {P.shape[-1]}")
    if x.shape[:-1] != P.shape[:-1]:
        raise ValueError(f"x and P leading dims must match, got {x.shape[:-1]} vs {P.shape[:-1]}")

    # Flatten all leading dims so the same vmap path supports vectors, batches, and N-D inputs.
    x_flat = x.reshape((-1, d))
    p_flat = P.reshape((-1, n_axes))
    Ucat = jnp.concatenate([params.U[a] for a in range(n_axes)], axis=1)  # (d, n_axes*r)
    Ut = Ucat.T
    Gram = Ut @ Ucat
    eye_r = jnp.eye(Gram.shape[0], dtype=Gram.dtype)
    M_skew = 0.5 * (params.M_raw - jnp.swapaxes(params.M_raw, -1, -2))  # (n_axes, r, r)
    axis_eye = jnp.eye(n_axes, dtype=x.dtype)

    def _apply_one(x1, P1):
        coeff = params.alpha * params.axis_scale * P1  # (n_axes,)
        # Scaled axis blocks followed by vectorized block-diagonal assembly.
        blocks = M_skew * coeff[:, None, None]  # (n_axes, r, r)
        Acat = jnp.einsum("ab,bij->aibj", axis_eye, blocks).reshape((n_axes * M_skew.shape[1], n_axes * M_skew.shape[2]))
        return _apply_cayley_lowrank(x1, Ucat, Acat, Ut=Ut, Gram=Gram, eye_r=eye_r)

    x_rot_flat = jax.vmap(_apply_one)(x_flat, p_flat)
    return x_rot_flat.reshape(x.shape)

def apply_bnrpe_batch(X: jnp.ndarray, P: jnp.ndarray, params: BNRPEParams) -> jnp.ndarray:
    """
    Convenience: X shape (L, d), P shape (L, n_axes).
    """
    return apply_bnrpe(X, P, params)

def attention_logits(Q: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """
    Compute attention logits: (L,d) @ (L,d)^T -> (L,L)
    """
    return Q @ K.T
