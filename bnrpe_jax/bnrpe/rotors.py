from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .params import BNRPEParams, skew


def _apply_bnrpe_oneaxis_flat(
    x_flat: jnp.ndarray,
    coeff: jnp.ndarray,
    U: jnp.ndarray,
    M_skew: jnp.ndarray,
) -> jnp.ndarray:
    # Exact single-axis Cayley path (r x r solve), vectorized over tokens.
    Ut = U.T
    Gram = Ut @ U
    MG = M_skew @ Gram
    c = 0.5 * coeff  # (L,)
    c_col = c[:, None]

    u = x_flat @ U
    y = x_flat + ((u @ M_skew.T) * c_col) @ Ut
    v = y @ U
    rhs = (v @ M_skew.T) * c_col

    eye = jnp.eye(U.shape[1], dtype=x_flat.dtype)[None, :, :]
    M = eye - c[:, None, None] * MG[None, :, :]
    z = jnp.linalg.solve(M, rhs[..., None])[..., 0]
    return y + z @ Ut


def _apply_bnrpe_twoaxis_flat(
    x_flat: jnp.ndarray,
    coeff0: jnp.ndarray,
    coeff1: jnp.ndarray,
    U0: jnp.ndarray,
    U1: jnp.ndarray,
    M0_skew: jnp.ndarray,
    M1_skew: jnp.ndarray,
) -> jnp.ndarray:
    # Exact two-axis Cayley path with batched block-Schur solves on (r x r) blocks.
    # This is algebraically equivalent to a direct (2r x 2r) solve but cheaper at common ranks.
    r = U0.shape[1]
    c0 = 0.5 * coeff0
    c1 = 0.5 * coeff1
    c0_col = c0[:, None]
    c1_col = c1[:, None]

    Ucat = jnp.concatenate([U0, U1], axis=1)  # (d, 2r)
    Utcat = Ucat.T
    Gram = Utcat @ Ucat
    G00 = Gram[:r, :r]
    G01 = Gram[:r, r:]
    G10 = Gram[r:, :r]
    G11 = Gram[r:, r:]

    u = x_flat @ Ucat
    u0 = u[:, :r]
    u1 = u[:, r:]
    corr0 = (u0 @ M0_skew.T) * c0_col
    corr1 = (u1 @ M1_skew.T) * c1_col
    y = x_flat + jnp.concatenate([corr0, corr1], axis=1) @ Utcat

    v = y @ Ucat
    v0 = v[:, :r]
    v1 = v[:, r:]
    rhs0 = (v0 @ M0_skew.T) * c0_col
    rhs1 = (v1 @ M1_skew.T) * c1_col

    eye = jnp.eye(r, dtype=x_flat.dtype)[None, :, :]
    b00 = M0_skew @ G00
    b01 = M0_skew @ G01
    b10 = M1_skew @ G10
    b11 = M1_skew @ G11

    A = eye - c0[:, None, None] * b00[None, :, :]
    B = -c0[:, None, None] * b01[None, :, :]
    C = -c1[:, None, None] * b10[None, :, :]
    D = eye - c1[:, None, None] * b11[None, :, :]

    # Solve A * Xa = rhs0 and A * Xb = B in batch.
    rhs0_e = rhs0[..., None]
    a_inv_rhs0 = jnp.linalg.solve(A, rhs0_e)[..., 0]
    a_inv_B = jnp.linalg.solve(A, B)

    schur = D - jnp.einsum("lij,ljk->lik", C, a_inv_B)
    schur_rhs = rhs1 - jnp.einsum("lij,lj->li", C, a_inv_rhs0)
    z1 = jnp.linalg.solve(schur, schur_rhs[..., None])[..., 0]
    z0 = a_inv_rhs0 - jnp.einsum("lij,lj->li", a_inv_B, z1)
    return y + jnp.concatenate([z0, z1], axis=1) @ Utcat


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
    coeff_all = params.alpha * params.axis_scale[None, :] * p_flat  # (L, n_axes)

    if n_axes == 2:
        axis0_all_zero = jnp.all(coeff_all[:, 0] == 0.0)
        axis1_all_zero = jnp.all(coeff_all[:, 1] == 0.0)

        def _only_axis0(_):
            return _apply_bnrpe_oneaxis_flat(x_flat, coeff_all[:, 0], params.U[0], M_skew[0])

        def _only_axis1(_):
            return _apply_bnrpe_oneaxis_flat(x_flat, coeff_all[:, 1], params.U[1], M_skew[1])

        def _both_axes(_):
            return _apply_bnrpe_twoaxis_flat(
                x_flat,
                coeff_all[:, 0],
                coeff_all[:, 1],
                params.U[0],
                params.U[1],
                M_skew[0],
                M_skew[1],
            )

        x_rot_flat = jax.lax.cond(
            axis1_all_zero,
            _only_axis0,
            lambda _: jax.lax.cond(axis0_all_zero, _only_axis1, _both_axes, operand=None),
            operand=None,
        )
        return x_rot_flat.reshape(x.shape)

    def _apply_one(x1, coeff):
        # Scaled axis blocks followed by vectorized block-diagonal assembly.
        blocks = M_skew * coeff[:, None, None]  # (n_axes, r, r)
        Acat = jnp.einsum("ab,bij->aibj", axis_eye, blocks).reshape((n_axes * M_skew.shape[1], n_axes * M_skew.shape[2]))
        return _apply_cayley_lowrank(x1, Ucat, Acat, Ut=Ut, Gram=Gram, eye_r=eye_r)

    x_rot_flat = jax.vmap(_apply_one)(x_flat, coeff_all)
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
