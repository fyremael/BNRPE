from __future__ import annotations
import jax
import jax.numpy as jnp
from .params import BNRPEParams, skew

def _lowrank_commutator_fro2(Ua, Aa, Ub, Ab) -> jnp.ndarray:
    """
    Compute ||[Ua Aa Ua^T, Ub Ab Ub^T]||_F^2 efficiently without forming d×d.

    Let Sa = Ua Aa Ua^T, Sb = Ub Ab Ub^T.
    Then:
      Sa Sb = Ua Aa (Ua^T Ub) Ab Ub^T
      Sb Sa = Ub Ab (Ub^T Ua) Aa Ua^T
    So comm = Ua T Ub^T - Ub T' Ua^T (sum of two low-rank terms).
    We compute Frobenius norm via traces on small matrices.
    """
    # small overlaps
    Gab = Ua.T @ Ub              # (r,r)
    # T  = Aa Gab Ab   (r,r)
    T  = Aa @ Gab @ Ab
    Tp = Ab @ Gab.T @ Aa

    # comm = Ua T Ub^T - Ub Tp Ua^T.
    # ||comm||_F^2 = ||Ua T Ub^T||_F^2 + ||Ub Tp Ua^T||_F^2 - 2 <Ua T Ub^T, Ub Tp Ua^T>
    # Use: ||U A V^T||_F^2 = tr( A^T (U^T U) A (V^T V) )  if U,V not orthonormal.
    Ga = Ua.T @ Ua
    Gb = Ub.T @ Ub

    # term1
    term1 = jnp.trace(T.T @ Ga @ T @ Gb)
    term2 = jnp.trace(Tp.T @ Gb @ Tp @ Ga)

    # cross inner product:
    # Let X = Ua T Ub^T, Y = Ub Tp Ua^T.
    # <X,Y> = tr(X^T Y) = tr(T^T (Ua^T Ub) Tp (Ua^T Ub)).
    cross = jnp.trace(T.T @ Gab @ Tp @ Gab)

    return term1 + term2 - 2.0 * cross

def commutator_budget_penalty(params: BNRPEParams, weight: float = 1.0) -> jnp.ndarray:
    """
    Penalty: sum_{a<b} ||[A_a, A_b]||_F^2 for low-rank A_a = U_a skew(M_a) U_a^T.

    This is a practical surrogate for a commutator budget.
    """
    n_axes, d, r = params.U.shape
    As = [skew(params.M_raw[a]) * (params.alpha * params.axis_scale[a]) for a in range(n_axes)]
    total = 0.0
    for a in range(n_axes):
        for b in range(a+1, n_axes):
            total = total + _lowrank_commutator_fro2(params.U[a], As[a], params.U[b], As[b])
    return weight * total
