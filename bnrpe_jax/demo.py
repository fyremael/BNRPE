"""
Demo: BNR-PE positional encoding on a toy sequence in JAX.

Run:
  python demo.py

This script:
- initializes BNR-PE params
- constructs positions for 1D or 2D coordinates
- applies Cayley--Woodbury rotor to Q and K
- computes attention logits
- verifies approximate norm preservation
"""
import jax
import jax.numpy as jnp

from bnrpe.params import init_params
from bnrpe.rotors import apply_bnrpe_batch, attention_logits
from bnrpe.regularizers import commutator_budget_penalty

def main():
    key = jax.random.PRNGKey(0)
    L = 64
    d = 64
    r = 8
    n_axes = 2  # e.g. time and frequency axes

    # alpha controls non-Abelian coupling strength (0 -> near Abelian baseline)
    params = init_params(key, d=d, r=r, n_axes=n_axes, alpha=0.2)

    # Toy Q,K
    k1, k2 = jax.random.split(key)
    Q = jax.random.normal(k1, (L, d))
    K = jax.random.normal(k2, (L, d))

    # Coordinates: 2D grid unrolled (x = t, y = 0) for simplicity
    t = jnp.arange(L, dtype=jnp.float32)
    P = jnp.stack([t, jnp.zeros_like(t)], axis=-1)  # (L,2)

    Qp = apply_bnrpe_batch(Q, P, params)
    Kp = apply_bnrpe_batch(K, P, params)

    logits = attention_logits(Qp, Kp)

    # Norm preservation diagnostic (Cayley should preserve norms up to numerical solve tolerance)
    n0 = jnp.linalg.norm(Q, axis=-1).mean()
    n1 = jnp.linalg.norm(Qp, axis=-1).mean()

    pen = commutator_budget_penalty(params, weight=1.0)

    print("logits shape:", logits.shape)
    print("mean ||Q|| before:", float(n0))
    print("mean ||Q|| after :", float(n1))
    print("abs diff:", float(jnp.abs(n1 - n0)))
    print("commutator penalty:", float(pen))

if __name__ == "__main__":
    main()
