import jax
import jax.numpy as jnp

from bnrpe.params import init_params, skew
from bnrpe.regularizers import _lowrank_commutator_fro2
from bnrpe.rotors import apply_bnrpe, apply_bnrpe_batch


def test_apply_bnrpe_supports_general_leading_dims():
    params = init_params(jax.random.PRNGKey(0), d=8, r=2, n_axes=2, alpha=0.2)

    x1 = jax.random.normal(jax.random.PRNGKey(1), (8,))
    p1 = jnp.array([1.0, 0.0], dtype=jnp.float32)
    y1 = apply_bnrpe(x1, p1, params)
    assert y1.shape == (8,)

    x2 = jax.random.normal(jax.random.PRNGKey(2), (4, 8))
    p2 = jnp.ones((4, 2), dtype=jnp.float32)
    y2 = apply_bnrpe(x2, p2, params)
    assert y2.shape == (4, 8)
    assert jnp.allclose(y2, apply_bnrpe_batch(x2, p2, params))

    x3 = jax.random.normal(jax.random.PRNGKey(3), (3, 5, 8))
    p3 = jnp.ones((3, 5, 2), dtype=jnp.float32)
    y3 = apply_bnrpe(x3, p3, params)
    assert y3.shape == (3, 5, 8)


def test_lowrank_commutator_matches_dense_reference():
    rel_errors = []
    for seed in range(20):
        params = init_params(jax.random.PRNGKey(seed), d=32, r=6, n_axes=2, alpha=0.7)
        Ua, Ub = params.U[0], params.U[1]
        Aa = skew(params.M_raw[0]) * (params.alpha * params.axis_scale[0])
        Ab = skew(params.M_raw[1]) * (params.alpha * params.axis_scale[1])

        est = _lowrank_commutator_fro2(Ua, Aa, Ub, Ab)
        Sa = Ua @ Aa @ Ua.T
        Sb = Ub @ Ab @ Ub.T
        dense = jnp.linalg.norm(Sa @ Sb - Sb @ Sa, ord="fro") ** 2
        rel = float(jnp.abs(est - dense) / (jnp.abs(dense) + 1e-12))
        rel_errors.append(rel)

    assert max(rel_errors) < 1e-4
