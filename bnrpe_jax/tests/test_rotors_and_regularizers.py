import jax
import jax.numpy as jnp

from bnrpe.params import init_params, skew
from bnrpe.regularizers import _lowrank_commutator_fro2
from bnrpe.rotors import _apply_cayley_lowrank, apply_bnrpe, apply_bnrpe_batch, generator_lowrank


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


def test_single_axis_fast_path_matches_reference():
    params = init_params(jax.random.PRNGKey(7), d=16, r=4, n_axes=2, alpha=0.3)
    X = jax.random.normal(jax.random.PRNGKey(8), (6, 16))

    t = jnp.arange(6, dtype=jnp.float32)
    P_axis0 = jnp.stack([t, jnp.zeros_like(t)], axis=-1)
    P_axis1 = jnp.stack([jnp.zeros_like(t), t], axis=-1)

    def ref_apply(X_, P_):
        def one(x, p):
            Ucat, Acat = generator_lowrank(p, params)
            return _apply_cayley_lowrank(x, Ucat, Acat)

        return jax.vmap(one)(X_, P_)

    out0 = apply_bnrpe_batch(X, P_axis0, params)
    ref0 = ref_apply(X, P_axis0)
    assert jnp.allclose(out0, ref0, atol=1e-5, rtol=1e-5)

    out1 = apply_bnrpe_batch(X, P_axis1, params)
    ref1 = ref_apply(X, P_axis1)
    assert jnp.allclose(out1, ref1, atol=1e-5, rtol=1e-5)
