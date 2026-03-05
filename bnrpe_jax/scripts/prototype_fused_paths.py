from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bnrpe.params import BNRPEParams, init_params
from bnrpe.rotors import apply_bnrpe_batch, attention_logits, generator_lowrank


def apply_rope(X: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    d = X.shape[-1]
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))
    theta = pos[:, None] * inv_freq[None, :]
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    x_even = X[:, 0::2]
    x_odd = X[:, 1::2]
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    Y = jnp.empty_like(X)
    Y = Y.at[:, 0::2].set(y_even)
    Y = Y.at[:, 1::2].set(y_odd)
    return Y


def make_hybrid_params(full: BNRPEParams, rank_corr: int, alpha_scale: float) -> BNRPEParams:
    rank_corr = max(1, min(rank_corr, full.U.shape[-1]))
    return BNRPEParams(
        U=full.U[:, :, :rank_corr],
        M_raw=full.M_raw[:, :rank_corr, :rank_corr],
        axis_scale=full.axis_scale,
        alpha=full.alpha * alpha_scale,
    )


def apply_cayley_neumann1_batch(
    X: jnp.ndarray, P: jnp.ndarray, params: BNRPEParams, correction_scale: float
) -> jnp.ndarray:
    # Single-pass approximation:
    # (I - C G)^-1 ~= I + C G, where C = 0.5 * A and G = U^T U.
    def apply_one(x: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        U, A = generator_lowrank(p, params)
        C = 0.5 * A
        u = U.T @ x
        y = x + U @ (C @ u)
        G = U.T @ U
        z0 = C @ (U.T @ y)
        z1 = z0 + (C @ G) @ z0
        approx = y + U @ z1
        return x + correction_scale * (approx - x)

    return jax.vmap(apply_one)(X, P)


def timed_steady(fn, *args, iters: int) -> float:
    f = jax.jit(fn)
    out = f(*args)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype pre-rotate vs hybrid fusion paths.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8, help="Full BNR rank.")
    parser.add_argument("--hybrid-rank", type=int, default=4, help="Hybrid correction rank.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Full BNR alpha.")
    parser.add_argument("--hybrid-alpha-scale", type=float, default=0.5, help="Hybrid correction alpha multiplier.")
    parser.add_argument("--single-pass-scale", type=float, default=0.001, help="Damping for single-pass correction.")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/fusion")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    q_key, k_key = jax.random.split(key)
    Q = jax.random.normal(q_key, (args.length, args.dim))
    K = jax.random.normal(k_key, (args.length, args.dim))
    pos = jnp.arange(args.length, dtype=jnp.float32)
    P = jnp.stack([pos, jnp.zeros_like(pos)], axis=-1)

    full_params = init_params(
        jax.random.PRNGKey(10_000 + args.seed),
        d=args.dim,
        r=args.rank,
        n_axes=2,
        alpha=args.alpha,
    )
    hybrid_params = make_hybrid_params(full_params, args.hybrid_rank, args.hybrid_alpha_scale)

    def logits_full(Q_, K_, P_):
        Qp = apply_bnrpe_batch(Q_, P_, full_params)
        Kp = apply_bnrpe_batch(K_, P_, full_params)
        return attention_logits(Qp, Kp)

    def logits_rope(Q_, K_, pos_):
        Qp = apply_rope(Q_, pos_)
        Kp = apply_rope(K_, pos_)
        return attention_logits(Qp, Kp)

    def logits_hybrid(Q_, K_, pos_, P_):
        # Prototype strategy: RoPE core plus lower-rank/lower-alpha BNR correction.
        Q_core = apply_rope(Q_, pos_)
        K_core = apply_rope(K_, pos_)
        Q_h = apply_bnrpe_batch(Q_core, P_, hybrid_params)
        K_h = apply_bnrpe_batch(K_core, P_, hybrid_params)
        return attention_logits(Q_h, K_h)

    def logits_hybrid_single_pass(Q_, K_, pos_, P_):
        # Faster single-pass approximation: one-step Neumann solve approximation on top of RoPE core.
        Q_core = apply_rope(Q_, pos_)
        K_core = apply_rope(K_, pos_)
        Q_h = apply_cayley_neumann1_batch(Q_core, P_, hybrid_params, correction_scale=args.single_pass_scale)
        K_h = apply_cayley_neumann1_batch(K_core, P_, hybrid_params, correction_scale=args.single_pass_scale)
        return attention_logits(Q_h, K_h)

    full = jax.jit(logits_full)(Q, K, P)
    full.block_until_ready()
    rope = jax.jit(logits_rope)(Q, K, pos)
    rope.block_until_ready()
    hybrid = jax.jit(logits_hybrid)(Q, K, pos, P)
    hybrid.block_until_ready()
    hybrid_single = jax.jit(logits_hybrid_single_pass)(Q, K, pos, P)
    hybrid_single.block_until_ready()

    full_time = timed_steady(logits_full, Q, K, P, iters=args.iters)
    rope_time = timed_steady(logits_rope, Q, K, pos, iters=args.iters)
    hybrid_time = timed_steady(logits_hybrid, Q, K, pos, P, iters=args.iters)
    hybrid_single_time = timed_steady(logits_hybrid_single_pass, Q, K, pos, P, iters=args.iters)

    def rel_mae(candidate: jnp.ndarray, reference: jnp.ndarray) -> float:
        num = jnp.mean(jnp.abs(candidate - reference))
        den = jnp.mean(jnp.abs(reference))
        return float(num / (den + 1e-12))

    rows = [
        {
            "mode": "full_prerotate_bnr",
            "steady_s": full_time,
            "tokens_per_s": args.length / full_time,
            "rel_mae_to_full": 0.0,
            "overhead_vs_rope_pct": 100.0 * (full_time / rope_time - 1.0),
        },
        {
            "mode": "rope_core",
            "steady_s": rope_time,
            "tokens_per_s": args.length / rope_time,
            "rel_mae_to_full": rel_mae(rope, full),
            "overhead_vs_rope_pct": 0.0,
        },
        {
            "mode": "hybrid_two_pass_rope_plus_lowrank_corr",
            "steady_s": hybrid_time,
            "tokens_per_s": args.length / hybrid_time,
            "rel_mae_to_full": rel_mae(hybrid, full),
            "overhead_vs_rope_pct": 100.0 * (hybrid_time / rope_time - 1.0),
        },
        {
            "mode": "hybrid_single_pass_neumann1",
            "steady_s": hybrid_single_time,
            "tokens_per_s": args.length / hybrid_single_time,
            "rel_mae_to_full": rel_mae(hybrid_single, full),
            "overhead_vs_rope_pct": 100.0 * (hybrid_single_time / rope_time - 1.0),
        },
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "jax_platform": jax.default_backend(),
        "config": {
            "length": args.length,
            "dim": args.dim,
            "rank": args.rank,
            "hybrid_rank": args.hybrid_rank,
            "alpha": args.alpha,
            "hybrid_alpha_scale": args.hybrid_alpha_scale,
            "iters": args.iters,
            "seed": args.seed,
        },
        "results": rows,
    }

    json_path = out_dir / "fusion_prototype.json"
    csv_path = out_dir / "fusion_prototype.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"{row['mode']}: steady_s={row['steady_s']:.6f}, "
            f"overhead_vs_rope_pct={row['overhead_vs_rope_pct']:.2f}, "
            f"rel_mae_to_full={row['rel_mae_to_full']:.6f}"
        )


if __name__ == "__main__":
    main()
