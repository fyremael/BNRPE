from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bnrpe.params import init_params
from bnrpe.rotors import apply_bnrpe_batch


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


def timed_jit(fn: Callable, *args, iters: int) -> tuple[float, float]:
    f = jax.jit(fn)

    t0 = time.perf_counter()
    out = f(*args)
    out.block_until_ready()
    compile_plus_first = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    out.block_until_ready()
    steady = (time.perf_counter() - t1) / iters
    return compile_plus_first, steady


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_positions(length: int, n_axes: int, profile: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    pos = jnp.arange(length, dtype=jnp.float32)
    if profile == "single_axis":
        if n_axes < 1:
            raise ValueError("single_axis profile requires n_axes >= 1")
        cols = [pos]
        for _ in range(1, n_axes):
            cols.append(jnp.zeros_like(pos))
        return pos, jnp.stack(cols, axis=-1)

    if profile == "dual_axis_non_degenerate":
        if n_axes < 2:
            raise ValueError("dual_axis_non_degenerate profile requires n_axes >= 2")
        axis1 = 0.5 * pos + 1.0 + 0.25 * jnp.sin(pos * 0.03125)
        cols = [pos, axis1]
        for a in range(2, n_axes):
            cols.append(0.1 * (a + 1) * pos + 0.1 * (a + 1))
        return pos, jnp.stack(cols, axis=-1)

    raise ValueError(f"Unknown position profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BNR-PE runtime overhead against RoPE.")
    parser.add_argument("--lengths", default="256,512,1024", help="Comma-separated sequence lengths.")
    parser.add_argument("--dims", default="128,256", help="Comma-separated embedding dimensions.")
    parser.add_argument("--ranks", default="4,8,16", help="Comma-separated BNR ranks.")
    parser.add_argument("--n-axes", type=int, default=2, help="Number of coordinate axes.")
    parser.add_argument(
        "--position-profiles",
        default="single_axis,dual_axis_non_degenerate",
        help="Comma-separated position profiles: single_axis, dual_axis_non_degenerate.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Global BNR alpha.")
    parser.add_argument("--iters", type=int, default=30, help="Timing iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks", help="Output directory.")
    args = parser.parse_args()

    lengths = parse_int_list(args.lengths)
    dims = parse_int_list(args.dims)
    ranks = parse_int_list(args.ranks)
    profiles = parse_str_list(args.position_profiles)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for L in lengths:
        profile_data = [build_positions(L, args.n_axes, profile) for profile in profiles]
        for d in dims:
            key = jax.random.PRNGKey(args.seed + L + d)
            X = jax.random.normal(key, (L, d))

            for profile, (pos, P) in zip(profiles, profile_data):
                rope_compile, rope_steady = timed_jit(apply_rope, X, pos, iters=args.iters)

                for r in ranks:
                    params = init_params(
                        jax.random.PRNGKey(args.seed + L + d + r),
                        d=d,
                        r=r,
                        n_axes=args.n_axes,
                        alpha=args.alpha,
                    )
                    bnr_compile, bnr_steady = timed_jit(
                        lambda X_, P_: apply_bnrpe_batch(X_, P_, params), X, P, iters=args.iters
                    )
                    rows.append(
                        {
                            "length": L,
                            "dim": d,
                            "rank": r,
                            "n_axes": args.n_axes,
                            "alpha": args.alpha,
                            "position_profile": profile,
                            "rope_compile_plus_first_s": rope_compile,
                            "rope_steady_s": rope_steady,
                            "rope_tokens_per_s": L / rope_steady,
                            "bnr_compile_plus_first_s": bnr_compile,
                            "bnr_steady_s": bnr_steady,
                            "bnr_tokens_per_s": L / bnr_steady,
                            "overhead_pct": 100.0 * (bnr_steady / rope_steady - 1.0),
                        }
                    )

    stamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": stamp,
        "jax_platform": jax.default_backend(),
        "config": {
            "lengths": lengths,
            "dims": dims,
            "ranks": ranks,
            "n_axes": args.n_axes,
            "alpha": args.alpha,
            "iters": args.iters,
            "seed": args.seed,
            "position_profiles": profiles,
        },
        "results": rows,
    }

    json_path = out_dir / "benchmark_overhead.json"
    csv_path = out_dir / "benchmark_overhead.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if rows:
        overheads = [float(x["overhead_pct"]) for x in rows]
        print(f"Overhead pct min/median/max: {min(overheads):.2f}/{sorted(overheads)[len(overheads)//2]:.2f}/{max(overheads):.2f}")


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
