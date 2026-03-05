from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bnrpe.params import init_params
from bnrpe.regularizers import commutator_budget_penalty
from bnrpe.rotors import apply_bnrpe_batch, attention_logits


def build_positions(length: int, profile: str) -> jnp.ndarray:
    t = jnp.arange(length, dtype=jnp.float32)
    if profile == "single_axis":
        return jnp.stack([t, jnp.zeros_like(t)], axis=-1)
    if profile == "dual_axis_non_degenerate":
        axis1 = 0.5 * t + 1.0 + 0.25 * jnp.sin(t * 0.03125)
        return jnp.stack([t, axis1], axis=-1)
    raise ValueError(f"Unknown position profile: {profile}")


def drift_metric(logits: jnp.ndarray) -> float:
    # If logits are pure Delta-only, each diagonal should have near-constant values.
    L = logits.shape[0]
    diag_stds = []
    diag_means = []
    for offset in range(-(L - 1), L):
        diag = jnp.diag(logits, k=offset)
        if diag.shape[0] > 1:
            diag_stds.append(jnp.std(diag))
            diag_means.append(jnp.mean(jnp.abs(diag)))
    mean_std = jnp.mean(jnp.stack(diag_stds))
    mean_mag = jnp.mean(jnp.stack(diag_means))
    return float(mean_std / (mean_mag + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BNR-PE experiment sweeps and emit paper-ready tables.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--ranks", default="0,4,8,16", help="Comma-separated ranks; 0 is identity baseline.")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.4", help="Comma-separated alpha values.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds.")
    parser.add_argument(
        "--position-profile",
        choices=["single_axis", "dual_axis_non_degenerate"],
        default="single_axis",
        help="Coordinate profile used to build P.",
    )
    parser.add_argument("--output-dir", default="artifacts/experiments")
    args = parser.parse_args()

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    P = build_positions(args.length, args.position_profile)

    rows = []
    for rank in ranks:
        for alpha in alphas:
            norm_errs = []
            drifts = []
            comms = []
            for seed in seeds:
                qk = jax.random.PRNGKey(seed)
                q_key, k_key = jax.random.split(qk)
                Q = jax.random.normal(q_key, (args.length, args.dim))
                K = jax.random.normal(k_key, (args.length, args.dim))

                if rank == 0:
                    Qp = Q
                    Kp = K
                    comm = 0.0
                else:
                    params = init_params(
                        jax.random.PRNGKey(10_000 + seed + rank),
                        d=args.dim,
                        r=rank,
                        n_axes=2,
                        alpha=alpha,
                    )
                    Qp = apply_bnrpe_batch(Q, P, params)
                    Kp = apply_bnrpe_batch(K, P, params)
                    comm = float(commutator_budget_penalty(params))

                n0 = jnp.linalg.norm(Q, axis=-1)
                n1 = jnp.linalg.norm(Qp, axis=-1)
                norm_err = float(jnp.mean(jnp.abs(n1 - n0)))
                logits = attention_logits(Qp, Kp)
                drift = drift_metric(logits)

                norm_errs.append(norm_err)
                drifts.append(drift)
                comms.append(comm)

            rows.append(
                {
                    "rank": rank,
                    "alpha": alpha,
                    "position_profile": args.position_profile,
                    "norm_err_mean": float(sum(norm_errs) / len(norm_errs)),
                    "drift_mean": float(sum(drifts) / len(drifts)),
                    "comm_penalty_mean": float(sum(comms) / len(comms)),
                }
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "jax_platform": jax.default_backend(),
        "config": {
            "length": args.length,
            "dim": args.dim,
            "ranks": ranks,
            "alphas": alphas,
            "seeds": seeds,
            "position_profile": args.position_profile,
        },
        "results": rows,
    }

    metrics_json = out_dir / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_md = out_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# BNR-PE Experiment Summary\n\n")
        f.write("| position_profile | rank | alpha | norm_err_mean | drift_mean | comm_penalty_mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['position_profile']} | {row['rank']} | {row['alpha']:.2f} | {row['norm_err_mean']:.6e} | "
                f"{row['drift_mean']:.6e} | {row['comm_penalty_mean']:.6e} |\n"
            )

    results_tex = out_dir / "results.tex"
    with results_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{r r r r r}\n")
        f.write("rank & alpha & norm\\_err\\_mean & drift\\_mean & comm\\_penalty\\_mean \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['rank']} & {row['alpha']:.2f} & {row['norm_err_mean']:.3e} & "
                f"{row['drift_mean']:.3e} & {row['comm_penalty_mean']:.3e} \\\\\n"
            )
        f.write("\\end{tabular}\n")

    print(f"Wrote {metrics_json}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {results_tex}")


if __name__ == "__main__":
    main()
