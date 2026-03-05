from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_seed_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def is_dominated(points: list[dict], i: int) -> bool:
    a = points[i]
    for j, b in enumerate(points):
        if i == j:
            continue
        better_or_equal = (
            float(b["overhead_vs_rope_pct_mean"]) <= float(a["overhead_vs_rope_pct_mean"])
            and float(b["rel_mae_to_full_mean"]) <= float(a["rel_mae_to_full_mean"])
        )
        strictly_better = (
            float(b["overhead_vs_rope_pct_mean"]) < float(a["overhead_vs_rope_pct_mean"])
            or float(b["rel_mae_to_full_mean"]) < float(a["rel_mae_to_full_mean"])
        )
        if better_or_equal and strictly_better:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hybrid single-pass fusion settings.")
    parser.add_argument("--hybrid-ranks", default="2,4,6,8")
    parser.add_argument("--single-pass-scales", default="0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--seeds", default="0,1", help="Comma-separated seeds for stability aggregation.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--position-profile",
        choices=["single_axis", "dual_axis_non_degenerate"],
        default="single_axis",
        help="Coordinate profile forwarded to prototype_fused_paths.",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/fusion_sweep")
    parser.add_argument("--max-rel-mae-mean", type=float, default=1.10)
    parser.add_argument("--max-rel-mae-std", type=float, default=0.02)
    parser.add_argument("--max-overhead-mean-pct", type=float, default=25.0)
    parser.add_argument("--max-overhead-std-pct", type=float, default=10.0)
    args = parser.parse_args()

    hybrid_ranks = parse_int_list(args.hybrid_ranks)
    scales = parse_float_list(args.single_pass_scales)
    seeds = parse_seed_list(args.seeds)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, float | int | str]] = []
    for hr in hybrid_ranks:
        for sc in scales:
            for seed in seeds:
                run_dir = out_root / f"hr{hr}_s{sc:g}" / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "scripts/prototype_fused_paths.py",
                    "--length",
                    str(args.length),
                    "--dim",
                    str(args.dim),
                    "--rank",
                    str(args.rank),
                    "--hybrid-rank",
                    str(hr),
                    "--alpha",
                    str(args.alpha),
                    "--position-profile",
                    args.position_profile,
                    "--single-pass-scale",
                    str(sc),
                    "--iters",
                    str(args.iters),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(run_dir),
                ]
                print("$", " ".join(cmd))
                subprocess.run(cmd, cwd=repo_root, check=True)

                csv_path = run_dir / "fusion_prototype.csv"
                with csv_path.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r["mode"] == "hybrid_single_pass_neumann1":
                            run_rows.append(
                                {
                                    "hybrid_rank": hr,
                                    "single_pass_scale": sc,
                                    "seed": seed,
                                    "position_profile": args.position_profile,
                                    "steady_s": float(r["steady_s"]),
                                    "tokens_per_s": float(r["tokens_per_s"]),
                                    "rel_mae_to_full": float(r["rel_mae_to_full"]),
                                    "overhead_vs_rope_pct": float(r["overhead_vs_rope_pct"]),
                                }
                            )
                            break

    grouped: dict[tuple[int, float], list[dict[str, float | int | str]]] = {}
    for row in run_rows:
        key = (int(row["hybrid_rank"]), float(row["single_pass_scale"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, float | int | str | bool]] = []
    for (hr, sc), items in grouped.items():
        rels = [float(x["rel_mae_to_full"]) for x in items]
        overheads = [float(x["overhead_vs_rope_pct"]) for x in items]
        tps = [float(x["tokens_per_s"]) for x in items]
        n = len(items)
        rel_mean = sum(rels) / n
        overhead_mean = sum(overheads) / n
        tps_mean = sum(tps) / n
        rel_std = math.sqrt(sum((x - rel_mean) ** 2 for x in rels) / n)
        overhead_std = math.sqrt(sum((x - overhead_mean) ** 2 for x in overheads) / n)
        stable = (
            rel_mean <= args.max_rel_mae_mean
            and rel_std <= args.max_rel_mae_std
            and overhead_mean <= args.max_overhead_mean_pct
            and overhead_std <= args.max_overhead_std_pct
        )
        rows.append(
            {
                "hybrid_rank": hr,
                "single_pass_scale": sc,
                "num_runs": n,
                "tokens_per_s_mean": tps_mean,
                "rel_mae_to_full_mean": rel_mean,
                "rel_mae_to_full_std": rel_std,
                "overhead_vs_rope_pct_mean": overhead_mean,
                "overhead_vs_rope_pct_std": overhead_std,
                "is_stable": stable,
            }
        )

    ranked = sorted(rows, key=lambda x: (x["rel_mae_to_full_mean"], x["overhead_vs_rope_pct_mean"]))
    frontier = [p for i, p in enumerate(ranked) if not is_dominated(ranked, i)]
    frontier = sorted(frontier, key=lambda x: (x["overhead_vs_rope_pct_mean"], x["rel_mae_to_full_mean"]))
    stable_rows = [x for x in ranked if bool(x["is_stable"])]
    recommended = stable_rows[0] if stable_rows else (frontier[0] if frontier else None)

    csv_runs = out_root / "sweep_runs.csv"
    csv_all = out_root / "sweep_results.csv"
    csv_frontier = out_root / "pareto_frontier.csv"
    csv_stable = out_root / "stable_candidates.csv"
    json_recommended = out_root / "recommended_config.json"
    md_summary = out_root / "summary.md"

    with csv_runs.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)

    with csv_all.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(ranked)

    with csv_frontier.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frontier[0].keys()))
        w.writeheader()
        w.writerows(frontier)

    with csv_stable.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(stable_rows)

    if recommended is not None:
        with json_recommended.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_by": "stable_min_rel_mae_then_overhead" if stable_rows else "pareto_fallback",
                    "thresholds": {
                        "max_rel_mae_mean": args.max_rel_mae_mean,
                        "max_rel_mae_std": args.max_rel_mae_std,
                        "max_overhead_mean_pct": args.max_overhead_mean_pct,
                        "max_overhead_std_pct": args.max_overhead_std_pct,
                    },
                    "recommended": recommended,
                },
                f,
                indent=2,
            )

    with md_summary.open("w", encoding="utf-8") as f:
        f.write("# Fusion Sweep Summary\n\n")
        f.write("## Top 5 by rel_mae_to_full_mean then overhead_mean\n\n")
        f.write("| hybrid_rank | single_pass_scale | rel_mae_mean | rel_mae_std | overhead_mean_pct | overhead_std_pct | tokens_per_s_mean | stable |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in ranked[:5]:
            f.write(
                f"| {row['hybrid_rank']} | {row['single_pass_scale']:.6g} | {row['rel_mae_to_full_mean']:.6f} | "
                f"{row['rel_mae_to_full_std']:.6f} | {row['overhead_vs_rope_pct_mean']:.2f} | "
                f"{row['overhead_vs_rope_pct_std']:.2f} | {row['tokens_per_s_mean']:.2f} | {row['is_stable']} |\n"
            )
        f.write("\n## Pareto Frontier\n\n")
        f.write("| hybrid_rank | single_pass_scale | rel_mae_mean | overhead_mean_pct | tokens_per_s_mean |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in frontier:
            f.write(
                f"| {row['hybrid_rank']} | {row['single_pass_scale']:.6g} | {row['rel_mae_to_full_mean']:.6f} | "
                f"{row['overhead_vs_rope_pct_mean']:.2f} | {row['tokens_per_s_mean']:.2f} |\n"
            )
        f.write("\n## Recommended Config\n\n")
        if recommended is None:
            f.write("No recommendation available.\n")
        else:
            f.write(
                f"- hybrid_rank: {recommended['hybrid_rank']}\n"
                f"- single_pass_scale: {recommended['single_pass_scale']}\n"
                f"- rel_mae_to_full_mean: {recommended['rel_mae_to_full_mean']:.6f}\n"
                f"- overhead_vs_rope_pct_mean: {recommended['overhead_vs_rope_pct_mean']:.2f}\n"
                f"- is_stable: {recommended['is_stable']}\n"
            )

    print(f"Wrote {csv_runs}")
    print(f"Wrote {csv_all}")
    print(f"Wrote {csv_frontier}")
    print(f"Wrote {csv_stable}")
    if recommended is not None:
        print(f"Wrote {json_recommended}")
    print(f"Wrote {md_summary}")


if __name__ == "__main__":
    main()
