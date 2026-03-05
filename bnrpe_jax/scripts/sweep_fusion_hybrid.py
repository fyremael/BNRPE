from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def is_dominated(points: list[dict], i: int) -> bool:
    a = points[i]
    for j, b in enumerate(points):
        if i == j:
            continue
        better_or_equal = (
            float(b["overhead_vs_rope_pct"]) <= float(a["overhead_vs_rope_pct"])
            and float(b["rel_mae_to_full"]) <= float(a["rel_mae_to_full"])
        )
        strictly_better = (
            float(b["overhead_vs_rope_pct"]) < float(a["overhead_vs_rope_pct"])
            or float(b["rel_mae_to_full"]) < float(a["rel_mae_to_full"])
        )
        if better_or_equal and strictly_better:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hybrid single-pass fusion settings.")
    parser.add_argument("--hybrid-ranks", default="2,4,6,8")
    parser.add_argument("--single-pass-scales", default="0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/fusion_sweep")
    args = parser.parse_args()

    hybrid_ranks = parse_int_list(args.hybrid_ranks)
    scales = parse_float_list(args.single_pass_scales)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for hr in hybrid_ranks:
        for sc in scales:
            run_dir = out_root / f"hr{hr}_s{sc:g}"
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
                "--single-pass-scale",
                str(sc),
                "--iters",
                str(args.iters),
                "--seed",
                str(args.seed),
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
                        rows.append(
                            {
                                "hybrid_rank": hr,
                                "single_pass_scale": sc,
                                "steady_s": float(r["steady_s"]),
                                "tokens_per_s": float(r["tokens_per_s"]),
                                "rel_mae_to_full": float(r["rel_mae_to_full"]),
                                "overhead_vs_rope_pct": float(r["overhead_vs_rope_pct"]),
                            }
                        )
                        break

    ranked = sorted(rows, key=lambda x: (x["rel_mae_to_full"], x["overhead_vs_rope_pct"]))
    frontier = [p for i, p in enumerate(ranked) if not is_dominated(ranked, i)]
    frontier = sorted(frontier, key=lambda x: (x["overhead_vs_rope_pct"], x["rel_mae_to_full"]))

    csv_all = out_root / "sweep_results.csv"
    csv_frontier = out_root / "pareto_frontier.csv"
    md_summary = out_root / "summary.md"

    with csv_all.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        w.writeheader()
        w.writerows(ranked)

    with csv_frontier.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frontier[0].keys()))
        w.writeheader()
        w.writerows(frontier)

    with md_summary.open("w", encoding="utf-8") as f:
        f.write("# Fusion Sweep Summary\n\n")
        f.write("## Top 5 by rel_mae_to_full then overhead\n\n")
        f.write("| hybrid_rank | single_pass_scale | rel_mae_to_full | overhead_vs_rope_pct | tokens_per_s |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in ranked[:5]:
            f.write(
                f"| {row['hybrid_rank']} | {row['single_pass_scale']:.6g} | {row['rel_mae_to_full']:.6f} | "
                f"{row['overhead_vs_rope_pct']:.2f} | {row['tokens_per_s']:.2f} |\n"
            )
        f.write("\n## Pareto Frontier\n\n")
        f.write("| hybrid_rank | single_pass_scale | rel_mae_to_full | overhead_vs_rope_pct | tokens_per_s |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in frontier:
            f.write(
                f"| {row['hybrid_rank']} | {row['single_pass_scale']:.6g} | {row['rel_mae_to_full']:.6f} | "
                f"{row['overhead_vs_rope_pct']:.2f} | {row['tokens_per_s']:.2f} |\n"
            )

    print(f"Wrote {csv_all}")
    print(f"Wrote {csv_frontier}")
    print(f"Wrote {md_summary}")


if __name__ == "__main__":
    main()
