from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> dict:
    by_rank: dict[int, list[float]] = defaultdict(list)
    by_rank_dim: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_rank_len: dict[tuple[int, int], list[float]] = defaultdict(list)
    converted: list[dict[str, int | float | str]] = []

    for row in rows:
        length = int(row["length"])
        dim = int(row["dim"])
        rank = int(row["rank"])
        overhead = float(row["overhead_pct"])
        profile = row.get("position_profile", "")
        converted.append(
            {
                "length": length,
                "dim": dim,
                "rank": rank,
                "position_profile": profile,
                "overhead_pct": overhead,
            }
        )
        by_rank[rank].append(overhead)
        by_rank_dim[(rank, dim)].append(overhead)
        by_rank_len[(rank, length)].append(overhead)

    summary = {
        "rank_overall": [],
        "rank_by_dim": [],
        "rank_by_length": [],
        "best_points": sorted(converted, key=lambda x: float(x["overhead_pct"]))[:5],
        "worst_points": sorted(converted, key=lambda x: float(x["overhead_pct"]), reverse=True)[:5],
    }

    for rank in sorted(by_rank):
        vals = by_rank[rank]
        summary["rank_overall"].append(
            {
                "rank": rank,
                "n": len(vals),
                "median_overhead_pct": median(vals),
                "min_overhead_pct": min(vals),
                "max_overhead_pct": max(vals),
            }
        )

    for rank, dim in sorted(by_rank_dim):
        vals = by_rank_dim[(rank, dim)]
        summary["rank_by_dim"].append(
            {
                "rank": rank,
                "dim": dim,
                "n": len(vals),
                "median_overhead_pct": median(vals),
                "min_overhead_pct": min(vals),
                "max_overhead_pct": max(vals),
            }
        )

    for rank, length in sorted(by_rank_len):
        vals = by_rank_len[(rank, length)]
        summary["rank_by_length"].append(
            {
                "rank": rank,
                "length": length,
                "n": len(vals),
                "median_overhead_pct": median(vals),
                "min_overhead_pct": min(vals),
                "max_overhead_pct": max(vals),
            }
        )
    return summary


def write_markdown(path: Path, summary: dict, config: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Dual-Axis Benchmark Matrix Summary\n\n")
        f.write(f"- Generated (UTC): {config['generated_at_utc']}\n")
        f.write(f"- Position profile: `{config['position_profile']}`\n")
        f.write(f"- Lengths: `{config['lengths']}`\n")
        f.write(f"- Dims: `{config['dims']}`\n")
        f.write(f"- Ranks: `{config['ranks']}`\n")
        f.write(f"- Alphas: `{config['alphas']}`\n")
        f.write(f"- Iters: `{config['iters']}`\n")
        f.write(f"- Seeds: `{config['seeds']}`\n\n")

        f.write("## Rank Overall\n\n")
        f.write("| rank | n | median_overhead_pct | min_overhead_pct | max_overhead_pct |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in summary["rank_overall"]:
            f.write(
                f"| {row['rank']} | {row['n']} | {row['median_overhead_pct']:.2f} | "
                f"{row['min_overhead_pct']:.2f} | {row['max_overhead_pct']:.2f} |\n"
            )

        f.write("\n## Rank by Dim\n\n")
        f.write("| rank | dim | n | median_overhead_pct | min_overhead_pct | max_overhead_pct |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in summary["rank_by_dim"]:
            f.write(
                f"| {row['rank']} | {row['dim']} | {row['n']} | {row['median_overhead_pct']:.2f} | "
                f"{row['min_overhead_pct']:.2f} | {row['max_overhead_pct']:.2f} |\n"
            )

        f.write("\n## Rank by Length\n\n")
        f.write("| rank | length | n | median_overhead_pct | min_overhead_pct | max_overhead_pct |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in summary["rank_by_length"]:
            f.write(
                f"| {row['rank']} | {row['length']} | {row['n']} | {row['median_overhead_pct']:.2f} | "
                f"{row['min_overhead_pct']:.2f} | {row['max_overhead_pct']:.2f} |\n"
            )

        f.write("\n## Best 5 Points (Lowest Overhead)\n\n")
        f.write("| rank | dim | length | overhead_pct |\n")
        f.write("|---:|---:|---:|---:|\n")
        for row in summary["best_points"]:
            f.write(
                f"| {row['rank']} | {row['dim']} | {row['length']} | {float(row['overhead_pct']):.2f} |\n"
            )

        f.write("\n## Worst 5 Points (Highest Overhead)\n\n")
        f.write("| rank | dim | length | overhead_pct |\n")
        f.write("|---:|---:|---:|---:|\n")
        for row in summary["worst_points"]:
            f.write(
                f"| {row['rank']} | {row['dim']} | {row['length']} | {float(row['overhead_pct']):.2f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize dual-axis benchmark matrix for research.")
    parser.add_argument("--output-dir", default="artifacts/research_matrix")
    parser.add_argument("--position-profile", default="dual_axis_non_degenerate")
    parser.add_argument("--lengths", default="128,256,512,1024")
    parser.add_argument("--dims", default="64,128,256,512")
    parser.add_argument("--ranks", default="4,8")
    parser.add_argument("--alphas", default="0.2")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seeds", default="0")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = parse_int_list(args.lengths)
    dims = parse_int_list(args.dims)
    ranks = parse_int_list(args.ranks)
    alphas = parse_float_list(args.alphas)
    seeds = parse_int_list(args.seeds)

    all_rows: list[dict[str, str]] = []
    run_records = []
    for alpha in alphas:
        for seed in seeds:
            run_name = f"alpha{alpha:g}_seed{seed}"
            run_out = out_dir / run_name
            run_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/benchmark_overhead.py",
                "--position-profiles",
                args.position_profile,
                "--output-dir",
                str(run_out),
                "--lengths",
                ",".join(str(x) for x in lengths),
                "--dims",
                ",".join(str(x) for x in dims),
                "--ranks",
                ",".join(str(x) for x in ranks),
                "--iters",
                str(args.iters),
                "--alpha",
                str(alpha),
                "--seed",
                str(seed),
            ]
            run(cmd, cwd=repo_root)
            rows = load_csv_rows(run_out / "benchmark_overhead.csv")
            for row in rows:
                row["alpha"] = str(alpha)
                row["seed"] = str(seed)
            all_rows.extend(rows)
            run_records.append({"alpha": alpha, "seed": seed, "output_dir": str(run_out)})

    matrix_csv = out_dir / "matrix_results.csv"
    if not all_rows:
        raise RuntimeError("No benchmark rows produced.")
    fieldnames = list(all_rows[0].keys())
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = summarize(all_rows)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at_utc,
        "position_profile": args.position_profile,
        "lengths": lengths,
        "dims": dims,
        "ranks": ranks,
        "alphas": alphas,
        "iters": args.iters,
        "seeds": seeds,
        "runs": run_records,
        "summary": summary,
    }
    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_md = out_dir / "summary.md"
    write_markdown(summary_md, summary, payload)

    print(f"Wrote {matrix_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")
    rank8 = [x for x in summary["rank_overall"] if int(x["rank"]) == 8]
    if rank8:
        print(f"Rank-8 median overhead: {rank8[0]['median_overhead_pct']:.2f}%")


if __name__ == "__main__":
    main()
