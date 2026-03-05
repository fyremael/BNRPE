from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _index(rows: list[dict], keys: tuple[str, ...], value_key: str) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for row in rows:
        out[tuple(row[k] for k in keys)] = float(row[value_key])
    return out


def _format(v: float) -> str:
    return f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two research matrix summary JSON files.")
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-dir", default="artifacts/research_compare")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    candidate_path = Path(args.candidate_json)
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)

    bsum = baseline["summary"]
    csum = candidate["summary"]

    b_rank = _index(bsum["rank_overall"], ("rank",), "median_overhead_pct")
    c_rank = _index(csum["rank_overall"], ("rank",), "median_overhead_pct")

    b_dim = _index(bsum["rank_by_dim"], ("rank", "dim"), "median_overhead_pct")
    c_dim = _index(csum["rank_by_dim"], ("rank", "dim"), "median_overhead_pct")

    b_len = _index(bsum["rank_by_length"], ("rank", "length"), "median_overhead_pct")
    c_len = _index(csum["rank_by_length"], ("rank", "length"), "median_overhead_pct")

    rank_rows = []
    for key in sorted(set(b_rank) | set(c_rank)):
        b = b_rank.get(key)
        c = c_rank.get(key)
        if b is None or c is None:
            continue
        rank_rows.append({"rank": key[0], "baseline": b, "candidate": c, "delta": c - b})

    dim_rows = []
    for key in sorted(set(b_dim) | set(c_dim)):
        b = b_dim.get(key)
        c = c_dim.get(key)
        if b is None or c is None:
            continue
        dim_rows.append({"rank": key[0], "dim": key[1], "baseline": b, "candidate": c, "delta": c - b})

    len_rows = []
    for key in sorted(set(b_len) | set(c_len)):
        b = b_len.get(key)
        c = c_len.get(key)
        if b is None or c is None:
            continue
        len_rows.append({"rank": key[0], "length": key[1], "baseline": b, "candidate": c, "delta": c - b})

    low_d_rank8 = [r for r in dim_rows if r["rank"] == 8 and int(r["dim"]) in (64, 128)]
    low_d_rank8_improved = all(r["delta"] < 0.0 for r in low_d_rank8) if low_d_rank8 else False
    overall_non_regression = all(r["delta"] <= 0.0 for r in rank_rows)

    verdict = "accept" if (low_d_rank8_improved and overall_non_regression) else "reject"

    payload = {
        "baseline_json": str(baseline_path),
        "candidate_json": str(candidate_path),
        "verdict": verdict,
        "checks": {
            "low_d_rank8_improved": low_d_rank8_improved,
            "overall_non_regression": overall_non_regression,
        },
        "rank_overall_delta": rank_rows,
        "rank_by_dim_delta": dim_rows,
        "rank_by_length_delta": len_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "comparison.json"
    out_md = out_dir / "comparison.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Research Matrix Comparison\n\n")
        f.write(f"- Baseline: `{baseline_path}`\n")
        f.write(f"- Candidate: `{candidate_path}`\n")
        f.write(f"- Verdict: **{verdict.upper()}**\n")
        f.write(f"- low_d_rank8_improved: `{low_d_rank8_improved}`\n")
        f.write(f"- overall_non_regression: `{overall_non_regression}`\n\n")

        f.write("## Rank Overall Median Delta\n\n")
        f.write("| rank | baseline | candidate | delta (cand-base) |\n")
        f.write("|---:|---:|---:|---:|\n")
        for r in rank_rows:
            f.write(
                f"| {r['rank']} | {_format(r['baseline'])} | {_format(r['candidate'])} | {_format(r['delta'])} |\n"
            )

        f.write("\n## Rank by Dim Median Delta\n\n")
        f.write("| rank | dim | baseline | candidate | delta |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in dim_rows:
            f.write(
                f"| {r['rank']} | {r['dim']} | {_format(r['baseline'])} | {_format(r['candidate'])} | {_format(r['delta'])} |\n"
            )

        f.write("\n## Rank by Length Median Delta\n\n")
        f.write("| rank | length | baseline | candidate | delta |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in len_rows:
            f.write(
                f"| {r['rank']} | {r['length']} | {_format(r['baseline'])} | {_format(r['candidate'])} | {_format(r['delta'])} |\n"
            )

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Verdict: {verdict.upper()}")


if __name__ == "__main__":
    main()
