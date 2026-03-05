from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


@dataclass
class GateCheck:
    name: str
    status: str
    detail: str


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gate_status(value: float, pass_max: float, warn_max: float) -> str:
    if value <= pass_max:
        return "pass"
    if value <= warn_max:
        return "warn"
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build readiness gate report from generated artifacts.")
    parser.add_argument("--bench-csv", default="artifacts/benchmarks/benchmark_overhead.csv")
    parser.add_argument("--experiments-json", default="artifacts/experiments/metrics.json")
    parser.add_argument("--fusion-csv", default="artifacts/fusion/fusion_prototype.csv")
    parser.add_argument("--sweep-csv", default="", help="Optional stable candidate CSV from sweep_fusion_hybrid.")
    parser.add_argument("--output-dir", default="artifacts/governance")
    parser.add_argument("--max-r4-overhead-pass", type=float, default=25.0)
    parser.add_argument("--max-r4-overhead-warn", type=float, default=40.0)
    parser.add_argument("--max-r8-overhead-pass", type=float, default=120.0)
    parser.add_argument("--max-r8-overhead-warn", type=float, default=180.0)
    parser.add_argument("--max-norm-err-pass", type=float, default=1e-4)
    parser.add_argument("--max-norm-err-warn", type=float, default=1e-3)
    parser.add_argument("--fusion-rel-margin-pass", type=float, default=0.01)
    parser.add_argument("--fusion-rel-margin-warn", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[GateCheck] = []

    bench_rows = load_csv_rows(Path(args.bench_csv))
    by_rank: dict[int, list[float]] = {}
    for row in bench_rows:
        rank = int(row["rank"])
        by_rank.setdefault(rank, []).append(float(row["overhead_pct"]))

    for rank, pass_lim, warn_lim in [
        (4, args.max_r4_overhead_pass, args.max_r4_overhead_warn),
        (8, args.max_r8_overhead_pass, args.max_r8_overhead_warn),
    ]:
        vals = by_rank.get(rank, [])
        if not vals:
            checks.append(GateCheck(name=f"benchmark_rank_{rank}_median_overhead", status="fail", detail="missing rank data"))
            continue
        med = median(vals)
        status = gate_status(med, pass_lim, warn_lim)
        checks.append(
            GateCheck(
                name=f"benchmark_rank_{rank}_median_overhead",
                status=status,
                detail=f"median_overhead_pct={med:.2f}, pass<={pass_lim:.2f}, warn<={warn_lim:.2f}",
            )
        )

    with Path(args.experiments_json).open(encoding="utf-8") as f:
        exp = json.load(f)
    exp_rows = exp.get("results", [])
    max_norm_err = max(float(r["norm_err_mean"]) for r in exp_rows) if exp_rows else float("inf")
    norm_status = gate_status(max_norm_err, args.max_norm_err_pass, args.max_norm_err_warn)
    checks.append(
        GateCheck(
            name="norm_preservation",
            status=norm_status,
            detail=f"max_norm_err_mean={max_norm_err:.6e}, pass<={args.max_norm_err_pass:.1e}, warn<={args.max_norm_err_warn:.1e}",
        )
    )

    fusion_rows = load_csv_rows(Path(args.fusion_csv))
    fusion_by_mode = {r["mode"]: r for r in fusion_rows}
    rope_rel = float(fusion_by_mode["rope_core"]["rel_mae_to_full"])
    full_over = float(fusion_by_mode["full_prerotate_bnr"]["overhead_vs_rope_pct"])
    single = fusion_by_mode["hybrid_single_pass_neumann1"]
    single_rel = float(single["rel_mae_to_full"])
    single_over = float(single["overhead_vs_rope_pct"])
    rel_delta = single_rel - rope_rel
    fusion_status = "pass"
    # Enforce single-pass speedup only when full path itself is above the rank-8 pass budget.
    speed_guard_required = full_over > args.max_r8_overhead_pass
    if speed_guard_required and single_over > full_over:
        fusion_status = "fail"
    elif rel_delta > args.fusion_rel_margin_warn:
        fusion_status = "fail"
    elif rel_delta > args.fusion_rel_margin_pass:
        fusion_status = "warn"
    speed_note = (
        "full_path_above_rank8_pass_budget_speed_guard_enforced"
        if speed_guard_required
        else "full_path_within_rank8_pass_budget_speed_guard_relaxed"
    )
    checks.append(
        GateCheck(
            name="fusion_single_pass_tradeoff",
            status=fusion_status,
            detail=(
                f"single_overhead_pct={single_over:.2f}, full_overhead_pct={full_over:.2f}, "
                f"rel_mae_delta_vs_rope={rel_delta:.6f}, {speed_note}"
            ),
        )
    )

    if args.sweep_csv:
        sweep_rows = load_csv_rows(Path(args.sweep_csv))
        stable_count = len([r for r in sweep_rows if str(r.get("is_stable", "")).lower() == "true"])
        sweep_status = "pass" if stable_count > 0 else "warn"
        checks.append(
            GateCheck(
                name="fusion_sweep_stable_candidates",
                status=sweep_status,
                detail=f"stable_candidates={stable_count}",
            )
        )

    has_fail = any(c.status == "fail" for c in checks)
    has_warn = any(c.status == "warn" for c in checks)
    overall = "RED" if has_fail else ("AMBER" if has_warn else "GREEN")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": [asdict(c) for c in checks],
    }

    json_path = out_dir / "phase2_gate_report.json"
    md_path = out_dir / "phase2_gate_report.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 2 Gate Report\n\n")
        f.write(f"- Overall: **{overall}**\n")
        f.write(f"- Generated (UTC): {payload['generated_at_utc']}\n\n")
        f.write("| Check | Status | Detail |\n")
        f.write("|---|---|---|\n")
        for c in checks:
            f.write(f"| {c.name} | {c.status.upper()} | {c.detail} |\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Overall status: {overall}")


if __name__ == "__main__":
    main()
