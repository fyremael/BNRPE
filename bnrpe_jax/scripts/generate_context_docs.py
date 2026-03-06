"""Generate contextual research snapshot docs from available artifact JSON files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


RESEARCH_SUMMARIES = [
    "bnrpe_jax/artifacts_ci/research_matrix_dual_axis/summary.json",
    "bnrpe_jax/artifacts_ci/research_matrix_dual_axis_direct_s012/summary.json",
    "bnrpe_jax/artifacts_full/research_matrix_dual_axis/summary.json",
]

GATE_REPORTS = [
    "bnrpe_jax/artifacts_ci/governance/phase2_gate_report.json",
    "bnrpe_jax/artifacts_full/governance/phase2_gate_report.json",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _is_tracked(repo_root: Path, path: Path) -> bool:
    rel = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", str(rel)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _extract_rank_metrics(payload: dict[str, Any]) -> tuple[str, str]:
    rank_4 = payload.get("rank_4_overall_median_overhead_pct")
    rank_8 = payload.get("rank_8_overall_median_overhead_pct")
    if rank_4 is None or rank_8 is None:
        rank_rows = payload.get("summary", {}).get("rank_overall", [])
        if isinstance(rank_rows, list):
            for row in rank_rows:
                if not isinstance(row, dict):
                    continue
                rank = row.get("rank")
                median = row.get("median_overhead_pct")
                if rank == 4 and rank_4 is None:
                    rank_4 = median
                if rank == 8 and rank_8 is None:
                    rank_8 = median
    rank_4_str = "n/a" if rank_4 is None else f"{float(rank_4):.2f}%"
    rank_8_str = "n/a" if rank_8 is None else f"{float(rank_8):.2f}%"
    return rank_4_str, rank_8_str


def _extract_gate_state(payload: dict[str, Any]) -> str:
    for key in ("overall_status", "status", "gate_status"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "n/a"


def generate(repo_root: Path, output_path: Path) -> None:
    lines = [
        "# Context Snapshot",
        "",
        "This page is auto-generated from tracked benchmark and governance artifacts.",
        "",
        "## Research Matrix Summaries",
        "",
        "| Source | Rank-4 Median Overhead | Rank-8 Median Overhead |",
        "|---|---:|---:|",
    ]

    found_summary = False
    for rel_path in RESEARCH_SUMMARIES:
        path = repo_root / rel_path
        if not _is_tracked(repo_root, path):
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        found_summary = True
        rank_4_str, rank_8_str = _extract_rank_metrics(payload)
        lines.append(f"| `{rel_path}` | {rank_4_str} | {rank_8_str} |")
    if not found_summary:
        lines.append("| _No summary files found_ | n/a | n/a |")

    lines.extend(["", "## Gate Reports", "", "| Source | Overall Status |", "|---|---|"])
    found_gate = False
    for rel_path in GATE_REPORTS:
        path = repo_root / rel_path
        if not _is_tracked(repo_root, path):
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        found_gate = True
        lines.append(f"| `{rel_path}` | `{_extract_gate_state(payload)}` |")
    if not found_gate:
        lines.append("| _No gate reports found_ | n/a |")

    lines.extend(
        [
            "",
            "## How To Refresh",
            "",
            "```bash",
            "cd bnrpe_jax",
            "python scripts/generate_context_docs.py",
            "```",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "context_snapshot.md",
        help="Output Markdown path for the context snapshot.",
    )
    args = parser.parse_args()
    generate(args.repo_root, args.output_path)
    print(f"Generated context snapshot at {args.output_path}")


if __name__ == "__main__":
    main()
