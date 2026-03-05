from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )


def grouped_bar_svg(
    title: str,
    categories: list[str],
    baseline_vals: list[float],
    candidate_vals: list[float],
    out_path: Path,
) -> None:
    width, height = 980, 420
    left, right, top, bottom = 90, 40, 70, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    vmax = max(max(baseline_vals), max(candidate_vals), 1.0) * 1.1
    n = len(categories)
    group_w = chart_w / max(n, 1)
    bar_w = min(28.0, group_w * 0.28)

    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="36" font-size="22" font-family="Segoe UI, Arial" fill="#0f172a">{esc(title)}</text>',
        f'<text x="{left}" y="56" font-size="13" font-family="Segoe UI, Arial" fill="#475569">Overhead % (lower is better)</text>',
    ]

    for i in range(6):
        y = top + (chart_h * i / 5)
        val = vmax * (1.0 - i / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Segoe UI, Arial" fill="#64748b">{val:.0f}</text>'
        )

    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#334155" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#334155" stroke-width="1.5"/>')

    for i, cat in enumerate(categories):
        gx = left + i * group_w + (group_w - (2 * bar_w + 8)) / 2
        b = baseline_vals[i]
        c = candidate_vals[i]
        b_h = (b / vmax) * chart_h
        c_h = (c / vmax) * chart_h
        by = top + chart_h - b_h
        cy = top + chart_h - c_h
        parts.append(f'<rect x="{gx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{b_h:.1f}" fill="#94a3b8"/>')
        parts.append(f'<rect x="{(gx + bar_w + 8):.1f}" y="{cy:.1f}" width="{bar_w:.1f}" height="{c_h:.1f}" fill="#0ea5e9"/>')
        parts.append(
            f'<text x="{left + (i + 0.5) * group_w:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial" fill="#334155">{esc(cat)}</text>'
        )

    lx = width - right - 230
    ly = 28
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#94a3b8"/>')
    parts.append(f'<text x="{lx+22}" y="{ly+12}" font-size="12" font-family="Segoe UI, Arial" fill="#334155">Baseline</text>')
    parts.append(f'<rect x="{lx+98}" y="{ly}" width="14" height="14" fill="#0ea5e9"/>')
    parts.append(f'<text x="{lx+120}" y="{ly+12}" font-size="12" font-family="Segoe UI, Arial" fill="#334155">Candidate</text>')

    out_path.write_text(svg_wrap(width, height, "".join(parts)), encoding="utf-8")


def delta_bar_svg(title: str, categories: list[str], deltas: list[float], out_path: Path) -> None:
    width, height = 980, 360
    left, right, top, bottom = 90, 40, 70, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    vmin = min(min(deltas), 0.0)
    vmax = max(max(deltas), 0.0)
    span = max(vmax - vmin, 1e-6)
    zero_y = top + (vmax / span) * chart_h
    n = len(categories)
    group_w = chart_w / max(n, 1)
    bar_w = min(44.0, group_w * 0.5)

    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="36" font-size="22" font-family="Segoe UI, Arial" fill="#0f172a">{esc(title)}</text>',
        f'<text x="{left}" y="56" font-size="13" font-family="Segoe UI, Arial" fill="#475569">Candidate - baseline overhead % (negative is better)</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#0f172a" stroke-width="1.4"/>',
    ]

    for i, cat in enumerate(categories):
        x = left + i * group_w + (group_w - bar_w) / 2
        val = deltas[i]
        h = abs(val) / span * chart_h
        y = zero_y - h if val >= 0 else zero_y
        color = "#ef4444" if val > 0 else "#16a34a"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w/2:.1f}" y="{(y - 6) if val>=0 else (y + h + 14):.1f}" text-anchor="middle" font-size="11" font-family="Segoe UI, Arial" fill="#334155">{val:.2f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w/2:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial" fill="#334155">{esc(cat)}</text>'
        )

    out_path.write_text(svg_wrap(width, height, "".join(parts)), encoding="utf-8")


def heatmap_svg(
    title: str,
    dims: list[int],
    lengths: list[int],
    grid: dict[tuple[int, int], float],
    out_path: Path,
) -> None:
    width, height = 980, 460
    left, right, top, bottom = 140, 70, 70, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    cw = chart_w / max(len(dims), 1)
    ch = chart_h / max(len(lengths), 1)

    vals = [grid[(l, d)] for l in lengths for d in dims]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1e-6)

    def color(v: float) -> str:
        t = (v - vmin) / span
        r = int(22 + 220 * t)
        g = int(163 - 120 * t)
        b = int(74 - 20 * t)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="{left}" y="36" font-size="22" font-family="Segoe UI, Arial" fill="#0f172a">{esc(title)}</text>',
        f'<text x="{left}" y="56" font-size="13" font-family="Segoe UI, Arial" fill="#475569">Candidate overhead % by (length, dim), rank=8</text>',
    ]

    for i, l in enumerate(lengths):
        y = top + i * ch
        parts.append(
            f'<text x="{left-14}" y="{y + ch/2 + 4:.1f}" text-anchor="end" font-size="12" font-family="Segoe UI, Arial" fill="#334155">{l}</text>'
        )
    for j, d in enumerate(dims):
        x = left + j * cw
        parts.append(
            f'<text x="{x + cw/2:.1f}" y="{top+chart_h+24}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial" fill="#334155">{d}</text>'
        )

    parts.append(f'<text x="{left-70}" y="{top+chart_h/2:.1f}" transform="rotate(-90 {left-70},{top+chart_h/2:.1f})" font-size="12" font-family="Segoe UI, Arial" fill="#475569">length</text>')
    parts.append(f'<text x="{left+chart_w/2:.1f}" y="{top+chart_h+52}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial" fill="#475569">dim</text>')

    for i, l in enumerate(lengths):
        for j, d in enumerate(dims):
            v = grid[(l, d)]
            x = left + j * cw
            y = top + i * ch
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{color(v)}" stroke="#ffffff" stroke-width="1"/>')
            parts.append(
                f'<text x="{x+cw/2:.1f}" y="{y+ch/2+4:.1f}" text-anchor="middle" font-size="12" font-family="Segoe UI, Arial" fill="#0f172a">{v:.1f}</text>'
            )

    lx = width - 44
    ly = top
    lh = chart_h
    for i in range(120):
        t = i / 119
        v = vmin + (1 - t) * (vmax - vmin)
        y = ly + t * lh
        parts.append(f'<rect x="{lx}" y="{y:.1f}" width="16" height="{lh/120+1:.2f}" fill="{color(v)}" stroke="none"/>')
    parts.append(f'<text x="{lx+22}" y="{ly+4}" font-size="11" font-family="Segoe UI, Arial" fill="#334155">{vmax:.1f}</text>')
    parts.append(f'<text x="{lx+22}" y="{ly+lh+4}" font-size="11" font-family="Segoe UI, Arial" fill="#334155">{vmin:.1f}</text>')

    out_path.write_text(svg_wrap(width, height, "".join(parts)), encoding="utf-8")


def gate_strip_svg(gate: dict, out_path: Path) -> None:
    checks = gate.get("checks", [])
    status_colors = {"pass": "#16a34a", "warn": "#d97706", "fail": "#dc2626"}
    width = 980
    height = 180 + 52 * len(checks)
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>']
    parts.append('<text x="32" y="38" font-size="24" font-family="Segoe UI, Arial" fill="#0f172a">CI Gate Snapshot</text>')
    overall = gate.get("overall_status", "UNKNOWN")
    overall_color = {"GREEN": "#16a34a", "AMBER": "#d97706", "RED": "#dc2626"}.get(overall, "#334155")
    parts.append(f'<rect x="32" y="54" width="210" height="44" rx="8" fill="{overall_color}"/>')
    parts.append(
        f'<text x="137" y="82" text-anchor="middle" font-size="20" font-family="Segoe UI, Arial" fill="#ffffff">OVERALL {esc(overall)}</text>'
    )

    y0 = 124
    for i, check in enumerate(checks):
        y = y0 + i * 52
        st = check.get("status", "").lower()
        color = status_colors.get(st, "#64748b")
        name = check.get("name", "")
        detail = check.get("detail", "")
        parts.append(f'<rect x="32" y="{y}" width="120" height="34" rx="6" fill="{color}"/>')
        parts.append(
            f'<text x="92" y="{y+23}" text-anchor="middle" font-size="15" font-family="Segoe UI, Arial" fill="#ffffff">{esc(st.upper())}</text>'
        )
        parts.append(
            f'<text x="166" y="{y+15}" font-size="13" font-family="Segoe UI, Arial" fill="#0f172a">{esc(name)}</text>'
        )
        parts.append(
            f'<text x="166" y="{y+31}" font-size="11" font-family="Segoe UI, Arial" fill="#475569">{esc(detail[:110])}</text>'
        )

    out_path.write_text(svg_wrap(width, height, "".join(parts)), encoding="utf-8")


def write_showcase_md(
    out_md: Path,
    visuals_dir: Path,
    comparison: dict,
    gate: dict,
) -> None:
    rank_delta = {int(r["rank"]): float(r["delta"]) for r in comparison["rank_overall_delta"]}
    low_d = [r for r in comparison["rank_by_dim_delta"] if int(r["rank"]) == 8 and int(r["dim"]) in (64, 128)]
    low_d_txt = ", ".join(f"d={int(r['dim'])}: {float(r['delta']):.2f}" for r in sorted(low_d, key=lambda x: int(x["dim"])))
    overall = gate.get("overall_status", "UNKNOWN")

    lines = [
        "# Research Showcase: Dual-Axis Breakthrough",
        "",
        f"- Matrix comparison verdict: **{comparison.get('verdict', '').upper()}**",
        f"- CI gate status: **{overall}**",
        f"- Rank-4 overall median delta: **{rank_delta.get(4, 0.0):.2f}%**",
        f"- Rank-8 overall median delta: **{rank_delta.get(8, 0.0):.2f}%**",
        f"- Rank-8 low-width deltas: **{low_d_txt}**",
        "",
        "## Visual Pack",
        "",
        f"![Rank-8 by Dim](./{visuals_dir.name}/rank8_by_dim_grouped.svg)",
        "",
        f"![Rank-4 by Dim](./{visuals_dir.name}/rank4_by_dim_grouped.svg)",
        "",
        f"![Rank-8 Length Delta](./{visuals_dir.name}/rank8_by_length_delta.svg)",
        "",
        f"![Rank-8 Heatmap](./{visuals_dir.name}/candidate_rank8_heatmap.svg)",
        "",
        f"![Gate Snapshot](./{visuals_dir.name}/gate_snapshot.svg)",
        "",
        "## Sources",
        "",
        "- `bnrpe_jax/artifacts_ci/research_compare_direct/comparison.json`",
        "- `bnrpe_jax/artifacts_ci/research_matrix_dual_axis/summary.json`",
        "- `bnrpe_jax/artifacts_ci/research_matrix_dual_axis_direct/summary.json`",
        "- `bnrpe_jax/artifacts_ci/governance/phase2_gate_report.json`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SVG showcase visuals for dual-axis research results.")
    parser.add_argument("--baseline-summary", default="artifacts_ci/research_matrix_dual_axis/summary.json")
    parser.add_argument("--candidate-summary", default="artifacts_ci/research_matrix_dual_axis_direct/summary.json")
    parser.add_argument("--comparison-json", default="artifacts_ci/research_compare_direct/comparison.json")
    parser.add_argument("--gate-json", default="artifacts_ci/governance/phase2_gate_report.json")
    parser.add_argument("--candidate-matrix-csv", default="artifacts_ci/research_matrix_dual_axis_direct/matrix_results.csv")
    parser.add_argument("--output-dir", default="../docs/showcase_2026-03-05")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "assets"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_json(repo_root / args.baseline_summary)
    candidate = load_json(repo_root / args.candidate_summary)
    comparison = load_json(repo_root / args.comparison_json)
    gate = load_json(repo_root / args.gate_json)
    candidate_rows = load_csv(repo_root / args.candidate_matrix_csv)

    def rank_dim_map(payload: dict, rank: int) -> tuple[list[int], list[float]]:
        rows = [r for r in payload["summary"]["rank_by_dim"] if int(r["rank"]) == rank]
        rows = sorted(rows, key=lambda x: int(x["dim"]))
        dims = [int(r["dim"]) for r in rows]
        vals = [float(r["median_overhead_pct"]) for r in rows]
        return dims, vals

    dims8, base8 = rank_dim_map(baseline, 8)
    _, cand8 = rank_dim_map(candidate, 8)
    grouped_bar_svg(
        "Rank-8 Dual-Axis Overhead by Model Width (d)",
        [str(x) for x in dims8],
        base8,
        cand8,
        visuals_dir / "rank8_by_dim_grouped.svg",
    )

    dims4, base4 = rank_dim_map(baseline, 4)
    _, cand4 = rank_dim_map(candidate, 4)
    grouped_bar_svg(
        "Rank-4 Dual-Axis Overhead by Model Width (d)",
        [str(x) for x in dims4],
        base4,
        cand4,
        visuals_dir / "rank4_by_dim_grouped.svg",
    )

    rank8_len = [r for r in comparison["rank_by_length_delta"] if int(r["rank"]) == 8]
    rank8_len = sorted(rank8_len, key=lambda x: int(x["length"]))
    delta_bar_svg(
        "Rank-8 Delta by Sequence Length",
        [str(int(r["length"])) for r in rank8_len],
        [float(r["delta"]) for r in rank8_len],
        visuals_dir / "rank8_by_length_delta.svg",
    )

    heat_rows = [r for r in candidate_rows if int(r["rank"]) == 8 and r.get("position_profile", "") == "dual_axis_non_degenerate"]
    dims = sorted({int(r["dim"]) for r in heat_rows})
    lengths = sorted({int(r["length"]) for r in heat_rows})
    grid = {(int(r["length"]), int(r["dim"])): float(r["overhead_pct"]) for r in heat_rows}
    heatmap_svg(
        "Candidate Rank-8 Overhead Surface",
        dims,
        lengths,
        grid,
        visuals_dir / "candidate_rank8_heatmap.svg",
    )

    gate_strip_svg(gate, visuals_dir / "gate_snapshot.svg")

    write_showcase_md(out_dir / "index.md", visuals_dir, comparison, gate)
    print(f"Wrote showcase to {out_dir}")


if __name__ == "__main__":
    main()
