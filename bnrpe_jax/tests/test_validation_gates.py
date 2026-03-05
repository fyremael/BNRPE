from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _local_tmp_dir():
    path = REPO_ROOT / ".tmp_test_artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_benchmark_module():
    path = REPO_ROOT / "scripts" / "benchmark_overhead.py"
    spec = importlib.util.spec_from_file_location("benchmark_overhead", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gate_inputs(tmp_path: Path, include_dual_profile: bool) -> tuple[Path, Path, Path]:
    bench_csv = tmp_path / "benchmark_overhead.csv"
    exp_json = tmp_path / "metrics.json"
    fusion_csv = tmp_path / "fusion_prototype.csv"

    bench_fields = [
        "length",
        "dim",
        "rank",
        "n_axes",
        "alpha",
        "position_profile",
        "rope_compile_plus_first_s",
        "rope_steady_s",
        "rope_tokens_per_s",
        "bnr_compile_plus_first_s",
        "bnr_steady_s",
        "bnr_tokens_per_s",
        "overhead_pct",
    ]
    bench_rows = [
        {
            "length": "256",
            "dim": "128",
            "rank": "4",
            "n_axes": "2",
            "alpha": "0.2",
            "position_profile": "single_axis",
            "rope_compile_plus_first_s": "0.1",
            "rope_steady_s": "0.001",
            "rope_tokens_per_s": "256000",
            "bnr_compile_plus_first_s": "0.1",
            "bnr_steady_s": "0.0011",
            "bnr_tokens_per_s": "232727",
            "overhead_pct": "10.0",
        },
        {
            "length": "256",
            "dim": "128",
            "rank": "8",
            "n_axes": "2",
            "alpha": "0.2",
            "position_profile": "single_axis",
            "rope_compile_plus_first_s": "0.1",
            "rope_steady_s": "0.001",
            "rope_tokens_per_s": "256000",
            "bnr_compile_plus_first_s": "0.1",
            "bnr_steady_s": "0.0012",
            "bnr_tokens_per_s": "213333",
            "overhead_pct": "20.0",
        },
    ]
    if include_dual_profile:
        bench_rows.extend(
            [
                {
                    "length": "256",
                    "dim": "128",
                    "rank": "4",
                    "n_axes": "2",
                    "alpha": "0.2",
                    "position_profile": "dual_axis_non_degenerate",
                    "rope_compile_plus_first_s": "0.1",
                    "rope_steady_s": "0.001",
                    "rope_tokens_per_s": "256000",
                    "bnr_compile_plus_first_s": "0.1",
                    "bnr_steady_s": "0.0015",
                    "bnr_tokens_per_s": "170666",
                    "overhead_pct": "50.0",
                },
                {
                    "length": "256",
                    "dim": "128",
                    "rank": "8",
                    "n_axes": "2",
                    "alpha": "0.2",
                    "position_profile": "dual_axis_non_degenerate",
                    "rope_compile_plus_first_s": "0.1",
                    "rope_steady_s": "0.001",
                    "rope_tokens_per_s": "256000",
                    "bnr_compile_plus_first_s": "0.1",
                    "bnr_steady_s": "0.003",
                    "bnr_tokens_per_s": "85333",
                    "overhead_pct": "200.0",
                },
            ]
        )
    _write_csv(bench_csv, bench_fields, bench_rows)

    with exp_json.open("w", encoding="utf-8") as f:
        json.dump({"results": [{"norm_err_mean": 1e-6}]}, f)

    fusion_fields = ["mode", "steady_s", "tokens_per_s", "rel_mae_to_full", "overhead_vs_rope_pct"]
    fusion_rows = [
        {
            "mode": "full_prerotate_bnr",
            "steady_s": "0.0020",
            "tokens_per_s": "128000",
            "rel_mae_to_full": "0.0",
            "overhead_vs_rope_pct": "10.0",
        },
        {
            "mode": "rope_core",
            "steady_s": "0.0018",
            "tokens_per_s": "142000",
            "rel_mae_to_full": "1.0000",
            "overhead_vs_rope_pct": "0.0",
        },
        {
            "mode": "hybrid_two_pass_rope_plus_lowrank_corr",
            "steady_s": "0.0022",
            "tokens_per_s": "116000",
            "rel_mae_to_full": "1.0004",
            "overhead_vs_rope_pct": "22.0",
        },
        {
            "mode": "hybrid_single_pass_neumann1",
            "steady_s": "0.0021",
            "tokens_per_s": "121000",
            "rel_mae_to_full": "0.9995",
            "overhead_vs_rope_pct": "16.0",
        },
    ]
    _write_csv(fusion_csv, fusion_fields, fusion_rows)
    return bench_csv, exp_json, fusion_csv


def _run_gate_report(bench_csv: Path, exp_json: Path, fusion_csv: Path, out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_gate_report.py"),
        "--bench-csv",
        str(bench_csv),
        "--experiments-json",
        str(exp_json),
        "--fusion-csv",
        str(fusion_csv),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with (out_dir / "phase2_gate_report.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_build_positions_dual_axis_is_non_degenerate():
    mod = _load_benchmark_module()
    pos, p = mod.build_positions(16, 2, "dual_axis_non_degenerate")
    assert pos.shape == (16,)
    assert p.shape == (16, 2)
    assert jnp.allclose(p[:, 0], pos)
    assert not bool(jnp.all(p[:, 1] == 0.0))


def test_build_positions_single_axis_zeros_other_axes():
    mod = _load_benchmark_module()
    _, p = mod.build_positions(8, 3, "single_axis")
    assert p.shape == (8, 3)
    assert bool(jnp.all(p[:, 1:] == 0.0))


def test_gate_report_fails_when_required_dual_axis_profile_missing():
    with _local_tmp_dir() as tmp_path:
        bench_csv, exp_json, fusion_csv = _write_gate_inputs(tmp_path, include_dual_profile=False)
        payload = _run_gate_report(bench_csv, exp_json, fusion_csv, tmp_path / "governance")
    checks = {c["name"]: c["status"] for c in payload["checks"]}
    assert checks["benchmark_rank_4_median_overhead_dual_axis_non_degenerate"] == "fail"
    assert checks["benchmark_rank_8_median_overhead_dual_axis_non_degenerate"] == "fail"
    assert payload["overall_status"] == "RED"


def test_gate_report_passes_with_required_dual_axis_profile_present():
    with _local_tmp_dir() as tmp_path:
        bench_csv, exp_json, fusion_csv = _write_gate_inputs(tmp_path, include_dual_profile=True)
        payload = _run_gate_report(bench_csv, exp_json, fusion_csv, tmp_path / "governance")
    checks = {c["name"]: c["status"] for c in payload["checks"]}
    assert checks["benchmark_rank_4_median_overhead_dual_axis_non_degenerate"] == "pass"
    assert checks["benchmark_rank_8_median_overhead_dual_axis_non_degenerate"] == "pass"
    assert payload["overall_status"] == "GREEN"
