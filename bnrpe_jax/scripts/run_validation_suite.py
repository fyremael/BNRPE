from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full BNR-PE validation suite.")
    parser.add_argument("--output-root", default="artifacts", help="Root output directory for generated artifacts.")
    parser.add_argument("--mode", choices=["full", "ci"], default="full")
    parser.add_argument("--with-sweep", action="store_true", help="Run hybrid sweep and recommendation stage.")
    parser.add_argument("--sweep-seeds", default="0,1")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)

    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=repo_root)
    bench_cmd = [sys.executable, "scripts/benchmark_overhead.py", "--output-dir", str(output_root / "benchmarks")]
    exp_cmd = [sys.executable, "scripts/run_experiment_tables.py", "--output-dir", str(output_root / "experiments")]
    fusion_cmd = [sys.executable, "scripts/prototype_fused_paths.py", "--output-dir", str(output_root / "fusion")]
    if args.mode == "ci":
        bench_cmd += ["--lengths", "256,512", "--dims", "128", "--ranks", "4,8", "--iters", "10"]
        exp_cmd += ["--length", "256", "--dim", "128", "--ranks", "0,4,8", "--alphas", "0.0,0.2", "--seeds", "0,1"]
        fusion_cmd += ["--length", "256", "--dim", "128", "--rank", "8", "--hybrid-rank", "4", "--iters", "15"]

    run(bench_cmd, cwd=repo_root)
    run(exp_cmd, cwd=repo_root)
    run(fusion_cmd, cwd=repo_root)

    sweep_csv = ""
    if args.with_sweep:
        sweep_cmd = [
            sys.executable,
            "scripts/sweep_fusion_hybrid.py",
            "--seeds",
            args.sweep_seeds,
            "--output-dir",
            str(output_root / "fusion_sweep"),
        ]
        if args.mode == "ci":
            sweep_cmd += ["--hybrid-ranks", "2,4", "--single-pass-scales", "0.0005,0.001", "--iters", "10", "--length", "256", "--dim", "128"]
        run(sweep_cmd, cwd=repo_root)
        sweep_csv = str(output_root / "fusion_sweep" / "sweep_results.csv")

    run(
        [
            sys.executable,
            "scripts/build_gate_report.py",
            "--bench-csv",
            str(output_root / "benchmarks" / "benchmark_overhead.csv"),
            "--experiments-json",
            str(output_root / "experiments" / "metrics.json"),
            "--fusion-csv",
            str(output_root / "fusion" / "fusion_prototype.csv"),
            "--output-dir",
            str(output_root / "governance"),
        ]
        + (
            [
                "--max-r4-overhead-pass",
                "40",
                "--max-r4-overhead-warn",
                "60",
                "--max-r8-overhead-pass",
                "240",
                "--max-r8-overhead-warn",
                "300",
            ]
            if args.mode == "ci"
            else []
        )
        + (["--sweep-csv", sweep_csv] if sweep_csv else []),
        cwd=repo_root,
    )
    print(f"Validation suite completed in mode={args.mode}.")
    print(f"Gate report: {output_root / 'governance' / 'phase2_gate_report.md'}")


if __name__ == "__main__":
    main()
