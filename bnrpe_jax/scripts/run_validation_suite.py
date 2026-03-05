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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)

    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=repo_root)
    run(
        [
            sys.executable,
            "scripts/benchmark_overhead.py",
            "--output-dir",
            str(output_root / "benchmarks"),
        ],
        cwd=repo_root,
    )
    run(
        [
            sys.executable,
            "scripts/run_experiment_tables.py",
            "--output-dir",
            str(output_root / "experiments"),
        ],
        cwd=repo_root,
    )
    run(
        [
            sys.executable,
            "scripts/prototype_fused_paths.py",
            "--output-dir",
            str(output_root / "fusion"),
        ],
        cwd=repo_root,
    )


if __name__ == "__main__":
    main()
