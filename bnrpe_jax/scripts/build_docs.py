"""Build the full documentation set (generated docs + MkDocs site)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise SystemExit(f"Command failed ({result.returncode}): {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build docs in strict mode (default: on).",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "site",
        help="MkDocs output directory.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    python = sys.executable
    strict_args = ["--strict"] if args.strict else []

    _run(
        [python, "bnrpe_jax/scripts/generate_api_docs.py", "--output-dir", "docs/api"],
        cwd=repo_root,
    )
    _run(
        [python, "bnrpe_jax/scripts/generate_context_docs.py", "--repo-root", ".", "--output-path", "docs/context_snapshot.md"],
        cwd=repo_root,
    )
    _run(
        [python, "-m", "mkdocs", "build", *strict_args, "--config-file", "mkdocs.yml", "--site-dir", str(args.site_dir)],
        cwd=repo_root,
    )
    print(f"Docs built successfully at {args.site_dir}")


if __name__ == "__main__":
    main()
