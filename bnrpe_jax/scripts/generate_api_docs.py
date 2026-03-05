"""Generate Markdown API reference pages from the live ``bnrpe`` package."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_PAGES = [
    ("bnrpe", "bnrpe.md"),
    ("bnrpe.params", "bnrpe_params.md"),
    ("bnrpe.rotors", "bnrpe_rotors.md"),
    ("bnrpe.regularizers", "bnrpe_regularizers.md"),
]

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def _doc(obj: Any) -> str:
    return inspect.getdoc(obj) or "No docstring provided."


def _render_class(name: str, obj: type) -> list[str]:
    lines = [f"### `{name}`", "", "```python", f"class {name}{_safe_signature(obj)}", "```", ""]
    lines.extend([_doc(obj), ""])
    if is_dataclass(obj):
        lines.append("### Dataclass Fields")
        lines.append("")
        for field in fields(obj):
            lines.append(f"- `{field.name}`: `{field.type}`")
        lines.append("")
    return lines


def _render_function(name: str, obj: Any) -> list[str]:
    lines = [f"### `{name}`", "", "```python", f"def {name}{_safe_signature(obj)}", "```", ""]
    lines.extend([_doc(obj), ""])
    return lines


def _iter_public_members(module_name: str) -> tuple[list[tuple[str, type]], list[tuple[str, Any]]]:
    module = importlib.import_module(module_name)
    classes: list[tuple[str, type]] = []
    functions: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isclass(obj) and obj.__module__ == module_name:
            classes.append((name, obj))
        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append((name, obj))
    return classes, functions


def _write_module_page(module_name: str, destination: Path) -> None:
    module = importlib.import_module(module_name)
    lines: list[str] = [f"# `{module_name}`", ""]
    lines.extend([_doc(module), ""])
    if module_name == "bnrpe":
        exported = getattr(module, "__all__", [])
        if exported:
            lines.append("## Exported Symbols")
            lines.append("")
            for symbol in exported:
                obj = getattr(module, symbol, None)
                if obj is None:
                    lines.append(f"- `{symbol}`")
                    continue
                obj_type = "function" if inspect.isfunction(obj) else "class" if inspect.isclass(obj) else type(obj).__name__
                lines.append(f"- `{symbol}` (`{obj_type}`, from `{getattr(obj, '__module__', 'unknown')}`)")
            lines.append("")

    classes, functions = _iter_public_members(module_name)
    if classes:
        lines.append("## Classes")
        lines.append("")
        for name, obj in classes:
            lines.extend(_render_class(name, obj))
    if functions:
        lines.append("## Functions")
        lines.append("")
        for name, obj in functions:
            lines.extend(_render_function(name, obj))
    if not classes and not functions and module_name != "bnrpe":
        lines.append("No public classes or functions found.")
        lines.append("")
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_index(destination: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# API Reference",
        "",
        "This section is auto-generated from the Python package source.",
        "",
        f"_Last generated: {now}_",
        "",
        "## Modules",
        "",
    ]
    for module_name, page_name in MODULE_PAGES:
        lines.append(f"- [`{module_name}`]({page_name})")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for module_name, page_name in MODULE_PAGES:
        _write_module_page(module_name, output_dir / page_name)
    _write_index(output_dir / "index.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "api",
        help="Directory where API Markdown files are written.",
    )
    args = parser.parse_args()
    generate(args.output_dir)
    print(f"Generated API docs in {args.output_dir}")


if __name__ == "__main__":
    main()
