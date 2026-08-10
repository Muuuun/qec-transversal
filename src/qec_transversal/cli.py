"""Command-line interface for CSS transversal-gate analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .css import CSSCode


def _parse_rows(value: object, *, name: str, n: int | None) -> tuple[np.ndarray, int | None]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    if not value:
        if n is None:
            return np.zeros((0, 0), dtype=np.uint8), None
        return np.zeros((0, n), dtype=np.uint8), n
    if all(isinstance(row, str) for row in value):
        widths = {len(row) for row in value}
        if len(widths) != 1:
            raise ValueError(f"{name} bit strings have inconsistent lengths")
        width = widths.pop()
        if any(set(row) - {"0", "1"} for row in value):
            raise ValueError(f"{name} bit strings may contain only 0 and 1")
        matrix = np.asarray([[int(bit) for bit in row] for row in value], dtype=np.uint8)
    else:
        matrix = np.asarray(value, dtype=np.uint8)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a list of equal-length rows")
        width = int(matrix.shape[1])
    if n is not None and width != n:
        raise ValueError(f"{name} has width {width}, expected n={n}")
    return matrix, width


def _load_code(path: Path) -> CSSCode:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("input JSON must be an object")
    n_value = document.get("n")
    n = int(n_value) if n_value is not None else None
    h_x, width_x = _parse_rows(document.get("H_X", document.get("hx", [])), name="H_X", n=n)
    inferred = n if n is not None else width_x
    h_z, width_z = _parse_rows(
        document.get("H_Z", document.get("hz", [])), name="H_Z", n=inferred
    )
    final_n = inferred if inferred is not None else width_z
    if final_n is None:
        raise ValueError("provide n when both H_X and H_Z are empty")
    if h_x.shape[1] == 0:
        h_x = np.zeros((0, final_n), dtype=np.uint8)
    return CSSCode(h_x, h_z, n=final_n)


def _write_json(result: dict[str, Any], output: Path | None, *, indent: int | None) -> None:
    rendered = json.dumps(result, indent=indent, sort_keys=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qec-transversal",
        description="Find strict-transversal logical Clifford generators of a CSS code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze H_X and H_Z from a JSON file")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("-o", "--output", type=Path)
    analyze.add_argument(
        "--group-cap",
        type=int,
        default=100_000,
        help="maximum logical matrices enumerated by the built-in exact closure",
    )
    analyze.add_argument("--include-constraints", action="store_true")
    analyze.add_argument("--include-physical", action="store_true")
    analyze.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "analyze":
            code = _load_code(args.input)
            result = code.analyze_transversal().to_dict(
                group_cap=args.group_cap,
                include_constraints=args.include_constraints,
                include_physical=args.include_physical,
            )
            _write_json(result, args.output, indent=None if args.compact else 2)
            return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qec-transversal: error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

