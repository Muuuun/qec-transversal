"""Command-line interface.

One subcommand per physical gate ansatz, mirroring
:mod:`qec_transversal.api`, plus the code-registry utilities.  Every gate
subcommand emits the same JSON envelope -- ``method``, ``ansatz``,
``completeness``, group orders, certificate, metadata -- so results from
different backends can be compared and diffed without special-casing.

Input is either a registry name (``--code steane``) or a JSON file holding
``H_X``/``H_Z`` (a CSS code) or ``H`` (symplectic ``[X | Z]`` rows for a
general stabilizer code).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .api import (
    diagonal_transversal_gates,
    monomial_clifford_group,
    one_block_clifford_group,
    partition_clifford_group,
    permutation_automorphism_group,
    strict_transversal_clifford,
)
from .codes.css import CSSCode
from .codes.registry import REGISTRY
from .codes.stabilizer import StabilizerCode
from .logical.synthesis import verify_logical_gate


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
        try:
            matrix = np.asarray(value, dtype=np.uint8)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name} entries must be 0 or 1: {error}") from error
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a list of equal-length rows")
        width = int(matrix.shape[1])
    if n is not None and width != n:
        raise ValueError(f"{name} has width {width}, expected n={n}")
    return matrix, width


def _load_code(path: Path) -> CSSCode | StabilizerCode:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("input JSON must be an object")
    if "H" in document:
        rows, _ = _parse_rows(document["H"], name="H", n=None)
        return StabilizerCode(rows)
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


def _registry_code(name: str) -> CSSCode:
    entry = REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"unknown code {name!r}; run 'qec-transversal list-codes' for the registry"
        )
    h_x, h_z = entry.build()
    return CSSCode(h_x, h_z)


def _code_document(name: str) -> dict[str, Any]:
    entry = REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"unknown code {name!r}; run 'qec-transversal list-codes' for the registry"
        )
    h_x, h_z = entry.build()
    return {
        "name": entry.name,
        "family": entry.family,
        "n": entry.n,
        "k": entry.k,
        "d": entry.d,
        "d_is_upper_bound": entry.d_is_upper_bound,
        "source": entry.source,
        "H_X": ["".join(str(bit) for bit in row) for row in h_x.tolist()],
        "H_Z": ["".join(str(bit) for bit in row) for row in h_z.tolist()],
    }


def _resolve(args) -> CSSCode | StabilizerCode:
    if (getattr(args, "input", None) is None) == (getattr(args, "code", None) is None):
        raise ValueError("provide exactly one of an input file or --code NAME")
    return _registry_code(args.code) if args.code else _load_code(args.input)


def _require_css(code: CSSCode | StabilizerCode) -> CSSCode:
    if not isinstance(code, CSSCode):
        raise ValueError("this subcommand needs a CSS code (H_X and H_Z)")
    return code


def _parse_partition(text: str, n: int) -> list[tuple[int, ...]]:
    """``"0,1;2,3;4"`` -> ``[(0, 1), (2, 3), (4,)]``; ``"pairs"`` / ``"singletons"``."""

    if text == "singletons":
        return [(q,) for q in range(n)]
    if text == "pairs":
        if n % 2:
            raise ValueError("the 'pairs' partition needs an even qubit count")
        return [(2 * i, 2 * i + 1) for i in range(n // 2)]
    cells = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        cells.append(tuple(int(part) for part in chunk.split(",")))
    return cells


def _code_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, nargs="?", help="JSON file with H_X/H_Z or H")
    parser.add_argument("--code", help="use a built-in code by registry name")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qec-transversal",
        description=(
            "Exact, certified analysis of depth-one code-preserving gates of "
            "stabilizer quantum codes."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="CSS strict-transversal report (the detailed 0.1-compatible format)",
    )
    _code_input(analyze)
    analyze.add_argument(
        "--group-cap",
        type=int,
        default=100_000,
        help="maximum logical matrices enumerated by the fallback exact closure",
    )
    analyze.add_argument("--include-constraints", action="store_true")
    analyze.add_argument("--include-physical", action="store_true")

    strict = subparsers.add_parser(
        "strict", help="one arbitrary single-qubit Clifford per qubit (CSS or non-CSS)"
    )
    _code_input(strict)
    strict.add_argument(
        "--method", choices=["auto", "css", "general"], default="auto",
        help="specialised CSS shear solver, general preservation algebra, or auto",
    )

    partition = subparsers.add_parser(
        "partition", help="one arbitrary Clifford per prescribed partition cell"
    )
    _code_input(partition)
    partition.add_argument(
        "--cells", default="pairs",
        help="'singletons', 'pairs', or an explicit '0,1;2,3;4' partition",
    )
    partition.add_argument(
        "--method", choices=["auto", "enumeration", "structure"], default="auto"
    )

    diagonal = subparsers.add_parser(
        "diagonal", help="strict diagonal gates in the Clifford hierarchy (CSS)"
    )
    _code_input(diagonal)
    diagonal.add_argument("--level", type=int, default=3)
    diagonal.add_argument("--family", choices=["Z", "X"], default="Z")

    monomial = subparsers.add_parser(
        "monomial", help="qubit permutation x local Clifford (needs python-igraph)"
    )
    _code_input(monomial)

    automorphisms = subparsers.add_parser(
        "automorphisms", help="SWAP-class permutation gates (needs python-igraph)"
    )
    _code_input(automorphisms)
    automorphisms.add_argument(
        "--method", choices=["codewords", "tanner"], default="codewords"
    )

    one_block = subparsers.add_parser(
        "one-block", help="the group generated by all depth-one one-block layers"
    )
    _code_input(one_block)
    one_block.add_argument("--involution-cap", type=int, default=16)
    one_block.add_argument("--time-budget", type=float, default=120.0)

    generate = subparsers.add_parser(
        "generate", help="emit the check matrices of a built-in code as JSON"
    )
    generate.add_argument("name", help="registry name, see list-codes")
    generate.add_argument("-o", "--output", type=Path)
    generate.add_argument("--compact", action="store_true")

    subparsers.add_parser("list-codes", help="list the built-in code registry")

    verify = subparsers.add_parser(
        "verify",
        help="decide whether a logical gate has a strict-transversal implementation",
    )
    verify.add_argument("--code", required=True, help="registry name")
    verify.add_argument("gate", help="S | SQRT_X | H | CZ | CNOT | SWAP")
    verify.add_argument("qubits", type=int, nargs="*", help="logical qubit indices")
    verify.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    indent = None if getattr(args, "compact", False) else 2
    try:
        if args.command == "analyze":
            code = _require_css(_resolve(args))
            result = code.analyze_transversal().to_dict(
                group_cap=args.group_cap,
                include_constraints=args.include_constraints,
                include_physical=args.include_physical,
            )
            _write_json(result, args.output, indent=indent)
            return 0
        if args.command == "strict":
            result = strict_transversal_clifford(_resolve(args), method=args.method)
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "partition":
            code = _resolve(args)
            cells = _parse_partition(args.cells, code.n)
            result = partition_clifford_group(code, cells, method=args.method)
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "diagonal":
            code = _require_css(_resolve(args))
            result = diagonal_transversal_gates(code, level=args.level, family=args.family)
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "monomial":
            result = monomial_clifford_group(_resolve(args))
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "automorphisms":
            code = _require_css(_resolve(args))
            result = permutation_automorphism_group(code, method=args.method)
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "one-block":
            code = _require_css(_resolve(args))
            result = one_block_clifford_group(
                code,
                name=args.code,
                involution_cap=args.involution_cap,
                time_budget_s=args.time_budget,
            )
            _write_json(result.to_dict(), args.output, indent=indent)
            return 0
        if args.command == "generate":
            _write_json(_code_document(args.name), args.output, indent=indent)
            return 0
        if args.command == "verify":
            code = _registry_code(args.code)
            result = verify_logical_gate(code, args.gate, *args.qubits)
            _write_json(result.to_dict(), None, indent=indent)
            return 0 if result.found else 1
        if args.command == "list-codes":
            for entry in REGISTRY.values():
                distance = "?" if entry.d is None else str(entry.d)
                if entry.d_is_upper_bound:
                    distance = "<=" + distance
                print(
                    f"{entry.name:22s} [[{entry.n},{entry.k},{distance}]]"
                    f"  {entry.family:20s} {entry.source}"
                )
            return 0
    except (OSError, ValueError, ImportError, json.JSONDecodeError) as error:
        print(f"qec-transversal: error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
