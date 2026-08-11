"""Machine-checkable witness export for strict-transversal verdicts.

A witness file contains everything an *independent* checker needs to verify
a code's strict-transversal census entry without trusting this library:

- the check matrices themselves;
- for each parameter-space constraint row, its provenance ``(check index,
  dual vector)`` — the checker re-derives the row as ``dual & check`` and
  verifies the dual annihilates the opposite check matrix;
- the kernel basis, whose membership and completeness the checker verifies
  with its own elimination (``rank(constraints) + dim(kernel) = n``);
- the paired logical basis with its pairing and commutation facts;
- every generator's claimed logical action, recomputable from the basis;
- the logical group as an explicit element list, checkable by closure.

The independent checker lives in ``tools/check_witness.py`` and imports
nothing from this package.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import numpy as np

from .css import CSSCode
from .gf2 import nullspace, row_basis


def _bits(matrix: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in np.atleast_2d(matrix)]


def _constraints_with_provenance(
    source: np.ndarray, target: np.ndarray
) -> list[dict[str, Any]]:
    """Regenerate the constraint system with per-row provenance.

    For each check, the restriction of ``nullspace(target)`` to its support
    is row-reduced with an augmented identity so every reduced row carries
    the combination that produced it; lifting that combination gives a full
    dual vector ``w`` with ``target @ w = 0`` and ``w & check = row``.  The
    checker's completeness argument needs only that every row is such a
    consequence and that ``rank(rows) + dim(kernel) = n``.
    """

    n = source.shape[1]
    target_perp = nullspace(target)
    records: list[dict[str, Any]] = []
    for index, check in enumerate(source):
        support = np.flatnonzero(check)
        if support.size == 0 or target_perp.shape[0] == 0:
            continue
        restricted = target_perp[:, support]
        augmented = np.hstack(
            [restricted, np.eye(target_perp.shape[0], dtype=np.uint8)]
        ).astype(np.uint8)
        reduced = augmented.copy()
        rows_, width = reduced.shape
        r = 0
        for c in range(support.size):
            hit = np.flatnonzero(reduced[r:, c])
            if hit.size == 0:
                continue
            sel = r + int(hit[0])
            if sel != r:
                reduced[[r, sel]] = reduced[[sel, r]]
            others = np.flatnonzero(reduced[:, c])
            others = others[others != r]
            if others.size:
                reduced[others] ^= reduced[r]
            r += 1
            if r == rows_:
                break
        for i in range(r):
            local = reduced[i, : support.size]
            if not local.any():
                continue
            combination = reduced[i, support.size :]
            dual = (combination @ target_perp) % 2
            row = (dual & check) % 2
            records.append(
                {"row": _bits(row)[0], "check_index": int(index), "dual": _bits(dual)[0]}
            )
    return records


def export_strict_witness(code: CSSCode, name: str = "") -> dict[str, Any]:
    """Complete strict-transversal witness for one CSS code."""

    analysis = code.analyze_transversal()

    def family_block(space, source, target):
        return {
            "constraints": _constraints_with_provenance(source, target),
            "kernel": _bits(space.basis) if space.dimension else [],
        }

    generators = []
    nontrivial_logicals = []
    for g in analysis.generators:
        record = {
            "family": g.family,
            "parameter": _bits(g.parameter)[0],
            "logical": _bits(g.logical_symplectic) if code.k else [],
        }
        generators.append(record)
        if not g.is_logical_identity:
            nontrivial_logicals.append(g.logical_symplectic)

    # explicit group closure (strict logical groups are small)
    identity = np.eye(2 * code.k, dtype=np.uint8)
    elements = {identity.tobytes(): identity}
    queue = [identity]
    while queue:
        element = queue.pop()
        for g in nontrivial_logicals:
            product = (element @ g) % 2
            key = product.tobytes()
            if key not in elements:
                elements[key] = product
                queue.append(product)
        if len(elements) > 100_000:
            raise AssertionError("strict logical group unexpectedly large for a witness")

    return {
        "schema": "qec-transversal-strict-witness/1",
        "name": name,
        "n": code.n,
        "k": code.k,
        "H_X": _bits(code.h_x),
        "H_Z": _bits(code.h_z),
        "logical_X": _bits(code.logical_x) if code.k else [],
        "logical_Z": _bits(code.logical_z) if code.k else [],
        "A_Z": family_block(analysis.a_z, code.h_x, code.h_z),
        "A_X": family_block(analysis.a_x, code.h_z, code.h_x),
        "generators": generators,
        "logical_group": {
            "order": len(elements),
            "elements": [_bits(m) for m in elements.values()] if code.k else [],
        },
    }


def write_witness(document: dict[str, Any], path: str) -> None:
    payload = json.dumps(document, separators=(",", ":")).encode()
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
    else:
        with open(path, "wb") as handle:
            handle.write(payload)
