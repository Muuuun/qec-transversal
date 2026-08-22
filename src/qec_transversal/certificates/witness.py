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

Two witness flavours are exported: the CSS strict witness
(``qec-transversal-strict-witness/1``) and the general stabilizer witness
(``qec-transversal-stabilizer-witness/1``) covering the non-CSS engine.
The independent checkers live in ``tools/check_witness.py`` and
``tools/check_stabilizer_witness.py`` and import nothing from this package.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import numpy as np

from ..ansatz.strict import LocalCliffordAnalysis
from ..codes.css import CSSCode
from ..codes.stabilizer import StabilizerCode
from ..utils.gf2 import nullspace


def _bits(matrix: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in np.atleast_2d(matrix)]


def _augmented_row_reduce(matrix: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row-reduce ``matrix`` with an augmented identity block.

    Returns ``(row, combination)`` pairs, one per nonzero reduced row, with
    ``combination @ matrix % 2 == row`` — the provenance a checker needs to
    certify that every reduced row is a combination of the input rows.
    """

    rows_, cols = matrix.shape
    reduced = np.hstack([matrix, np.eye(rows_, dtype=np.uint8)]).astype(np.uint8)
    r = 0
    for c in range(cols):
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
    return [
        (reduced[i, :cols], reduced[i, cols:])
        for i in range(r)
        if reduced[i, :cols].any()
    ]


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

    target_perp = nullspace(target)
    records: list[dict[str, Any]] = []
    for index, check in enumerate(source):
        support = np.flatnonzero(check)
        if support.size == 0 or target_perp.shape[0] == 0:
            continue
        for _local, combination in _augmented_row_reduce(target_perp[:, support]):
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


def _stabilizer_constraints_with_provenance(
    code: StabilizerCode,
) -> list[dict[str, Any]]:
    """Regenerate the local-Clifford constraint system with provenance.

    Mirrors :func:`qec_transversal.stabilizer.local_clifford_algebra`: a
    stabilizer row ``s = (x|z)`` maps under block entries ``(a_i, b_i, c_i,
    d_i)`` to ``s'`` with ``x'_i = a_i x_i + c_i z_i`` and ``z'_i = b_i x_i +
    d_i z_i``, and membership ``s' in rowspan(H)`` is equivalent to ``<w, s'>
    = 0`` for every dual ``w = (wx|wz)`` with ``H w = 0``.  Expanded per
    qubit, that inner product is the bilinear constraint row

        ``row[0::4] = wx & x``, ``row[1::4] = wz & x``,
        ``row[2::4] = wx & z``, ``row[3::4] = wz & z``.

    Duals are drawn from ``nullspace(H)`` restricted to the support columns
    of ``s`` (the LDPC-friendly construction of the solver) and lifted with
    the augmented-identity technique, so every constraint row ships the full
    dual vector as its provenance and the checker can re-derive it.
    """

    n = code.n
    perp = nullspace(code.h)
    records: list[dict[str, Any]] = []
    for index, s in enumerate(code.h):
        x, z = s[:n], s[n:]
        support = np.flatnonzero(x | z)
        if support.size == 0 or perp.shape[0] == 0:
            continue
        columns = np.concatenate([support, n + support])
        for _local, combination in _augmented_row_reduce(perp[:, columns]):
            dual = (combination @ perp) % 2
            wx, wz = dual[:n], dual[n:]
            row = np.zeros(4 * n, dtype=np.uint8)
            row[0::4] = wx & x
            row[1::4] = wz & x
            row[2::4] = wx & z
            row[3::4] = wz & z
            if not row.any():
                continue
            records.append(
                {"row": _bits(row)[0], "stab_index": int(index), "dual": _bits(dual)[0]}
            )
    return records


def export_stabilizer_witness(
    code: StabilizerCode, analysis: LocalCliffordAnalysis, name: str = ""
) -> dict[str, Any]:
    """Complete strict-transversal witness for a general stabilizer code.

    Tier A of the exportable-certificate plan: only the *enumeration* route
    is supported (the structured Wedderburn route is a filed follow-up), and
    the independent checker in ``tools/check_stabilizer_witness.py``
    re-verifies, with numpy alone:

    - soundness — every algebra basis element's block-action matrix maps
      each stabilizer row into ``rowspan(H)``, and every group element lies
      in the basis span with all per-qubit blocks of determinant one;
    - algebra completeness — each constraint row is a certified linear
      consequence of the definition (its dual annihilates ``H``), the basis
      rows are independent, and ``rank(constraints) + dim(basis) = 4n``, so
      the span of the shipped basis is exactly the solution algebra;
    - group completeness — re-enumerated by the checker's own ``2^dim``
      sweep for small algebra dimensions;
    - logical actions — recomputed from the symplectic pairing with the
      exported logical basis.
    """

    if analysis.structured_order is not None or not analysis.enumeration_complete:
        raise ValueError(
            "stabilizer witness export requires a completed enumeration; "
            "the structured-route certificate is a separate follow-up"
        )
    if len(analysis.elements) > 100_000:
        raise AssertionError("strict transversal group unexpectedly large for a witness")

    generators = []
    for g in analysis.generators:
        entries = np.zeros(4 * code.n, dtype=np.uint8)
        for i, block in enumerate(g.blocks):
            entries[4 * i : 4 * i + 4] = block
        generators.append(
            {
                "entries": _bits(entries)[0],
                "logical": _bits(g.logical_symplectic) if code.k else [],
            }
        )

    return {
        "schema": "qec-transversal-stabilizer-witness/1",
        "name": name,
        "n": code.n,
        "k": code.k,
        "H": _bits(code.h),
        "logical": _bits(code.logical) if code.k else [],
        "algebra": {
            "constraints": _stabilizer_constraints_with_provenance(code),
            "basis": _bits(analysis.algebra) if analysis.algebra.shape[0] else [],
        },
        "generators": generators,
        "group": {
            "route": "enumeration",
            "order": len(analysis.elements),
            "elements": [_bits(entries)[0] for entries in analysis.elements],
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
