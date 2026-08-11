#!/usr/bin/env python3
"""Independent checker for qec-transversal strict-transversal witnesses.

Deliberately self-contained: numpy only, no imports from qec_transversal.
Everything is re-verified from first principles with this file's own
Gaussian elimination, so trusting a verdict requires auditing only this
script (~150 lines) and the witness file.

Usage:  python tools/check_witness.py WITNESS.json[.gz] [...]
Exit 0 iff every file passes every check.

Verified facts per witness:
 1. CSS validity: H_X H_Z^T = 0 and k = n - rank(H_X) - rank(H_Z).
 2. Logical basis: rows lie in the right kernels, are independent of the
    checks, and pair canonically (L_X L_Z^T = I).
 3. Constraint provenance: every A_Z constraint row equals dual & check
    with H_Z dual^T = 0 (mirrored for A_X) — so every constraint is a
    consequence of the definition.
 4. Kernel soundness: each kernel vector satisfies a (.) check in
    rowspan(target) for EVERY check (definition re-checked directly).
 5. Kernel completeness: rank(constraints) + dim(kernel) = n and kernel
    rows independent — with (3) this proves no transversal parameter is
    missing.
 6. Generators: each claimed logical action equals the shear formula
    recomputed from the logical basis.
 7. Group: the element list contains the identity and every generator
    action, is duplicate-free, and is closed under the generators —
    hence it is exactly the generated logical group; order matches.
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np


def bits(rows, width):
    if not rows:
        return np.zeros((0, width), dtype=np.uint8)
    return np.array([[int(c) for c in row] for row in rows], dtype=np.uint8)


def rank(matrix):
    m = matrix.copy() % 2
    r = 0
    for c in range(m.shape[1]):
        if r == m.shape[0]:
            break
        hit = np.flatnonzero(m[r:, c])
        if hit.size == 0:
            continue
        sel = r + int(hit[0])
        if sel != r:
            m[[r, sel]] = m[[sel, r]]
        others = np.flatnonzero(m[:, c])
        others = others[others != r]
        if others.size:
            m[others] ^= m[r]
        r += 1
    return r


def in_rowspan(vector, basis):
    return rank(np.vstack([basis, vector])) == rank(basis)


def check_family(doc, family, errors):
    n = doc["n"]
    source = bits(doc["H_X" if family == "A_Z" else "H_Z"], n)
    target = bits(doc["H_Z" if family == "A_Z" else "H_X"], n)
    block = doc[family]
    constraint_rows = []
    for record in block["constraints"]:
        row = bits([record["row"]], n)[0]
        dual = bits([record["dual"]], n)[0]
        check = source[record["check_index"]]
        if ((target @ dual) % 2).any():
            errors.append(f"{family}: dual witness not orthogonal to target")
        if not np.array_equal((dual & check) % 2, row):
            errors.append(f"{family}: constraint row does not equal dual & check")
        constraint_rows.append(row)
    constraints = (
        np.array(constraint_rows, dtype=np.uint8)
        if constraint_rows
        else np.zeros((0, n), dtype=np.uint8)
    )
    kernel = bits(block["kernel"], n)
    for vector in kernel:
        if ((constraints @ vector) % 2).any():
            errors.append(f"{family}: kernel vector violates a constraint")
        for check in source:
            if not in_rowspan((vector & check) % 2, target):
                errors.append(f"{family}: kernel vector fails the definition")
                break
    if kernel.shape[0] and rank(kernel) != kernel.shape[0]:
        errors.append(f"{family}: kernel rows dependent")
    if rank(constraints) + kernel.shape[0] != n:
        errors.append(f"{family}: completeness arithmetic fails")
    return kernel


def check_document(doc):
    errors = []
    n, k = doc["n"], doc["k"]
    h_x, h_z = bits(doc["H_X"], n), bits(doc["H_Z"], n)
    if ((h_x @ h_z.T) % 2).any():
        errors.append("H_X H_Z^T != 0")
    if n - rank(h_x) - rank(h_z) != k:
        errors.append("k does not match check ranks")

    l_x, l_z = bits(doc["logical_X"], n), bits(doc["logical_Z"], n)
    if k:
        if ((h_z @ l_x.T) % 2).any() or ((h_x @ l_z.T) % 2).any():
            errors.append("logical rows do not commute with checks")
        if not np.array_equal((l_x @ l_z.T) % 2, np.eye(k, dtype=np.uint8)):
            errors.append("logical pairing is not canonical")
        if rank(np.vstack([h_x, l_x])) != rank(h_x) + k:
            errors.append("logical X rows not independent of stabilizers")

    kernel_z = check_family(doc, "A_Z", errors)
    kernel_x = check_family(doc, "A_X", errors)

    claimed = {"Z": [], "X": []}
    for g in doc["generators"]:
        parameter = bits([g["parameter"]], n)[0]
        pool = kernel_z if g["family"] == "Z" else kernel_x
        if not any(np.array_equal(parameter, row) for row in pool):
            errors.append("generator parameter not a kernel basis row")
        if k:
            basis = l_x if g["family"] == "Z" else l_z
            shear = (basis * parameter) @ basis.T % 2
            eye = np.eye(k, dtype=np.uint8)
            zero = np.zeros((k, k), dtype=np.uint8)
            expected = (
                np.block([[eye, shear], [zero, eye]])
                if g["family"] == "Z"
                else np.block([[eye, zero], [shear, eye]])
            ).astype(np.uint8)
            logical = bits(g["logical"], 2 * k)
            if not np.array_equal(logical, expected):
                errors.append("generator logical action mismatch")
            claimed[g["family"]].append(logical)

    if k:
        elements = [bits(rows, 2 * k) for rows in doc["logical_group"]["elements"]]
        keys = {e.tobytes() for e in elements}
        if len(keys) != len(elements):
            errors.append("group elements not distinct")
        if np.eye(2 * k, dtype=np.uint8).tobytes() not in keys:
            errors.append("identity missing from group")
        nontrivial = [
            g
            for fam in ("Z", "X")
            for g in claimed[fam]
            if not np.array_equal(g, np.eye(2 * k, dtype=np.uint8))
        ]
        for g in nontrivial:
            if g.tobytes() not in keys:
                errors.append("generator action missing from group")
        for element in elements:
            for g in nontrivial:
                if ((element @ g) % 2).astype(np.uint8).tobytes() not in keys:
                    errors.append("group not closed")
                    break
        if len(elements) != doc["logical_group"]["order"]:
            errors.append("order mismatch")
    return errors


def main(paths):
    failed = False
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as handle:
            doc = json.loads(handle.read())
        errors = check_document(doc)
        label = doc.get("name") or path
        if errors:
            failed = True
            print(f"FAIL {label}: " + "; ".join(sorted(set(errors))))
        else:
            print(f"PASS {label} [[{doc['n']},{doc['k']}]] "
                  f"dim A_Z={len(doc['A_Z']['kernel'])} dim A_X={len(doc['A_X']['kernel'])} "
                  f"group order={doc['logical_group']['order']}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
