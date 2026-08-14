#!/usr/bin/env python3
"""Independent checker for qec-transversal stabilizer (non-CSS) witnesses.

Deliberately self-contained: numpy only, no imports from qec_transversal.
Everything is re-verified from first principles with this file's own
Gaussian elimination, so trusting a verdict requires auditing only this
script and the witness file (schema qec-transversal-stabilizer-witness/1).

Usage:  python tools/check_stabilizer_witness.py WITNESS.json[.gz] [...]
Exit 0 iff every file passes every check.

Verified facts per witness:
 1. Stabilizer validity: H is symplectically self-orthogonal and
    k = n - rank(H).
 2. Logical basis: rows commute with H, are independent of it, and pair
    canonically (symplectic Gram matrix [[0,I],[I,0]]).
 3. Constraint provenance: every constraint row equals the bilinear form
    row[0::4]=wx&x, row[1::4]=wz&x, row[2::4]=wx&z, row[3::4]=wz&z built
    from (H[stab_index], dual) with H dual = 0 — so every constraint is a
    linear consequence of "H M lies in rowspan(H)".
 4. Algebra soundness: each basis element's block-action matrix maps every
    stabilizer row into rowspan(H) (definition re-checked directly), and
    pairwise blockwise products stay in the span (closure).
 5. Algebra completeness: basis rows independent, annihilating the
    constraints, with rank(constraints) + dim(basis) = 4n — with (3) this
    proves no strict-transversal solution is missing from the span.
 6. Group soundness: elements distinct, identity present, each in the
    basis span with every 2x2 block of determinant one, closed under
    blockwise products; every non-identity element carries a generator
    record whose logical action is recomputed via symplectic products
    against the logical basis (residue confirmed inside rowspan(H)).
 7. Group completeness: for dim(basis) <= 20 the checker re-enumerates
    all 2^dim algebra elements itself and matches the unit set exactly;
    beyond that the verdict is "sound, completeness not re-verified".
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np

SCHEMA = "qec-transversal-stabilizer-witness/1"
SWEEP_DIM_CAP = 20


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


def symp(left, right, n):
    """Matrix of symplectic products <left_i, right_j> in the (X|Z) layout."""
    swapped = np.hstack([left[:, n:], left[:, :n]])
    return (swapped @ right.T) % 2


def block_action(entries, n):
    """The 2n x 2n row-action matrix of per-qubit blocks (a, b, c, d)."""
    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    idx = np.arange(n)
    matrix[idx, idx] = entries[0::4]
    matrix[idx, n + idx] = entries[1::4]
    matrix[n + idx, idx] = entries[2::4]
    matrix[n + idx, n + idx] = entries[3::4]
    return matrix


def block_multiply(left, right, n):
    """Blockwise product of two flat (a, b, c, d)-per-qubit entry vectors."""
    blocks = (left.reshape(n, 2, 2) @ right.reshape(n, 2, 2)) % 2
    return blocks.reshape(-1).astype(np.uint8)


def block_determinants(entries):
    a, b, c, d = entries[0::4], entries[1::4], entries[2::4], entries[3::4]
    return (a & d) ^ (b & c)


def check_document(doc):
    errors = []
    caveats = []
    if doc.get("schema") != SCHEMA:
        return [f"unexpected schema {doc.get('schema')!r}"], caveats
    n, k = doc["n"], doc["k"]
    h = bits(doc["H"], 2 * n)
    if symp(h, h, n).any():
        errors.append("H not symplectically self-orthogonal")
    if n - rank(h) != k:
        errors.append("k does not match rank(H)")

    logical = bits(doc["logical"], 2 * n)
    if k:
        if logical.shape[0] != 2 * k:
            errors.append("logical basis has wrong row count")
        eye = np.eye(k, dtype=np.uint8)
        zero = np.zeros((k, k), dtype=np.uint8)
        canonical = np.block([[zero, eye], [eye, zero]]).astype(np.uint8)
        if not np.array_equal(symp(logical, logical, n), canonical):
            errors.append("logical pairing is not canonical")
        if symp(h, logical, n).any():
            errors.append("logical rows do not commute with H")
        if rank(np.vstack([h, logical])) != rank(h) + 2 * k:
            errors.append("logical rows not independent of stabilizers")

    constraint_rows = []
    for record in doc["algebra"]["constraints"]:
        row = bits([record["row"]], 4 * n)[0]
        dual = bits([record["dual"]], 2 * n)[0]
        if not 0 <= record["stab_index"] < h.shape[0]:
            errors.append("constraint stab_index out of range")
            continue
        stab = h[record["stab_index"]]
        if ((h @ dual) % 2).any():
            errors.append("constraint dual not orthogonal to H")
        x, z = stab[:n], stab[n:]
        wx, wz = dual[:n], dual[n:]
        rebuilt = np.zeros(4 * n, dtype=np.uint8)
        rebuilt[0::4] = wx & x
        rebuilt[1::4] = wz & x
        rebuilt[2::4] = wx & z
        rebuilt[3::4] = wz & z
        if not np.array_equal(rebuilt, row):
            errors.append("constraint row does not match its bilinear formula")
        constraint_rows.append(row)
    constraints = (
        np.array(constraint_rows, dtype=np.uint8)
        if constraint_rows
        else np.zeros((0, 4 * n), dtype=np.uint8)
    )

    basis = bits(doc["algebra"]["basis"], 4 * n)
    for element in basis:
        matrix = block_action(element, n)
        for s in h:
            if not in_rowspan((s @ matrix) % 2, h):
                errors.append("algebra basis element fails the definition")
                break
        if ((constraints @ element) % 2).any():
            errors.append("algebra basis element violates a constraint")
    if basis.shape[0] and rank(basis) != basis.shape[0]:
        errors.append("algebra basis rows dependent")
    if rank(constraints) + basis.shape[0] != 4 * n:
        errors.append("algebra completeness arithmetic fails")
    for left in basis:
        for right in basis:
            if not in_rowspan(block_multiply(left, right, n), basis):
                errors.append("algebra not closed under products")
                break

    group = doc["group"]
    if group.get("route") != "enumeration":
        errors.append("unsupported group route")
    elements = [row for row in bits(group["elements"], 4 * n)]
    keys = {element.tobytes() for element in elements}
    if len(keys) != len(elements):
        errors.append("group elements not distinct")
    identity = np.zeros(4 * n, dtype=np.uint8)
    identity[0::4] = 1
    identity[3::4] = 1
    if identity.tobytes() not in keys:
        errors.append("identity missing from group")
    for element in elements:
        if not in_rowspan(element, basis):
            errors.append("group element outside the algebra span")
        if not block_determinants(element).all():
            errors.append("group element has a non-invertible block")
    for left in elements:
        for right in elements:
            if block_multiply(left, right, n).tobytes() not in keys:
                errors.append("group not closed")
                break
    if len(elements) != group["order"]:
        errors.append("order mismatch")

    generator_keys = set()
    for g in doc["generators"]:
        entries = bits([g["entries"]], 4 * n)[0]
        generator_keys.add(entries.tobytes())
        if entries.tobytes() not in keys:
            errors.append("generator missing from group elements")
        if k:
            images = (logical @ block_action(entries, n)) % 2
            x_coeff = symp(images, logical[k:], n)
            z_coeff = symp(images, logical[:k], n)
            expected = np.hstack([x_coeff, z_coeff]).astype(np.uint8)
            if not np.array_equal(bits(g["logical"], 2 * k), expected):
                errors.append("generator logical action mismatch")
            for residue in (images ^ (expected @ logical) % 2) % 2:
                if not in_rowspan(residue, h):
                    errors.append("generator logical residue outside the stabilizer")
                    break
    if keys - generator_keys - {identity.tobytes()}:
        errors.append("non-identity group element lacks a generator record")

    dim = basis.shape[0]
    if dim <= SWEEP_DIM_CAP:
        sweep = set()
        for mask in range(1 << dim):
            entries = np.zeros(4 * n, dtype=np.uint8)
            for bit in range(dim):
                if (mask >> bit) & 1:
                    entries ^= basis[bit]
            if block_determinants(entries).all():
                sweep.add(entries.tobytes())
        if sweep != keys:
            errors.append("group element list does not match the checker's sweep")
    else:
        caveats.append("sound, completeness not re-verified")
    return errors, caveats


def main(paths):
    failed = False
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as handle:
            doc = json.loads(handle.read())
        errors, caveats = check_document(doc)
        label = doc.get("name") or path
        if errors:
            failed = True
            print(f"FAIL {label}: " + "; ".join(sorted(set(errors))))
        else:
            note = f" ({'; '.join(caveats)})" if caveats else ""
            print(f"PASS {label} [[{doc['n']},{doc['k']}]] "
                  f"dim algebra={len(doc['algebra']['basis'])} "
                  f"group order={doc['group']['order']}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
