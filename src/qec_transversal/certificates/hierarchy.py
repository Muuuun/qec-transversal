"""Smith-form completeness certificates for ``Z_{2^L}`` constraint modules.

A kernel generator list only proves *soundness* (each generator satisfies the
congruences).  The certificate here proves *completeness*: the exported
``(V, exponents)`` pair lets an independent checker verify that the returned
generators span the entire kernel, via divisibility of the transformed columns
and mod-2 independence of their cofactors.
"""

from __future__ import annotations

import numpy as np

from ..utils.gf2 import rank as f2_rank
from ..utils.modular import _valuation


def smith_kernel_certificate(
    constraints: list[tuple[np.ndarray, int]], n: int, *, exponent: int = 3
) -> dict:
    """Kernel-completeness certificate for the constraint module.

    Computes a unimodular column transform ``V`` and diagonal exponents
    ``a_i`` with ``A V = U^{-1} diag(2^{a_i})`` for some unimodular ``U``.
    The exported certificate is only ``(V, exponents)``: an independent
    checker recomputes ``M = A V``, verifies column ``i`` is divisible by
    ``2^{a_i}`` and that the cofactor columns ``M_i / 2^{a_i}`` are linearly
    independent mod 2 — which forces the kernel to be exactly
    ``span{ 2^{L-a_i} V e_i }``, certifying completeness rather than mere
    soundness.
    """

    modulus = 1 << exponent
    if constraints:
        A = np.array(
            [(factor * pattern) % modulus for pattern, factor in constraints],
            dtype=np.int64,
        )
    else:
        A = np.zeros((0, n), dtype=np.int64)
    m = A.shape[0]
    V = np.eye(n, dtype=np.int64)
    work = A.copy()
    exponents = [exponent] * n  # v(0) = L

    rank_pos = 0
    for _ in range(min(m, n)):
        # locate entry of minimal valuation in the remaining block
        best = None
        for i in range(rank_pos, m):
            for j in range(rank_pos, n):
                value = int(work[i, j]) % modulus
                if value:
                    v = _valuation(value, exponent)
                    if best is None or v < best[0]:
                        best = (v, i, j)
        if best is None:
            break
        v_star, bi, bj = best
        work[[rank_pos, bi]] = work[[bi, rank_pos]]
        work[:, [rank_pos, bj]] = work[:, [bj, rank_pos]]
        V[:, [rank_pos, bj]] = V[:, [bj, rank_pos]]
        pivot = int(work[rank_pos, rank_pos]) % modulus
        unit = pivot >> v_star
        unit_inv = pow(unit, -1, modulus)
        # scale the pivot COLUMN by the unit inverse (a V-operation)
        work[:, rank_pos] = (work[:, rank_pos] * unit_inv) % modulus
        V[:, rank_pos] = (V[:, rank_pos] * unit_inv) % modulus
        # clear the pivot row with column ops (V-operations)
        for j in range(n):
            if j == rank_pos:
                continue
            entry = int(work[rank_pos, j]) % modulus
            if entry:
                coeff = (entry >> v_star) % modulus
                work[:, j] = (work[:, j] - coeff * work[:, rank_pos]) % modulus
                V[:, j] = (V[:, j] - coeff * V[:, rank_pos]) % modulus
        # clear the pivot column with row ops (U-operations, untracked)
        for i in range(m):
            if i == rank_pos:
                continue
            entry = int(work[i, rank_pos]) % modulus
            if entry:
                coeff = entry >> v_star
                work[i] = (work[i] - coeff * work[rank_pos]) % modulus
        exponents[rank_pos] = v_star
        rank_pos += 1

    kernel = []
    for i in range(n):
        a_i = exponents[i]
        if a_i >= 1:
            gen = ((1 << (exponent - a_i)) * V[:, i]) % modulus
            if gen.any():
                kernel.append(gen)
    return {
        "modulus": modulus,
        "V": V % modulus,
        "exponents": exponents,
        "kernel": np.asarray(kernel, dtype=np.int64)
        if kernel
        else np.zeros((0, n), dtype=np.int64),
    }


def check_kernel_certificate(
    constraints: list[tuple[np.ndarray, int]], n: int, certificate: dict, *, exponent: int = 3
) -> bool:
    """Independent verification of a Smith kernel certificate.

    Checks: V unimodular (odd determinant via mod-2 rank); every column of
    ``M = A V`` divisible by ``2^{a_i}`` with mod-2 independent cofactors;
    every claimed kernel generator annihilates ``A``; and the claimed
    kernel matches ``{2^{L-a_i} V e_i}`` exactly.
    """

    modulus = 1 << exponent
    V = np.asarray(certificate["V"], dtype=np.int64) % modulus
    exponents = list(certificate["exponents"])
    if V.shape != (n, n) or len(exponents) != n:
        return False
    # V unimodular over Z_{2^L}  <=>  invertible mod 2
    if f2_rank((V % 2).astype(np.uint8)) != n:
        return False
    if constraints:
        A = np.array(
            [(factor * pattern) % modulus for pattern, factor in constraints],
            dtype=np.int64,
        )
    else:
        A = np.zeros((0, n), dtype=np.int64)
    M = (A @ V) % modulus
    cofactors = []
    for i in range(n):
        a_i = exponents[i]
        if not 0 <= a_i <= exponent:
            return False
        column = M[:, i] % modulus
        if a_i < exponent:
            if np.any(column % (1 << a_i)):
                return False
            cofactors.append(((column >> a_i) % 2).astype(np.uint8))
        else:
            if column.any():
                return False
    if cofactors:
        stacked = np.asarray(cofactors, dtype=np.uint8)
        if f2_rank(stacked) != stacked.shape[0]:
            return False
    expected = []
    for i in range(n):
        a_i = exponents[i]
        if a_i >= 1:
            gen = ((1 << (exponent - a_i)) * V[:, i]) % modulus
            if gen.any():
                expected.append(gen)
    claimed = np.asarray(certificate["kernel"], dtype=np.int64) % modulus
    expected_arr = (
        np.asarray(expected, dtype=np.int64)
        if expected
        else np.zeros((0, n), dtype=np.int64)
    )
    if claimed.shape != expected_arr.shape or not np.array_equal(
        np.sort(claimed, axis=0), np.sort(expected_arr, axis=0)
    ):
        return False
    for gen in claimed:
        if np.any((A @ gen) % modulus):
            return False
    return True
