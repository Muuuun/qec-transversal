"""Structural ZX-duality candidates for the built-in code families.

Every candidate is only a *guess* derived from the family's algebraic
structure; :class:`~.matching.MatchingAnalysis` certifies or rejects it, so
an invalid candidate can never contaminate results.
"""

from __future__ import annotations

import numpy as np


def two_block_inversion(l: int, m: int) -> np.ndarray:
    """Block swap with group inversion, the BB duality ``tau_0``.

    Left qubit at ``(i, j)`` maps to the right qubit at ``(-i, -j)`` and
    back; a ZX-duality for every two-block code ``H_X = [A|B]``,
    ``H_Z = [B^T|A^T]`` because ``A^T = A(x^-1, y^-1)``.
    """

    lm = l * m
    tau = np.zeros(2 * lm, dtype=int)
    for i in range(l):
        for j in range(m):
            left = i * m + j
            right = lm + ((-i) % l) * m + ((-j) % m)
            tau[left] = right
            tau[right] = left
    return tau


def gb_inversion(size: int) -> np.ndarray:
    """The univariate (generalized-bicycle) version of ``tau_0``."""

    tau = np.zeros(2 * size, dtype=int)
    for i in range(size):
        left, right = i, size + (-i) % size
        tau[left] = right
        tau[right] = left
    return tau


def two_block_swap_xy(l: int, m: int, *, invert: bool) -> np.ndarray:
    """Block swap with the ``x <-> y`` reflection, for symmetric BB codes
    (``l = m`` and ``B(x, y) = A(y, x)``).  ``invert`` additionally negates
    both coordinates."""

    if l != m:
        raise ValueError("the x<->y swap candidate needs l = m")
    lm = l * m
    tau = np.zeros(2 * lm, dtype=int)
    for i in range(l):
        for j in range(m):
            if invert:
                ti, tj = (-j) % l, (-i) % m
            else:
                ti, tj = j, i
            left = i * m + j
            right = lm + ti * m + tj
            tau[left] = right
            tau[right] = left
    return tau


def two_block_reflection(l: int, m: int, *, invert: bool) -> np.ndarray:
    """Same-block ``x <-> y`` reflection for symmetric BB codes.

    Maps ``(i, j)`` to ``(j, i)`` (or ``(-j, -i)`` with ``invert``) inside
    each block; when ``B(x, y) = A(y, x)`` this conjugates ``A`` onto
    ``B^T`` in place, exchanging X- and Z-checks without a block swap."""

    if l != m:
        raise ValueError("the reflection candidate needs l = m")
    lm = l * m
    tau = np.zeros(2 * lm, dtype=int)
    for block in range(2):
        for i in range(l):
            for j in range(m):
                if invert:
                    ti, tj = (-j) % l, (-i) % m
                else:
                    ti, tj = j, i
                tau[block * lm + i * m + j] = block * lm + ti * m + tj
    return tau


def hgp_transpose(n1: int, r1: int) -> np.ndarray:
    """The swap fold of a symmetric hypergraph product ``HGP(h, h)``.

    Sector one (``n1 x n1`` qubits) and sector two (``r1 x r1``) are each
    transposed; the diagonals are the fold's fixed points.
    """

    total = n1 * n1 + r1 * r1
    tau = np.zeros(total, dtype=int)
    for u in range(n1):
        for v in range(n1):
            tau[u * n1 + v] = v * n1 + u
    offset = n1 * n1
    for a in range(r1):
        for b in range(r1):
            tau[offset + a * r1 + b] = offset + b * r1 + a
    return tau


def kasai_block_negation(width: int, lift: int, *, reverse_blocks: bool, extension: int = 1) -> np.ndarray:
    """Candidate for Kasai quasi-cyclic pairs: negate the circulant index
    within each of the ``width`` blocks, optionally reversing block order.
    ``extension`` tensors the candidate with the identity on the GF(2^e)
    expansion."""

    base = np.zeros(width * lift, dtype=int)
    for block in range(width):
        target_block = (width - 1 - block) if reverse_blocks else block
        for p in range(lift):
            base[block * lift + p] = target_block * lift + (-p) % lift
    if extension == 1:
        tau = base
    else:
        tau = np.zeros(width * lift * extension, dtype=int)
        for q in range(width * lift):
            for bit in range(extension):
                tau[q * extension + bit] = base[q] * extension + bit
    return tau


#: Candidate dualities per registry code: name -> list of (label, tau).
def candidates_for(name: str) -> list[tuple[str, np.ndarray]]:
    bb = {
        "bb72": (6, 6), "bb90": (15, 3), "bb108": (9, 6), "gross": (12, 6),
        "two-gross": (12, 12), "bb360": (30, 6), "bb756": (21, 18),
        "bb54": (3, 9), "bb98-symmetric": (7, 7), "bb162-symmetric": (9, 9),
        "coprime30": (3, 5), "coprime42": (3, 7), "coprime70": (5, 7),
        "coprime126": (7, 9), "coprime154": (7, 11), "trivariate30": (3, 5),
    }
    gb = {"gb48": 24, "gb46": 23, "gb126": 63}
    hgp = {
        "hgp-hamming": (7, 3), "toric-4": (4, 4), "toric-10": (10, 10),
        "surface-5": (5, 4), "lacross65": (7, 4), "lacross400": (16, 12),
    }
    if name in bb:
        l, m = bb[name]
        result = [("tau0: block swap + inversion", two_block_inversion(l, m))]
        if l == m:
            result.append(("same-block x<->y reflection", two_block_reflection(l, m, invert=False)))
            result.append(("same-block inverted reflection", two_block_reflection(l, m, invert=True)))
            result.append(("block swap + x<->y", two_block_swap_xy(l, m, invert=False)))
            result.append(("block swap + inverted x<->y", two_block_swap_xy(l, m, invert=True)))
        return result
    if name in gb:
        return [("tau0: block swap + inversion", gb_inversion(gb[name]))]
    if name in hgp:
        n1, r1 = hgp[name]
        return [("HGP transpose fold", hgp_transpose(n1, r1))]
    if name == "lifted-b1":
        # 7x7 blocks of 63-circulants: treat as a two-block code over the
        # group Z_7 x Z_63 (block index x circulant index).
        return [("tau0: block swap + inversion", two_block_inversion(7, 63))]
    if name == "kasai-binary-294":
        return [
            ("block negation", kasai_block_negation(6, 49, reverse_blocks=False)),
            ("block reversal + negation", kasai_block_negation(6, 49, reverse_blocks=True)),
        ]
    if name == "kasai-binary-1104":
        return [
            ("block negation", kasai_block_negation(8, 138, reverse_blocks=False)),
            ("block reversal + negation", kasai_block_negation(8, 138, reverse_blocks=True)),
        ]
    if name == "kasai-gf256-2352":
        return [
            ("block negation x I8", kasai_block_negation(6, 49, reverse_blocks=False, extension=8)),
            ("block reversal + negation x I8", kasai_block_negation(6, 49, reverse_blocks=True, extension=8)),
        ]
    return []
