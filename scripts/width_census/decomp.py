"""Is a stabilizer code a tensor product of two smaller codes?  Certified test.

``S`` splits along a bipartition ``(A, B)`` iff the stabilizer elements
supported inside ``A``, together with those supported inside ``B``, already
span ``S``.  The search is exhaustive over bipartitions, so "indecomposable"
is a proof rather than an assertion.
"""
from __future__ import annotations

import itertools

import numpy as np

from qec_transversal.utils.gf2 import rank


def _dim_supported_in(rows: np.ndarray, region, n: int) -> int:
    """Dimension of the subspace of ``<rows>`` vanishing outside ``region``."""

    outside = [q for q in range(n) if q not in region]
    if not outside:
        return rank(rows)
    cols = np.array(outside + [n + q for q in outside], dtype=int)
    return rank(rows) - rank(rows[:, cols])


def decomposition(code):
    """A nontrivial tensor factorisation of the qubits, or ``None``."""

    n = code.n
    rows = np.asarray(code.h, dtype=np.uint8)
    total = rank(rows)
    for size in range(1, n // 2 + 1):
        for region in itertools.combinations(range(n), size):
            other = tuple(q for q in range(n) if q not in region)
            if (
                _dim_supported_in(rows, region, n)
                + _dim_supported_in(rows, other, n)
                == total
            ):
                return [region, other]
    return None
