r"""The preservation algebra of a stabilizer code on a prescribed partition.

This is the unifying object of the package.  Fix a stabilizer code ``S`` on
``n`` qubits and a partition ``P`` of those qubits into cells.  Consider

.. math::

    A_P(S) = \Big\{\, M = \bigoplus_{C \in P} M_C \;:\; S M \subseteq S \,\Big\}
             \subseteq \bigoplus_{C \in P} M_{2|C|}(\mathbb F_2).

Dropping invertibility is what makes the problem linear: ``A_P(S)`` is a
GF(2) *subspace*, computed by a single kernel, and it is closed under
multiplication (if ``SM = RS`` and ``SM' = R'S`` then ``S M M' = R R' S``), so
it is a finite-dimensional unital ``F_2``-algebra.  The physical
code-preserving Clifford gates on the partition, modulo Paulis, are then

.. math::

    A_P(S)^\times \cap \prod_{C \in P} \mathrm{Sp}(2|C|, 2).

For singleton cells the symplectic condition is free, because over
``\mathbb F_2`` one has ``GL(2, 2) = Sp(2, 2)``; the strict site-dependent
transversal Clifford group is therefore exactly the unit group ``A_P(S)^x``.

The linearisation is the stabilizer-code analogue of the Van den Nest-Dehaene-
De Moor local-Clifford linearisation for graph states (PRA 70, 034302 (2004));
the endomorphism-algebra viewpoint goes back to Rains and was developed for
transversal Clifford classification by Dasu and Burton (arXiv:2507.10519).
See ``docs/related_work.md`` for the positioning.

Constraint construction is LDPC-friendly: for a stabilizer row ``s`` the image
``sM`` is supported inside the closure of ``supp(s)`` under the partition, so
the dual space ``S^\perp`` is restricted to those columns before row
reduction and a weight-``w`` row contributes ``O(w)`` constraints rather than
``O(n)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..utils.gf2 import BinaryMatrix, nullspace, row_basis

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from ..codes.stabilizer import StabilizerCode

#: Above this algebra dimension a ``2^dim`` enumeration is refused and the
#: structured unit-group route (:mod:`.unit_group`) is used instead.
ENUMERATION_DIM_CAP = 24

#: The six elements of SL(2,2) = Sp(2,2) as (a, b, c, d) with M = [[a, b], [c, d]].
_SL22 = [
    (1, 0, 0, 1),  # I
    (1, 1, 0, 1),  # S      (X -> Y)
    (1, 0, 1, 1),  # sqrtX  (Z -> Y)
    (0, 1, 1, 0),  # H      (X <-> Z)
    (1, 1, 1, 0),  # HS
    (0, 1, 1, 1),  # SH
]


def _block_action_matrix(entries: np.ndarray, n: int) -> BinaryMatrix:
    """The ``2n x 2n`` symplectic matrix of per-qubit blocks ``(a,b,c,d)``
    in the ``(x_0..x_{n-1} | z_0..z_{n-1})`` row-vector convention."""

    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    a, b, c, d = entries[0::4], entries[1::4], entries[2::4], entries[3::4]
    idx = np.arange(n)
    matrix[idx, idx] = a
    matrix[idx, n + idx] = b
    matrix[n + idx, idx] = c
    matrix[n + idx, n + idx] = d
    return matrix


def local_clifford_algebra(code: StabilizerCode) -> BinaryMatrix:
    """Basis of ``{M block-diagonal : H M \\subseteq rowspan(H)}`` over F_2.

    Unknowns are the ``4n`` block entries ``(a_i, b_i, c_i, d_i)``.  A row
    ``s = (x|z)`` maps to ``s'`` with ``x'_i = a_i x_i + c_i z_i`` and
    ``z'_i = b_i x_i + d_i z_i``; membership ``s' in rowspan(H)`` is tested
    against ``N = rowspan(H)^\\perp``, giving linear constraints.  Constraints
    for ``s`` only touch qubits in its support, so ``N`` is restricted there
    first (the LDPC-friendly construction of the CSS solver).
    """

    n = code.n
    perp = nullspace(code.h)
    constraints: list[np.ndarray] = []
    for s in code.h:
        x, z = s[:n], s[n:]
        support = np.flatnonzero(x | z)
        if support.size == 0:
            continue
        columns = np.concatenate([support, n + support])
        local = row_basis(perp[:, columns], ncols=2 * support.size)
        for w in local:
            wx = np.zeros(n, dtype=np.uint8)
            wz = np.zeros(n, dtype=np.uint8)
            wx[support] = w[: support.size]
            wz[support] = w[support.size :]
            row = np.zeros(4 * n, dtype=np.uint8)
            row[0::4] = wx & x
            row[1::4] = wz & x
            row[2::4] = wx & z
            row[3::4] = wz & z
            if row.any():
                constraints.append(row)
    if constraints:
        constraint_matrix = row_basis(np.asarray(constraints, dtype=np.uint8))
    else:
        constraint_matrix = np.zeros((0, 4 * n), dtype=np.uint8)
    return nullspace(constraint_matrix)


def _local_symplectic_form(width: int) -> BinaryMatrix:
    form = np.zeros((2 * width, 2 * width), dtype=np.uint8)
    identity = np.eye(width, dtype=np.uint8)
    form[:width, width:] = identity
    form[width:, :width] = identity
    return form


def _cell_layout(n: int, cells: list[tuple[int, ...]]) -> tuple[list[tuple[int, int]], int]:
    covered = sorted(q for cell in cells for q in cell)
    if covered != list(range(n)):
        raise ValueError("cells must partition the qubits exactly")
    layout: list[tuple[int, int]] = []
    offset = 0
    for cell in cells:
        layout.append((offset, len(cell)))
        offset += (2 * len(cell)) ** 2
    return layout, offset


def _preserving_constraints(
    rows: BinaryMatrix,
    n: int,
    cells: list[tuple[int, ...]],
    layout: list[tuple[int, int]],
    total: int,
) -> list[np.ndarray]:
    r"""Constraint rows for ``{M cell-block-diagonal : rows M \subseteq rowspan(rows)}``.

    ``rows`` need not be self-orthogonal: the construction only uses the row
    space and its ordinary dual, so it applies to the stabilizer and to its
    normalizer alike.
    """

    cell_of = {}
    for cell in cells:
        for q in cell:
            cell_of[q] = cell
    perp = nullspace(rows)
    constraints: list[np.ndarray] = []
    for s_row in rows:
        x, z = s_row[:n], s_row[n:]
        raw_support = np.flatnonzero(x | z)
        if raw_support.size == 0:
            continue
        # the image spreads over every cell touching the support, so dual
        # vectors must be restricted to the cell closure, not the support
        support = np.unique(
            np.concatenate([np.asarray(cell_of[int(q)], dtype=int) for q in raw_support])
        )
        columns = np.concatenate([support, n + support])
        local = row_basis(perp[:, columns], ncols=2 * support.size)
        for w in local:
            wx = np.zeros(n, dtype=np.uint8)
            wz = np.zeros(n, dtype=np.uint8)
            wx[support] = w[: support.size]
            wz[support] = w[support.size :]
            row = np.zeros(total, dtype=np.uint8)
            for cell, (start, width) in zip(cells, layout):
                idx = np.asarray(cell, dtype=int)
                u = np.concatenate([x[idx], z[idx]])
                v = np.concatenate([wx[idx], wz[idx]])
                if u.any() and v.any():
                    row[start : start + (2 * width) ** 2] = np.outer(u, v).reshape(-1)
            if row.any():
                constraints.append(row)
    return constraints


def partition_algebra(
    code: StabilizerCode, cells: list[tuple[int, ...]], *, refine: bool = True
) -> tuple[BinaryMatrix, list[tuple[int, int]]]:
    r"""Basis of the preservation algebra on ``cells``.

    ``cells`` partitions the qubits; each cell of width ``w`` contributes a
    free ``2w x 2w`` block acting on its local ``(x.. | z..)`` coordinates.
    Returns the algebra basis over the concatenated block entries together
    with the ``(offset, width)`` layout of each cell.

    **The involution, and why ``refine`` is the default.**  Write
    ``sigma(M) = Omega M^T Omega``.  A symplectic ``M`` satisfies
    ``M^{-1} = sigma(M)``, so the group we are after is the *unitary group*
    of the algebra with involution ``(A, sigma)``,

        ``G = {M in A^x : sigma(M) M = 1}``.

    That description is only available if ``sigma`` maps ``A`` to itself, and
    for the naive algebra ``A(S) = {M : S M subseteq S}`` it does not.  A short
    computation identifies the image exactly:

        ``sigma(A(S)) = A(N)``,  ``N = S^perp`` the normalizer,

    because ``<s, w> = 0`` for the ordinary dual sends the defining condition
    to ``N M subseteq N`` (using ``S^T Omega = N``).  Hence the *sigma-stable*
    object is the intersection

        ``A'(S) = A(S) cap A(N)``,

    and it loses nothing: an invertible symplectic ``M`` with ``S M = S`` also
    satisfies ``N M = N`` (pairings are preserved), and its inverse
    ``sigma(M)`` again lies in ``A'``, so

        ``A'(S)^x cap prod_C Sp(2|C|,2)  =  A(S)^x cap prod_C Sp(2|C|,2)``.

    The refinement therefore returns the *same* gate group from a strictly
    smaller algebra -- typically much smaller on multi-qubit cells, where the
    naive algebra carries a large ``sigma``-asymmetric part that no symplectic
    element ever uses.  Measured: ``[[4,2,2]]`` pairs 20 -> 16, ``[[6,2,2]]``
    pairs 28 -> 20, ``[[8,3,2]]`` pairs 27 -> 22, ``iceberg-8`` pairs 36 -> 24.
    Two of those cross the enumeration cap, turning an honest ``UNKNOWN`` into
    an exact answer.

    Pass ``refine=False`` for the pre-0.2.1 algebra ``A(S)`` itself, which is
    still the right object when the question is about code-preserving *linear*
    maps rather than about gates.
    """

    n = code.n
    layout, total = _cell_layout(n, cells)
    constraints = _preserving_constraints(code.h, n, cells, layout, total)
    if refine:
        constraints += _preserving_constraints(
            code.normalizer, n, cells, layout, total
        )
    if constraints:
        constraint_matrix = row_basis(np.asarray(constraints, dtype=np.uint8))
    else:
        constraint_matrix = np.zeros((0, total), dtype=np.uint8)
    return nullspace(constraint_matrix), layout


def symplectic_involution(
    entries: np.ndarray, layout: list[tuple[int, int]]
) -> np.ndarray:
    r"""``sigma(M) = Omega M^T Omega``, blockwise, in the flat entry encoding.

    An anti-automorphism with ``sigma^2 = id``.  ``M`` is blockwise symplectic
    exactly when ``sigma(M) M = 1``, which is what makes the gate group the
    unitary group of ``(A'(S), sigma)`` -- see :func:`partition_algebra`.
    """

    out = np.zeros_like(entries)
    for start, width in layout:
        size = 2 * width
        block = entries[start : start + size * size].reshape(size, size)
        form = _local_symplectic_form(width)
        out[start : start + size * size] = ((form @ block.T @ form) % 2).reshape(-1)
    return out


def _cell_blocks(entries: np.ndarray, layout: list[tuple[int, int]]):
    for start, width in layout:
        size = 2 * width
        yield entries[start : start + size * size].reshape(size, size)


def _partition_action_matrix(
    entries: np.ndarray, cells: list[tuple[int, ...]], layout: list[tuple[int, int]], n: int
) -> BinaryMatrix:
    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for cell, block in zip(cells, _cell_blocks(entries, layout)):
        idx = np.asarray(cell, dtype=int)
        coords = np.concatenate([idx, n + idx])
        matrix[np.ix_(coords, coords)] = block
    return matrix


def _partition_multiply(cells, layout, width):
    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.zeros(width, dtype=np.uint8)
        for (start, cell_width) in layout:
            size = 2 * cell_width
            A = a[start : start + size * size].reshape(size, size)
            B = b[start : start + size * size].reshape(size, size)
            out[start : start + size * size] = (A @ B % 2).reshape(-1)
        return out

    return multiply
