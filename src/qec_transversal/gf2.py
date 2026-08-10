"""Small, dependency-light linear algebra helpers over :math:`GF(2)`.

The implementation intentionally uses NumPy ``uint8`` matrices in the first
release.  The public API is kept backend-neutral so a bit-packed/M4RI backend
can replace it without changing the CSS analysis layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

BinaryMatrix = NDArray[np.uint8]


def as_binary_matrix(matrix: object, *, ncols: int | None = None) -> BinaryMatrix:
    """Return a two-dimensional binary ``uint8`` matrix.

    Empty row collections need ``ncols`` because their width cannot otherwise
    be inferred.  Inputs containing values other than zero and one are rejected
    instead of being silently reduced modulo two.
    """

    array = np.asarray(matrix, dtype=np.uint8)
    if array.size == 0:
        width = ncols
        if array.ndim == 2:
            width = array.shape[1] if ncols is None else ncols
        if width is None:
            raise ValueError("ncols is required for an empty matrix")
        return np.zeros((0, width), dtype=np.uint8)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    if ncols is not None and array.shape[1] != ncols:
        raise ValueError(f"expected {ncols} columns, got {array.shape[1]}")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("GF(2) matrices may contain only 0 and 1")
    return array.copy()


def rref(matrix: object) -> tuple[BinaryMatrix, tuple[int, ...]]:
    """Reduced row-echelon form over GF(2), together with pivot columns."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    reduced = (array & 1).copy()
    rows, cols = reduced.shape
    pivots: list[int] = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row == rows:
            break
        candidates = np.flatnonzero(reduced[pivot_row:, col])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]
        other_rows = np.flatnonzero(reduced[:, col])
        other_rows = other_rows[other_rows != pivot_row]
        if other_rows.size:
            reduced[other_rows] ^= reduced[pivot_row]
        pivots.append(col)
        pivot_row += 1

    return reduced[:pivot_row], tuple(pivots)


def row_basis(matrix: object, *, ncols: int | None = None) -> BinaryMatrix:
    """Return the nonzero RREF rows, a canonical basis of the row space."""

    array = as_binary_matrix(matrix, ncols=ncols)
    return rref(array)[0]


def rank(matrix: object) -> int:
    """Rank over GF(2)."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    return len(rref(array)[1])


def nullspace(matrix: object) -> BinaryMatrix:
    """Return a row basis of the right nullspace over GF(2)."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    reduced, pivots = rref(array)
    free = [col for col in range(array.shape[1]) if col not in pivots]
    basis = np.zeros((len(free), array.shape[1]), dtype=np.uint8)
    for basis_row, free_col in enumerate(free):
        basis[basis_row, free_col] = 1
        for reduced_row, pivot_col in enumerate(pivots):
            basis[basis_row, pivot_col] = reduced[reduced_row, free_col]
    return basis


def gf2_inverse(matrix: object) -> BinaryMatrix:
    """Invert a square matrix over GF(2)."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("expected a square matrix")
    size = array.shape[0]
    if size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    augmented = np.concatenate([array & 1, np.eye(size, dtype=np.uint8)], axis=1)
    reduced, pivots = rref(augmented)
    if reduced.shape[0] != size or pivots[:size] != tuple(range(size)):
        raise ValueError("matrix is singular over GF(2)")
    if not np.array_equal(reduced[:, :size], np.eye(size, dtype=np.uint8)):
        raise ValueError("matrix is singular over GF(2)")
    return reduced[:, size:]


def is_in_rowspace(vector: object, basis: object) -> bool:
    """Test membership in a row space over GF(2)."""

    basis_array = np.asarray(basis, dtype=np.uint8)
    if basis_array.ndim != 2:
        raise ValueError("expected a two-dimensional basis")
    vector_array = np.asarray(vector, dtype=np.uint8).reshape(1, -1)
    if vector_array.shape[1] != basis_array.shape[1]:
        raise ValueError("vector and basis widths differ")
    return rank(np.vstack([basis_array, vector_array])) == rank(basis_array)


def quotient_complement(ambient: object, subspace: object) -> BinaryMatrix:
    """Choose representatives completing ``subspace`` to ``ambient``.

    Both arguments are row generators.  The returned rows form a basis of an
    arbitrary complement, so their cosets form a basis of ``ambient/subspace``.
    """

    ambient_array = np.asarray(ambient, dtype=np.uint8)
    subspace_array = np.asarray(subspace, dtype=np.uint8)
    if ambient_array.ndim != 2 or subspace_array.ndim != 2:
        raise ValueError("expected two-dimensional matrices")
    if ambient_array.shape[1] != subspace_array.shape[1]:
        raise ValueError("ambient and subspace widths differ")

    ambient_basis = row_basis(ambient_array)
    current = row_basis(subspace_array, ncols=ambient_array.shape[1])
    ambient_rank = rank(ambient_basis)
    if rank(np.vstack([ambient_basis, current])) != ambient_rank:
        raise ValueError("the alleged subspace is not contained in the ambient space")

    representatives: list[BinaryMatrix] = []
    current_rank = rank(current)
    for row in ambient_basis:
        candidate = np.vstack([current, row])
        candidate_rank = rank(candidate)
        if candidate_rank > current_rank:
            representatives.append(row.copy())
            current = row_basis(candidate)
            current_rank = candidate_rank
    if not representatives:
        return np.zeros((0, ambient_array.shape[1]), dtype=np.uint8)
    return np.asarray(representatives, dtype=np.uint8)


def symplectic_form(qubits: int) -> BinaryMatrix:
    """The binary symplectic form ``[[0,I],[I,0]]`` for row vectors."""

    if qubits < 0:
        raise ValueError("qubits must be non-negative")
    form = np.zeros((2 * qubits, 2 * qubits), dtype=np.uint8)
    identity = np.eye(qubits, dtype=np.uint8)
    form[:qubits, qubits:] = identity
    form[qubits:, :qubits] = identity
    return form


def symplectic_product(left: object, right: object, *, qubits: int) -> BinaryMatrix:
    """Pair stacks of row vectors with the binary symplectic form."""

    left_array = np.atleast_2d(np.asarray(left, dtype=np.uint8))
    right_array = np.atleast_2d(np.asarray(right, dtype=np.uint8))
    expected = 2 * qubits
    if left_array.shape[1] != expected or right_array.shape[1] != expected:
        raise ValueError(f"symplectic vectors must have width {expected}")
    return (left_array @ symplectic_form(qubits) @ right_array.T) & 1


def is_symplectic(matrix: object, *, qubits: int) -> bool:
    """Whether a matrix preserves the binary symplectic form."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.shape != (2 * qubits, 2 * qubits):
        return False
    form = symplectic_form(qubits)
    return np.array_equal((array @ form @ array.T) & 1, form)


def supports(rows: Iterable[Sequence[int]]) -> list[list[int]]:
    """Return support indices for a collection of binary rows."""

    return [np.flatnonzero(np.asarray(row, dtype=np.uint8)).astype(int).tolist() for row in rows]

