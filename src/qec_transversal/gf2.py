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
            if ncols is not None and array.shape[1] not in (0, ncols):
                raise ValueError(f"expected {ncols} columns, got {array.shape[1]}")
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


def gf2_matmul(left: object, right: object) -> BinaryMatrix:
    """Matrix product over GF(2).

    Large products are routed through float32 BLAS, which is far faster than
    NumPy's integer fallback; float32 sums are exact for inner dimensions
    below ``2**24``.
    """

    left_array = np.asarray(left, dtype=np.uint8)
    right_array = np.asarray(right, dtype=np.uint8)
    inner = left_array.shape[-1]
    work = left_array.size * (right_array.shape[-1] if right_array.ndim > 1 else 1)
    if work > 262_144 and inner < (1 << 24):
        # Apple Accelerate's sgemm pollutes the floating-point flags (notably
        # underflow) even though every value here is an exact small integer.
        with np.errstate(all="ignore"):
            product = left_array.astype(np.float32) @ right_array.astype(np.float32)
        return (product % 2).astype(np.uint8)
    return (left_array @ right_array) & 1


_PACKED_MIN_COLS = 257  # switch to the bit-packed kernel for wide matrices


def _rref_packed(array: np.ndarray) -> tuple[BinaryMatrix, tuple[int, ...]]:
    """Bit-packed reduced row echelon form (uint64 words, 64 bits/op).

    Semantically identical to the uint8 path; used automatically for wide
    matrices.  Equivalence is fuzz-tested against the dense path.
    """

    rows, cols = array.shape
    if rows == 0:
        return array.copy(), ()
    words = (cols + 63) // 64
    padded = np.zeros((rows, words * 64), dtype=np.uint8)
    padded[:, :cols] = array & 1
    packed = np.packbits(padded, axis=1, bitorder="little")
    packed = packed.reshape(rows, words * 8).view(np.uint64).reshape(rows, words)

    pivots: list[int] = []
    pivot_row = 0
    for col in range(cols):
        word, bit = divmod(col, 64)
        column_bits = (packed[pivot_row:, word] >> np.uint64(bit)) & np.uint64(1)
        candidates = np.flatnonzero(column_bits)
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            packed[[pivot_row, selected]] = packed[[selected, pivot_row]]
        all_bits = (packed[:, word] >> np.uint64(bit)) & np.uint64(1)
        others = np.flatnonzero(all_bits)
        others = others[others != pivot_row]
        if others.size:
            packed[others] ^= packed[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    unpacked = np.unpackbits(
        packed[:pivot_row].view(np.uint8).reshape(pivot_row, -1),
        axis=1,
        bitorder="little",
    )[:, :cols]
    return unpacked.astype(np.uint8), tuple(pivots)


def rref(matrix: object) -> tuple[BinaryMatrix, tuple[int, ...]]:
    """Reduced row-echelon form over GF(2), together with pivot columns."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    if array.shape[1] >= _PACKED_MIN_COLS and array.shape[0] > 1:
        return _rref_packed(array)
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
    basis[np.arange(len(free)), free] = 1
    if pivots and free:
        basis[:, list(pivots)] = reduced[:, free].T
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


def reduce_rows(
    rows: object, reduced: BinaryMatrix, pivots: Sequence[int]
) -> BinaryMatrix:
    """Reduce rows against a precomputed RREF, returning the residues.

    RREF rows vanish at every other pivot column, so the pivot coordinates of
    an input row are exactly its coefficients on the basis, and one matrix
    product reduces every row at once.
    """

    rows_array = np.atleast_2d(np.asarray(rows, dtype=np.uint8)) & 1
    if not reduced.shape[0]:
        return rows_array.copy()
    if rows_array.shape[1] != reduced.shape[1]:
        raise ValueError("rows and basis widths differ")
    coefficients = rows_array[:, list(pivots)]
    return (rows_array ^ gf2_matmul(coefficients, reduced)) & 1


def rowspace_residues(rows: object, basis: object) -> BinaryMatrix:
    """Reduce each row against ``rowspan(basis)``, returning the residues.

    A residue row is zero exactly when the corresponding input row lies in the
    row space.  ``basis`` may be any generating set; it is row-reduced once,
    after which every input row is reduced with a single matrix product.
    """

    basis_array = np.asarray(basis, dtype=np.uint8)
    if basis_array.ndim != 2:
        raise ValueError("expected a two-dimensional basis")
    rows_array = np.atleast_2d(np.asarray(rows, dtype=np.uint8)) & 1
    if rows_array.shape[1] != basis_array.shape[1]:
        raise ValueError("rows and basis widths differ")
    reduced, pivots = rref(basis_array)
    return reduce_rows(rows_array, reduced, pivots)


def is_in_rowspace(vector: object, basis: object) -> bool:
    """Test membership in a row space over GF(2)."""

    vector_array = np.asarray(vector, dtype=np.uint8).reshape(1, -1)
    return not rowspace_residues(vector_array, basis).any()


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

    width = ambient_array.shape[1]
    ambient_basis = row_basis(ambient_array)
    if rowspace_residues(subspace_array & 1, ambient_basis).any():
        raise ValueError("the alleged subspace is not contained in the ambient space")

    # Reduce every ambient basis row against the subspace in one batched
    # product, then run an incremental RREF over the residues.  The working
    # set only ever holds the complement, whose size is the quotient
    # dimension, so this pass stays cheap even for large ambient spaces.
    residues = rowspace_residues(ambient_basis, subspace_array)
    work = np.zeros((0, width), dtype=np.uint8)
    pivots: list[int] = []
    representatives: list[BinaryMatrix] = []
    for index, batched_residue in enumerate(residues):
        if pivots:
            residue = (batched_residue ^ ((batched_residue[pivots] @ work) & 1)) & 1
        else:
            residue = batched_residue.copy()
        support = np.flatnonzero(residue)
        if support.size == 0:
            continue
        representatives.append(ambient_basis[index].copy())
        pivot = int(support[0])
        eliminate = np.flatnonzero(work[:, pivot]) if work.shape[0] else np.empty(0, int)
        if eliminate.size:
            work[eliminate] ^= residue
        work = np.vstack([work, residue])
        pivots.append(pivot)
    if not representatives:
        return np.zeros((0, width), dtype=np.uint8)
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
    # Multiplying by the form only swaps the X and Z halves of each vector.
    swapped = np.hstack([left_array[:, qubits:], left_array[:, :qubits]])
    return gf2_matmul(swapped, right_array.T)


def is_symplectic(matrix: object, *, qubits: int) -> bool:
    """Whether a matrix preserves the binary symplectic form."""

    array = np.asarray(matrix, dtype=np.uint8)
    if array.shape != (2 * qubits, 2 * qubits):
        return False
    swapped = np.hstack([array[:, qubits:], array[:, :qubits]])
    return np.array_equal(gf2_matmul(swapped, array.T), symplectic_form(qubits))


def supports(rows: Iterable[Sequence[int]]) -> list[list[int]]:
    """Return support indices for a collection of binary rows."""

    return [np.flatnonzero(np.asarray(row, dtype=np.uint8)).astype(int).tolist() for row in rows]

