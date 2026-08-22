"""The binary symplectic space ``F_2^{2n}`` and its group ``Sp(2n, 2)``.

Physical Pauli operators modulo phases are row vectors ``(x | z)`` in
``F_2^{2n}``; commutation is the symplectic form ``[[0, I], [I, 0]]``, and a
Clifford gate modulo Paulis is an element of ``Sp(2n, 2)`` acting on the
right.  Everything here is representation-level: no code, ansatz, or
certificate logic.
"""

from __future__ import annotations

import numpy as np

from .gf2 import BinaryMatrix, gf2_matmul, rank


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


def symplectic_group_order(logical_qubits: int) -> int:
    """Return ``|Sp(2k, 2)|``."""

    if logical_qubits < 0:
        raise ValueError("logical_qubits must be non-negative")
    order = 2 ** (logical_qubits * logical_qubits)
    for index in range(1, logical_qubits + 1):
        order *= 4**index - 1
    return order


def symplectic_gram_schmidt(
    rows: BinaryMatrix, qubits: int
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Split rows spanning a non-degenerate symplectic space into dual pairs
    ``(a_i, b_i)`` with ``<a_i, b_j> = delta_ij`` and all other products 0."""

    remaining = [row.copy() for row in rows]
    first: list[BinaryMatrix] = []
    second: list[BinaryMatrix] = []
    while remaining:
        a = remaining.pop(0)
        partner_index = next(
            (
                i
                for i, r in enumerate(remaining)
                if symplectic_product(a, r, qubits=qubits)[0, 0]
            ),
            None,
        )
        if partner_index is None:
            raise ValueError("symplectic form is degenerate on the given rows")
        b = remaining.pop(partner_index)
        cleaned = []
        for r in remaining:
            r = r ^ (symplectic_product(r, b, qubits=qubits)[0, 0] * a)
            r = r ^ (symplectic_product(r, a, qubits=qubits)[0, 0] * b)
            cleaned.append(r)
        remaining = cleaned
        first.append(a)
        second.append(b)
    k = len(first)
    width = rows.shape[1] if k else 2 * qubits
    return (
        np.asarray(first, dtype=np.uint8).reshape(k, width),
        np.asarray(second, dtype=np.uint8).reshape(k, width),
    )


def symplectic_transvection(direction: np.ndarray) -> np.ndarray:
    """The symplectic transvection ``t_v : x -> x + <x, v> v`` (row action).

    With the repository's form ``[[0, I], [I, 0]]``, ``<x, v>`` is the dot
    product of ``x`` with the half-swapped ``v``, so the matrix is
    ``I + outer(J v, v)``.
    """

    v = np.asarray(direction, dtype=np.uint8) & 1
    two_k = v.shape[0]
    if two_k % 2 or not v.any():
        raise ValueError("direction must be a nonzero symplectic row vector")
    k = two_k // 2
    swapped = np.concatenate([v[k:], v[:k]])
    return (np.eye(two_k, dtype=np.uint8) ^ (swapped[:, None] & v[None, :])).astype(np.uint8)


def _transvection_direction(matrix: np.ndarray) -> np.ndarray | None:
    """The direction vector when ``matrix`` is a symplectic transvection.

    A symplectic matrix with ``rank(t - I) = 1`` is necessarily ``t_v`` with
    ``v`` spanning the row space of ``t - I`` (the rank-one block must be
    ``outer(J v, v)`` for the form to be preserved); the reconstruction is
    verified exactly before returning.
    """

    two_k = matrix.shape[0]
    difference = matrix ^ np.eye(two_k, dtype=np.uint8)
    if rank(difference) != 1:
        return None
    row_index = int(np.flatnonzero(difference.any(axis=1))[0])
    v = difference[row_index]
    try:
        candidate = symplectic_transvection(v)
    except ValueError:
        return None
    if np.array_equal(candidate, matrix):
        return v
    return None
