"""Code concatenation, the canonical source of large-cell depth-one layers.

Concatenating an ``[[n_o, 1, d_o]]`` outer code with an ``[[n_i, 1, d_i]]``
inner code gives ``[[n_o n_i, 1, >= d_o d_i]]``: each outer physical qubit is
replaced by an inner block, outer ``X_j, Z_j`` by the inner logical
``Xbar, Zbar`` of block ``j``.

The construction matters here because it decouples the two quantities that get
confused in locality arguments.  Take the partition into inner blocks: the cell
size is ``n_i``, which grows without bound, while the partition distance is the
*outer* distance ``d_o`` -- one entire inner block can fail arbitrarily and the
outer code still corrects it.  So a depth-one layer of ``n_i``-qubit native
gates can be fault tolerant with ``n_i`` far larger than the code distance.
"""

from __future__ import annotations

import numpy as np

from .gf2 import BinaryMatrix
from .stabilizer import StabilizerCode


def concatenate(outer: StabilizerCode, inner: StabilizerCode) -> StabilizerCode:
    """Concatenate a ``k = 1`` inner code beneath an outer stabilizer code.

    Every outer qubit becomes one inner block, in order, so block ``j`` owns
    physical qubits ``j * n_i ... (j + 1) * n_i - 1`` -- the partition returned
    by :func:`inner_block_cells`.
    """

    if inner.k != 1:
        raise ValueError("the inner code must encode exactly one logical qubit")
    n_o, n_i = outer.n, inner.n
    n = n_o * n_i
    x_bar, z_bar = inner.logical[0], inner.logical[1]

    rows: list[np.ndarray] = []
    for block in range(n_o):
        for check in inner.h:
            row = np.zeros(2 * n, dtype=np.uint8)
            row[block * n_i : (block + 1) * n_i] = check[:n_i]
            row[n + block * n_i : n + (block + 1) * n_i] = check[n_i:]
            rows.append(row)
    for check in outer.h:
        rows.append(lift_pauli(check, outer.n, x_bar, z_bar))
    return StabilizerCode(np.asarray(rows, dtype=np.uint8))


def lift_pauli(
    vector: object, outer_qubits: int, x_bar: BinaryMatrix, z_bar: BinaryMatrix
) -> BinaryMatrix:
    """Rewrite an outer Pauli in the concatenated code's physical coordinates."""

    outer_vector = np.asarray(vector, dtype=np.uint8).reshape(-1)
    n_i = x_bar.size // 2
    n = outer_qubits * n_i
    lifted = np.zeros(2 * n, dtype=np.uint8)
    for j in range(outer_qubits):
        piece = np.zeros(2 * n_i, dtype=np.uint8)
        if outer_vector[j]:
            piece ^= x_bar
        if outer_vector[outer_qubits + j]:
            piece ^= z_bar
        lifted[j * n_i : (j + 1) * n_i] ^= piece[:n_i]
        lifted[n + j * n_i : n + (j + 1) * n_i] ^= piece[n_i:]
    return lifted


def inner_block_cells(outer_qubits: int, inner_qubits: int) -> list[tuple[int, ...]]:
    """The partition of the concatenated code into inner blocks."""

    return [
        tuple(range(block * inner_qubits, (block + 1) * inner_qubits))
        for block in range(outer_qubits)
    ]


def shor_code() -> StabilizerCode:
    """Shor's ``[[9,1,3]]`` code, the standard inner block."""

    n = 9
    rows: list[np.ndarray] = []
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]:
        row = np.zeros(2 * n, dtype=np.uint8)
        row[n + a] = row[n + b] = 1
        rows.append(row)
    for block in [(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)]:
        row = np.zeros(2 * n, dtype=np.uint8)
        for qubit in block:
            row[qubit] = 1
        rows.append(row)
    return StabilizerCode(np.asarray(rows, dtype=np.uint8))
