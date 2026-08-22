"""Partition distance: the exact fault-tolerance criterion for a depth-one layer.

A depth-one layer of native gates factors as ``U = prod_C U_C`` over the cells
of a partition ``P = {C_1, ..., C_m}`` of the physical qubits.  A fault in the
native gate on ``C`` is an arbitrary error supported on ``C`` -- and it *stays*
on ``C``, because ``U_C`` acts only there (an error before the gate conjugates
to an error after it, still inside ``C``).  So the error model induced by one
faulty gate is exactly "any Pauli supported on one cell", and correctability is
decided by Knill-Laflamme against that error set.

For a stabilizer code, ``E, F`` supported on cells ``C, D`` are distinguishable
unless ``E F`` lies in ``N(S) \\ S``.  Writing

    bwt_P(L) = |{C in P : supp(L) meets C}|,
    d_P      = min over L in N(S) \\ S of bwt_P(L),

that reads:

    one unflagged faulty gate is correctable  <=>  d_P >= 3,
    r unflagged faulty gates are correctable  <=>  d_P >= 2r + 1,
    one *flagged* (located) faulty gate       <=>  d_P >= 2.

``d_P`` -- not the cell size ``l``, and not the code distance ``d`` -- is the
quantity a depth-one construction has to defend.  The two are related only by
``d_P >= ceil(d / l)``, which recovers the familiar sufficient condition
``2l < d ==> d_P >= 3`` while showing that ``l >= d`` implies *nothing*: a
large cell does not have to contain a logical operator.

Deciding ``d_P >= 3`` needs no SAT solver.  ``bwt_P(L) <= 2`` means ``L`` is
supported inside ``C u D`` for one of the ``O(m^2)`` cell unions, and

    "N(S) \\ S meets the Paulis supported on T"

is a rank computation on ``2|T|`` columns: the Paulis on ``T`` commuting with
``S`` form the kernel ``N_T``, and such a ``v`` is a *nontrivial* logical iff it
fails to commute with some logical basis element (for ``v in N(S)``, the
symplectic products against a paired logical basis are exactly its logical
coordinates).  Both inputs are local to ``T``, so each test costs an
elimination on the handful of checks that touch ``T``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .gf2 import (
    BinaryMatrix,
    as_binary_matrix,
    gf2_inverse,
    gf2_matmul,
    is_symplectic,
    nullspace,
    reduce_rows,
    rref,
    symplectic_product,
)
from .stabilizer import StabilizerCode


def spread(matrix: object, *, qubits: int) -> int:
    """The actual spread ``sigma(M)`` of a symplectic matrix.

    ``sigma(M) = max_i max_{P in {X_i, Y_i, Z_i}} wt(M P)``: the largest support
    an ideal single-qubit Pauli acquires.  It is at most the cell size of any
    partition ``M`` is block-diagonal for, and often far smaller -- a qubit
    permutation has ``sigma = 1`` on cells of any size.

    A small spread is a statement about the *ideal* layer only.  It bounds
    error propagation through a correct gate, never the damage a faulty native
    gate does to its own outputs; that is governed by the partition into native
    gate supports.  Use it to decide whether a coarse layer can be re-expressed
    on a finer partition, then compute :func:`partition_distance` there.
    """

    m = as_binary_matrix(matrix)
    if m.shape != (2 * qubits, 2 * qubits):
        raise ValueError("matrix must be 2n x 2n for the given qubit count")
    if qubits == 0:
        return 0
    worst = 0
    for i in range(qubits):
        x_image = m[i]
        z_image = m[qubits + i]
        for image in (x_image, z_image, (x_image ^ z_image)):
            support = image[:qubits] | image[qubits:]
            worst = max(worst, int(support.sum()))
    return worst


def _normalise_cells(cells: Iterable[Sequence[int]], n: int) -> list[tuple[int, ...]]:
    normalised = [tuple(sorted(int(q) for q in cell)) for cell in cells]
    covered = sorted(q for cell in normalised for q in cell)
    if covered != list(range(n)):
        raise ValueError("cells must partition the qubits exactly")
    return normalised


def singleton_cells(n: int) -> list[tuple[int, ...]]:
    """The strict-transversal partition: one cell per qubit."""

    return [(q,) for q in range(n)]


def matching_cells(tau: Sequence[int]) -> list[tuple[int, ...]]:
    """Cells of the involution ``tau``: matched pairs, plus fixed points."""

    n = len(tau)
    seen: set[int] = set()
    cells: list[tuple[int, ...]] = []
    for q in range(n):
        if q in seen:
            continue
        partner = int(tau[q])
        if partner == q:
            cells.append((q,))
            seen.add(q)
        else:
            if int(tau[partner]) != q:
                raise ValueError("tau is not an involution")
            cells.append(tuple(sorted((q, partner))))
            seen.update((q, partner))
    return cells


@dataclass(frozen=True)
class PartitionDistance:
    """Result of a partition-distance search up to ``searched_to`` cells."""

    cells: int
    max_cell_size: int
    #: exact ``d_P`` when a witness was found, else ``None``
    value: int | None
    #: certified ``d_P >= lower_bound`` in every case
    lower_bound: int
    #: how many cells the exhaustive search covered
    searched_to: int
    #: a minimal-block-weight logical operator ``(X | Z)``, when found
    witness: BinaryMatrix | None
    #: the cell indices its support meets
    witness_cells: tuple[int, ...]

    @property
    def single_fault_correctable(self) -> bool | None:
        """``True``/``False`` once the search reached two cells, else ``None``."""

        if self.value is not None and self.value <= 2:
            return False
        if self.searched_to >= 2:
            return True
        return None

    @property
    def correctable_faults(self) -> int | None:
        """``r`` such that any ``r`` faulty cells are correctable, if decided."""

        if self.value is not None:
            return (self.value - 1) // 2
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": self.cells,
            "max_cell_size": self.max_cell_size,
            "partition_distance": self.value,
            "lower_bound": self.lower_bound,
            "searched_to": self.searched_to,
            "single_fault_correctable": self.single_fault_correctable,
            "witness": None
            if self.witness is None
            else "".join(str(int(b)) for b in self.witness),
            "witness_cells": list(self.witness_cells),
        }


def _logical_supported_on(code: StabilizerCode, qubits: np.ndarray) -> BinaryMatrix | None:
    """A Pauli in ``N(S) \\ S`` supported inside ``qubits``, or ``None``.

    ``N_T`` is the kernel of the symplectic pairing against the checks that
    touch ``T`` (checks disjoint from ``T`` pair trivially with everything
    supported there, so they contribute no constraint).  Membership in ``S`` is
    then read off the logical coordinates: for ``v in N(S)``, ``v`` lies in
    ``S`` exactly when it commutes with every logical basis element.
    """

    n = code.n
    if qubits.size == 0 or code.k == 0:
        return None
    x_part = code.h[:, :n][:, qubits]
    z_part = code.h[:, n:][:, qubits]
    touching = np.flatnonzero((x_part | z_part).any(axis=1))
    # <v, s> = v_x . s_z + v_z . s_x, in local (x_T | z_T) coordinates
    constraints = np.hstack([z_part[touching], x_part[touching]])
    if constraints.shape[0] == 0:
        kernel = np.eye(2 * qubits.size, dtype=np.uint8)
    else:
        kernel = nullspace(constraints)
    if kernel.shape[0] == 0:
        return None
    lx = code.logical[:, :n][:, qubits]
    lz = code.logical[:, n:][:, qubits]
    pairing = np.hstack([lz, lx])  # <v, L> in the same local coordinates
    coordinates = (kernel @ pairing.T) % 2
    nontrivial = np.flatnonzero(coordinates.any(axis=1))
    if nontrivial.size == 0:
        return None
    local = kernel[nontrivial[0]]
    witness = np.zeros(2 * n, dtype=np.uint8)
    witness[qubits] = local[: qubits.size]
    witness[n + qubits] = local[qubits.size :]
    return witness


def partition_distance(
    code: StabilizerCode,
    cells: Iterable[Sequence[int]],
    *,
    max_blocks: int = 2,
) -> PartitionDistance:
    """Exhaustively search for a logical operator of block weight ``<= max_blocks``.

    Returns the exact ``d_P`` with a witness when one exists at that block
    weight, and otherwise the certificate ``d_P > max_blocks`` -- both verdicts
    are decided, because the search over cell subsets of that size is complete.
    ``max_blocks = 2`` (the default) settles the fault-tolerance criterion
    ``d_P >= 3`` for a single unflagged faulty gate; the cost is
    ``O(m^max_blocks)`` local eliminations, so raising it is only sensible for
    coarse partitions.
    """

    cell_list = _normalise_cells(cells, code.n)
    m = len(cell_list)
    if max_blocks < 1:
        raise ValueError("max_blocks must be at least 1")
    reach = min(max_blocks, m)
    arrays = [np.asarray(cell, dtype=int) for cell in cell_list]
    max_cell = max((len(cell) for cell in cell_list), default=0)
    for size in range(1, reach + 1):
        for choice in combinations(range(m), size):
            support = np.concatenate([arrays[i] for i in choice])
            witness = _logical_supported_on(code, support)
            if witness is not None:
                touched = tuple(
                    i
                    for i in choice
                    if (witness[arrays[i]] | witness[code.n + arrays[i]]).any()
                )
                return PartitionDistance(
                    cells=m,
                    max_cell_size=max_cell,
                    value=len(touched),
                    lower_bound=len(touched),
                    searched_to=size,
                    witness=witness,
                    witness_cells=touched,
                )
    return PartitionDistance(
        cells=m,
        max_cell_size=max_cell,
        value=None,
        lower_bound=reach + 1,
        searched_to=reach,
        witness=None,
        witness_cells=(),
    )


# ---------------------------------------------------------------------------
# constructive side: realizing a target logical Clifford on a given partition
# ---------------------------------------------------------------------------


def _solve_gf2(system: BinaryMatrix, rhs: BinaryMatrix) -> BinaryMatrix | None:
    augmented = np.hstack([system, np.asarray(rhs, dtype=np.uint8).reshape(-1, 1)])
    reduced, pivots = rref(augmented)
    width = system.shape[1]
    if width in pivots:
        return None
    solution = np.zeros(width, dtype=np.uint8)
    for row, pivot in zip(reduced, pivots):
        solution[pivot] = row[width]
    return solution


def destabilizers(code: StabilizerCode) -> BinaryMatrix:
    """Rows ``D_i`` completing the checks to a symplectic basis.

    ``<D_i, h_j> = delta_ij``, ``<D_i, D_j> = 0``, and ``<D_i, L> = 0`` for
    every logical basis row, so ``[h ; D ; logical]`` is a basis of ``F_2^{2n}``
    in which the code's structure is standard.
    """

    n = code.n
    found: list[np.ndarray] = []
    for i in range(code.rank):
        known = [code.h, code.logical] + (
            [np.asarray(found, dtype=np.uint8)] if found else []
        )
        rows = np.vstack(known)
        # <d, v> = d . swap(v), so the constraint matrix is swap(rows) transposed
        constraints = np.hstack([rows[:, n:], rows[:, :n]])
        target = np.zeros(constraints.shape[0], dtype=np.uint8)
        target[i] = 1
        solution = _solve_gf2(constraints, target)
        if solution is None:  # pragma: no cover - the system is always solvable
            raise AssertionError("destabilizer system unsolvable")
        found.append(solution)
    return np.asarray(found, dtype=np.uint8).reshape(code.rank, 2 * n)


def logical_action(code: StabilizerCode, matrix: object) -> BinaryMatrix:
    """The ``2k x 2k`` action induced on ``S^perp / S`` by a physical symplectic."""

    n, k = code.n, code.k
    if k == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    images = gf2_matmul(code.logical, as_binary_matrix(matrix))
    x_coeff = symplectic_product(images, code.logical[k:], qubits=n)
    z_coeff = symplectic_product(images, code.logical[:k], qubits=n)
    return np.hstack([x_coeff, z_coeff]).astype(np.uint8)


def preserves_code(code: StabilizerCode, matrix: object) -> bool:
    """Does the physical symplectic map the stabilizer row space to itself?"""

    image = gf2_matmul(code.h, as_binary_matrix(matrix))
    return not reduce_rows(image, *code._h_rref).any()


def encoded_lift(code: StabilizerCode, target: object) -> BinaryMatrix:
    """A physical symplectic preserving ``S`` and inducing ``target`` on the logicals.

    Every logical Clifford is realizable this way: fix the checks and their
    destabilizers, move the logical basis by ``target``, and read off the
    change of basis.  The lift is generally a full ``n``-qubit Clifford, so it
    is a *building block* for a coarse-partition layer (apply it inside one
    cell), never a fault-tolerance claim by itself -- that is what
    :func:`partition_distance` decides for the resulting partition.
    """

    n, k = code.n, code.k
    action = as_binary_matrix(target)
    if action.shape != (2 * k, 2 * k):
        raise ValueError("target must be a 2k x 2k symplectic matrix")
    if k and not is_symplectic(action, qubits=k):
        raise ValueError("target is not symplectic")
    anti = destabilizers(code)
    basis = np.vstack([code.h, anti, code.logical])
    images = np.vstack([code.h, anti, gf2_matmul(action, code.logical)])
    lift = gf2_matmul(gf2_inverse(basis), images)
    if not is_symplectic(lift, qubits=n):  # pragma: no cover - guarded by construction
        raise AssertionError("lift is not symplectic")
    return lift


def block_diagonal(
    blocks: Sequence[object], cells: Iterable[Sequence[int]], n: int
) -> BinaryMatrix:
    """Assemble per-cell symplectic blocks into one ``2n x 2n`` layer matrix."""

    cell_list = _normalise_cells(cells, n)
    if len(blocks) != len(cell_list):
        raise ValueError("one block per cell is required")
    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for cell, block in zip(cell_list, blocks):
        array = as_binary_matrix(block)
        width = len(cell)
        if array.shape != (2 * width, 2 * width):
            raise ValueError("block shape does not match its cell")
        index = np.asarray(cell, dtype=int)
        coordinates = np.concatenate([index, n + index])
        matrix[np.ix_(coordinates, coordinates)] = array
    return matrix
