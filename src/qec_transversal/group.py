"""Exact matrix-group orders for logical symplectic images.

Two engines are provided.  :func:`schreier_sims_order` computes the exact
order of a group of GF(2) matrices acting on row vectors through a
stabilizer chain, which handles groups far too large to enumerate (the
Steane-style full logical Clifford groups, ``|Sp(2k, 2)|`` for ``k`` up to
about eight).  :func:`generated_group_order` is the original breadth-first
element enumeration, kept as a cross-check and as a fallback when the chain
exceeds its node budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import BinaryMatrix, gf2_inverse, gf2_matmul


def symplectic_group_order(logical_qubits: int) -> int:
    """Return ``|Sp(2k, 2)|``."""

    if logical_qubits < 0:
        raise ValueError("logical_qubits must be non-negative")
    order = 2 ** (logical_qubits * logical_qubits)
    for index in range(1, logical_qubits + 1):
        order *= 4**index - 1
    return order


@dataclass(frozen=True)
class GroupOrder:
    exact: bool
    order: int | None
    lower_bound: int


def generated_group_order(generators: list[BinaryMatrix], *, cap: int = 100_000) -> GroupOrder:
    """Compute an exact group order by closure, stopping after ``cap`` elements.

    When the cap is exceeded the truncated count is reported honestly as a
    lower bound; :func:`schreier_sims_order` is the engine that can still be
    exact in that regime.
    """

    if cap < 1:
        raise ValueError("cap must be positive")
    if generators:
        shape = generators[0].shape
        if shape[0] != shape[1] or any(generator.shape != shape for generator in generators):
            raise ValueError("generators must be square matrices of the same size")
        size = shape[0]
    else:
        size = 0

    identity = np.eye(size, dtype=np.uint8)
    seen: set[bytes] = {identity.tobytes()}
    queue = [identity]
    cursor = 0
    while cursor < len(queue):
        element = queue[cursor]
        cursor += 1
        for generator in generators:
            product = gf2_matmul(element, generator)
            key = product.tobytes()
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > cap:
                return GroupOrder(exact=False, order=None, lower_bound=len(seen))
            queue.append(product)
    return GroupOrder(exact=True, order=len(seen), lower_bound=len(seen))


def _pack_rows(matrix: BinaryMatrix) -> tuple[int, ...]:
    """Each matrix row as an integer bitmask, for fast vector actions."""

    return tuple(int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little") for row in matrix)


def _apply(rows: tuple[int, ...], vector: int) -> int:
    """Image of a row-vector bitmask under the matrix with packed ``rows``."""

    image = 0
    remaining = vector
    index = 0
    while remaining:
        if remaining & 1:
            image ^= rows[index]
        remaining >>= 1
        index += 1
    return image


class _BudgetExceeded(Exception):
    """Raised when a stabilizer-chain orbit outgrows the node budget."""


class _Level:
    """One stabilizer-chain level: a base vector, strong generators fixing
    all earlier bases, and the orbit transversal of the base under them."""

    __slots__ = ("base", "gens", "orbit")

    def __init__(self, base: int):
        self.base = base
        self.gens: list[tuple[BinaryMatrix, tuple[int, ...]]] = []
        # orbit point -> (transversal element u with base*u = point, u^-1)
        self.orbit: dict[int, tuple[BinaryMatrix, BinaryMatrix]] = {}

    def rebuild_orbit(self, dimension: int, limit: int) -> None:
        identity = np.eye(dimension, dtype=np.uint8)
        self.orbit = {self.base: (identity, identity)}
        queue = [self.base]
        while queue:
            point = queue.pop()
            element = self.orbit[point][0]
            for matrix, rows in self.gens:
                image = _apply(rows, point)
                if image not in self.orbit:
                    if len(self.orbit) >= limit:
                        raise _BudgetExceeded
                    product = (element @ matrix) & 1
                    self.orbit[image] = (product, gf2_inverse(product))
                    queue.append(image)


def schreier_sims_order(
    generators: list[BinaryMatrix], *, node_budget: int = 2_000_000
) -> int | None:
    """Exact order of ``<generators>`` acting on row vectors of ``F_2^d``.

    A deterministic Schreier-Sims stabilizer chain: the algorithm reaches a
    fixpoint exactly when every Schreier generator sifts to the identity,
    which by Schreier's lemma certifies ``order = prod(orbit sizes)``.
    Returns ``None`` when the chain grows past ``node_budget`` transversal
    points, in which case the caller should fall back to enumeration.
    """

    filtered = []
    for generator in generators:
        matrix = (np.asarray(generator, dtype=np.uint8) & 1).copy()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("generators must be square matrices")
        if not np.array_equal(matrix, np.eye(matrix.shape[0], dtype=np.uint8)):
            filtered.append(matrix)
    if not filtered:
        return 1
    dimension = filtered[0].shape[0]
    if any(matrix.shape[0] != dimension for matrix in filtered):
        raise ValueError("generators must share one dimension")

    identity = np.eye(dimension, dtype=np.uint8)
    levels: list[_Level] = []

    def moved_basis_vector(matrix: BinaryMatrix) -> int:
        for index in range(dimension):
            expected = np.zeros(dimension, dtype=np.uint8)
            expected[index] = 1
            if not np.array_equal(matrix[index], expected):
                return 1 << index
        raise AssertionError("non-identity matrix moves some basis vector")

    def add_strong_generator(matrix: BinaryMatrix, level_index: int) -> None:
        """Install a generator known to fix the first ``level_index`` bases."""

        if level_index == len(levels):
            levels.append(_Level(moved_basis_vector(matrix)))
        packed = _pack_rows(matrix)
        for level in levels[: level_index + 1]:
            level.gens.append((matrix, packed))
        for level in levels[: level_index + 1]:
            level.rebuild_orbit(dimension, node_budget)

    def sift(matrix: BinaryMatrix) -> tuple[BinaryMatrix | None, int]:
        """Reduce through the chain; returns (residue, level) with residue
        ``None`` when the element factors completely over the chain."""

        current = matrix
        rows = _pack_rows(current)
        for index, level in enumerate(levels):
            image = _apply(rows, level.base)
            entry = level.orbit.get(image)
            if entry is None:
                return current, index
            current = (current @ entry[1]) & 1
            rows = _pack_rows(current)
        if np.array_equal(current, identity):
            return None, len(levels)
        return current, len(levels)

    try:
        for matrix in filtered:
            residue, level_index = sift(matrix)
            if residue is not None:
                add_strong_generator(residue, level_index)

        # Fixpoint pass: every Schreier generator of every level must sift to
        # the identity through the deeper chain.  Each failure installs a new
        # strong generator and restarts verification from that level.
        level_index = 0
        while level_index < len(levels):
            level = levels[level_index]
            if sum(len(item.orbit) for item in levels) > node_budget:
                return None
            restart = False
            for point, (point_element, _point_inverse) in list(level.orbit.items()):
                for matrix, rows in level.gens:
                    image = _apply(rows, point)
                    image_entry = level.orbit.get(image)
                    if image_entry is None:
                        level.rebuild_orbit(dimension, node_budget)
                        restart = True
                        break
                    schreier = (point_element @ matrix) & 1
                    schreier = (schreier @ image_entry[1]) & 1
                    residue, residue_level = sift(schreier)
                    if residue is not None:
                        add_strong_generator(residue, residue_level)
                        restart = True
                        break
                if restart:
                    break
            if restart:
                continue
            level_index += 1
    except _BudgetExceeded:
        return None

    order = 1
    for level in levels:
        order *= len(level.orbit)
    return order
