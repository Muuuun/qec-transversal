"""Word-tracking Schreier-Sims chain: membership with an explicit witness.

The order engines of :mod:`.group` answer "how big is the group"; this chain
answers "is this target in the group, and if so, as which product of the
collected generators".  It reaches a verified fixpoint (every Schreier
generator sifts to the identity), which certifies both the order and the
completeness of membership testing: a target that fails to sift is *provably*
outside the group, so a ``None`` answer is a certified non-membership rather
than a failed search.
"""

from __future__ import annotations

import numpy as np

from ..utils.gf2 import gf2_inverse, gf2_matmul
from ..utils.polynomials import _matrix_order
from .group import _apply, _pack_rows


class _WordLevel:
    __slots__ = ("base", "gens", "orbit")

    def __init__(self, base: int):
        self.base = base
        # strong generators: (matrix, packed rows, word)
        self.gens: list[tuple[np.ndarray, tuple[int, ...], list[int]]] = []
        # orbit point -> (element u with base*u = point, u^-1, word of u)
        self.orbit: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}


class WordBSGS:
    """Deterministic Schreier-Sims chain that tracks generator words.

    The chain reaches a verified fixpoint (every Schreier generator sifts to
    the identity), which certifies both the group order and the completeness
    of membership testing: a target that fails to sift is provably outside
    the group.  Words are sequences of generator indices whose left-to-right
    matrix product reproduces the element; generator inverses are expanded as
    ``g^(order-1)`` using exact element orders, so words can be long — the
    caller records lengths and does not optimize.
    """

    _WORD_LIMIT = 2_000_000

    def __init__(self, generators: list[np.ndarray], *, node_budget: int = 2_000_000):
        self.generators = [
            (np.asarray(g, dtype=np.uint8) & 1).copy() for g in generators
        ]
        if not self.generators:
            raise ValueError("WordBSGS needs at least one generator")
        self.dimension = self.generators[0].shape[0]
        self.identity = np.eye(self.dimension, dtype=np.uint8)
        self.orders = [_matrix_order(g) for g in self.generators]
        self.node_budget = node_budget
        self.levels: list[_WordLevel] = []
        self._build()

    # -- word arithmetic ---------------------------------------------------

    def _inverse_word(self, word: list[int]) -> list[int]:
        out: list[int] = []
        for index in reversed(word):
            out.extend([index] * (self.orders[index] - 1))
        if len(out) > self._WORD_LIMIT:
            raise RuntimeError("synthesis word grew past the safety limit")
        return out

    def word_product(self, word: list[int]) -> np.ndarray:
        result = self.identity.copy()
        for index in word:
            result = gf2_matmul(result, self.generators[index])
        return result

    # -- chain construction ------------------------------------------------

    def _moved_basis_vector(self, matrix: np.ndarray) -> int:
        for index in range(self.dimension):
            expected = np.zeros(self.dimension, dtype=np.uint8)
            expected[index] = 1
            if not np.array_equal(matrix[index], expected):
                return 1 << index
        raise AssertionError("non-identity matrix moves some basis vector")

    def _rebuild_orbit(self, level: _WordLevel) -> None:
        level.orbit = {level.base: (self.identity, self.identity, [])}
        queue = [level.base]
        while queue:
            point = queue.pop()
            element, _, word = level.orbit[point]
            for matrix, packed, gen_word in level.gens:
                image = _apply(packed, point)
                if image not in level.orbit:
                    if sum(len(lvl.orbit) for lvl in self.levels) >= self.node_budget:
                        raise RuntimeError("WordBSGS exceeded its node budget")
                    product = gf2_matmul(element, matrix)
                    new_word = word + gen_word
                    if len(new_word) > self._WORD_LIMIT:
                        raise RuntimeError("synthesis word grew past the safety limit")
                    level.orbit[image] = (product, gf2_inverse(product), new_word)
                    queue.append(image)

    def _sift(
        self, matrix: np.ndarray, word: list[int]
    ) -> tuple[np.ndarray | None, list[int], int]:
        """Reduce through the chain: returns (residue, residue word, level).

        Residue ``None`` means the element sifted to the identity; the
        returned word then satisfies ``product(word(u_L) .. word(u_1)) =
        element`` and is reconstructed by :meth:`factor`.
        """

        current = matrix
        current_word = list(word)
        for index, level in enumerate(self.levels):
            packed = _pack_rows(current)
            image = _apply(packed, level.base)
            entry = level.orbit.get(image)
            if entry is None:
                return current, current_word, index
            current = gf2_matmul(current, entry[1])
            current_word = current_word + self._inverse_word(entry[2])
        if np.array_equal(current, self.identity):
            return None, current_word, len(self.levels)
        return current, current_word, len(self.levels)

    def _add_strong_generator(self, matrix: np.ndarray, word: list[int], level_index: int) -> None:
        if level_index == len(self.levels):
            self.levels.append(_WordLevel(self._moved_basis_vector(matrix)))
        packed = _pack_rows(matrix)
        for level in self.levels[: level_index + 1]:
            level.gens.append((matrix, packed, word))
        for level in self.levels[: level_index + 1]:
            self._rebuild_orbit(level)

    def _build(self) -> None:
        for index, matrix in enumerate(self.generators):
            if np.array_equal(matrix, self.identity):
                continue
            residue, residue_word, level_index = self._sift(matrix, [index])
            if residue is not None:
                self._add_strong_generator(residue, residue_word, level_index)
        level_index = 0
        while level_index < len(self.levels):
            level = self.levels[level_index]
            restart = False
            for point, (point_element, _inv, point_word) in list(level.orbit.items()):
                for matrix, packed, gen_word in level.gens:
                    image = _apply(packed, point)
                    image_entry = level.orbit.get(image)
                    if image_entry is None:
                        self._rebuild_orbit(level)
                        restart = True
                        break
                    schreier = gf2_matmul(gf2_matmul(point_element, matrix), image_entry[1])
                    schreier_word = point_word + gen_word + self._inverse_word(image_entry[2])
                    residue, residue_word, residue_level = self._sift(schreier, schreier_word)
                    if residue is not None:
                        self._add_strong_generator(residue, residue_word, residue_level)
                        restart = True
                        break
                if restart:
                    break
            if restart:
                continue
            level_index += 1

    # -- public interface --------------------------------------------------

    def order(self) -> int:
        order = 1
        for level in self.levels:
            order *= len(level.orbit)
        return order

    def factor(self, target: np.ndarray) -> list[int] | None:
        """A word with left-to-right product ``target``, or ``None``.

        ``None`` is a *certified* non-membership: the chain passed its
        deterministic fixpoint, so sifting is a complete membership test.
        """

        current = (np.asarray(target, dtype=np.uint8) & 1).copy()
        if current.shape != (self.dimension, self.dimension):
            raise ValueError("target has the wrong shape")
        transversal_words: list[list[int]] = []
        for level in self.levels:
            packed = _pack_rows(current)
            image = _apply(packed, level.base)
            entry = level.orbit.get(image)
            if entry is None:
                return None
            transversal_words.append(entry[2])
            current = gf2_matmul(current, entry[1])
        if not np.array_equal(current, self.identity):
            return None
        word: list[int] = []
        for transversal_word in reversed(transversal_words):
            word.extend(transversal_word)
        return word
