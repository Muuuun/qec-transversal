r"""The unitary group of an ``F_2``-algebra with involution, without a sweep.

For an algebra ``A`` carrying an involution ``sigma`` (an anti-automorphism
with ``sigma^2 = id``), the object the gate solvers actually want is the
*unitary group*

.. math::

    G \;=\; \{\, u \in A : \sigma(u)\,u = 1 \,\} \;=\;
    A^{\times} \cap \textstyle\prod_C \mathrm{Sp}(2|C|, 2)

for the partition algebra, where ``sigma(M) = Omega M^T Omega`` blockwise.
Note ``G \subseteq A^{\times}`` for free: ``sigma(u) u = 1`` exhibits
``sigma(u) \in A`` as the inverse.

Until now the cut was taken by *enumerating* ``A^{\times}`` and filtering,
which costs ``|A^{\times}|`` steps -- 393216 of them to find the 6144 elements
of the ``[[8,3,2]]`` pair group, and hopeless once ``|A^{\times}|`` reaches
``10^9``.  This module replaces that sweep with an orbit computation.

**The map.**  Put ``phi(u) = sigma(u) u``.  Its fibers are exactly the left
cosets of ``G``: ``phi(wu) = sigma(u) sigma(w) w u`` equals ``phi(u)`` iff
``sigma(w) w = 1``.  Hence ``|A^{\times}| = |G| \cdot |\mathrm{im}\, phi|``.

**The action.**  ``phi`` is the orbit map of the *congruence action* of
``A^{\times}`` on ``A``,

.. math::

    a \cdot u \;=\; \sigma(u)\, a\, u ,

a right action because ``sigma`` is an anti-automorphism.  The orbit of ``1``
is precisely ``im phi`` and its stabilizer is precisely ``G``, so
orbit-stabilizer *is* the index formula, and it comes with a transversal:
Schreier's lemma turns that transversal into generators of ``G``.

Two properties make this cheap where the sweep was not.  The orbit lies in the
``sigma``-symmetric part of ``A`` -- roughly half the dimension -- and its size
is ``|A^{\times}| / |G|``, so a *large* gate group makes the orbit *small*.
The action of a fixed ``u`` is ``F_2``-linear in ``a``, so each orbit step is
one precomputed matrix-vector product over ``F_2``, run here on bit-packed
integers.

Every exit is honest: ``status = "exact"`` only when the orbit closed inside
its cap and every returned generator was verified to satisfy
``sigma(w) w = 1`` on the nose.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..utils.gf2 import solve_gf2
from .finite_algebra import AlgebraF2

#: Orbit points held before giving up.  The orbit is ``|A^x| / |G|``, so this
#: bites only when the gate group is tiny inside a huge unit group.
DEFAULT_ORBIT_CAP = 2_000_000

#: Transversal points visited while harvesting Schreier generators.  Reaching
#: it costs only the *completeness* of the generating set: ``|G|`` itself comes
#: from the index and stays exact.
DEFAULT_SCHREIER_CAP = 20_000


@dataclass
class UnitaryGroupResult:
    """Outcome of the orbit-stabilizer cut.

    ``order`` is exact whenever ``status == "exact"``.  ``generators_complete``
    says whether the returned set is *proven* to generate ``G`` -- by Schreier's
    lemma (the whole transversal was processed) or by an order probe that
    matched ``order``.  When it is ``False`` the generators still all lie in
    ``G``; they just are not known to fill it.
    """

    status: str  # "exact" | "unknown"
    order: int | None = None
    generators: list[np.ndarray] = field(default_factory=list)  # coords in A
    generators_complete: bool = False
    orbit_size: int | None = None
    checks: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "order": self.order,
            "orbit_size": self.orbit_size,
            "generator_count": len(self.generators),
            "generators_complete": self.generators_complete,
            "checks": dict(self.checks),
            "detail": self.detail,
        }


def _pack(coords: np.ndarray) -> int:
    value = 0
    for index in np.flatnonzero(np.asarray(coords, dtype=np.uint8)):
        value |= 1 << int(index)
    return value


def _unpack(value: int, dim: int) -> np.ndarray:
    coords = np.zeros(dim, dtype=np.uint8)
    while value:
        low = value & -value
        coords[low.bit_length() - 1] = 1
        value ^= low
    return coords


def _bits(value: int):
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low


def _columns(matrix: np.ndarray) -> list[int]:
    """Column ``j`` of a GF(2) matrix as an integer with bit ``i`` = ``M[i, j]``."""

    return [_pack(matrix[:, j]) for j in range(matrix.shape[1])]


def _verify_involution(
    algebra: AlgebraF2, sigma: np.ndarray, *, samples: int = 24, seed: int = 5
) -> dict[str, Any]:
    """Checks that ``sigma`` really is an involution of ``A`` in coordinates.

    Anti-multiplicativity of the blockwise ``Omega M^T Omega`` map holds in the
    ambient product of matrix algebras as a matter of algebra, and ``A`` is a
    subalgebra, so stability of ``A`` under ``sigma`` (which is what having a
    coordinate matrix at all certifies) is the substantive per-instance fact.
    The random pairs below are a defensive check on the encoding, not the proof.
    """

    dimension = algebra.dim
    checks: dict[str, Any] = {"sigma_square_is_identity": False, "sigma_fixes_one": False}
    if sigma.shape != (dimension, dimension):
        return checks
    identity = np.eye(dimension, dtype=np.uint8)
    checks["sigma_square_is_identity"] = bool(
        np.array_equal((sigma @ sigma) % 2, identity)
    )
    checks["sigma_fixes_one"] = bool(
        np.array_equal((sigma @ algebra.one_coords) % 2, algebra.one_coords)
    )
    rng = np.random.default_rng(seed)
    ok = True
    for _ in range(samples):
        a = rng.integers(0, 2, size=dimension, dtype=np.uint8)
        b = rng.integers(0, 2, size=dimension, dtype=np.uint8)
        left = (sigma @ algebra.coords_multiply(a, b)) % 2
        right = algebra.coords_multiply((sigma @ b) % 2, (sigma @ a) % 2)
        if not np.array_equal(left, right):
            ok = False
            break
    checks["anti_multiplicative_on_samples"] = bool(ok)
    return checks


def unitary_group(
    algebra: AlgebraF2,
    sigma: np.ndarray,
    *,
    unit_order: int,
    unit_generators: list[np.ndarray],
    orbit_cap: int = DEFAULT_ORBIT_CAP,
    schreier_cap: int = DEFAULT_SCHREIER_CAP,
    order_probe: Callable[[list[np.ndarray]], int | None] | None = None,
    probe_every: int = 4,
) -> UnitaryGroupResult:
    r"""Exact ``|G|`` and generators of ``G = {u : sigma(u) u = 1}``.

    ``sigma`` is the ``d x d`` coordinate matrix of the involution, so that
    ``sigma @ coords(x) = coords(sigma(x))``; ``unit_order`` and
    ``unit_generators`` are the certified output of
    :func:`.unit_group.unit_group`.

    The orbit of ``1`` under ``a . u = sigma(u) a u`` is computed by breadth
    first search with a transversal, giving ``|G| = |A^x| / |orbit|`` and,
    through Schreier's lemma, generators.  ``order_probe`` (optional) is called
    with the generators collected so far and should return the order of the
    group they generate; as soon as it returns ``order`` the generating set is
    certified complete and the walk stops early.
    """

    dimension = algebra.dim
    sigma = (np.asarray(sigma, dtype=np.uint8) & 1).copy()
    checks = _verify_involution(algebra, sigma)
    if not all(bool(value) for value in checks.values()):
        return UnitaryGroupResult(
            status="unknown",
            checks=checks,
            detail="sigma failed its involution checks",
        )
    generators = [
        (np.asarray(g, dtype=np.uint8) & 1).copy() for g in unit_generators
    ]
    if not generators:
        return UnitaryGroupResult(
            status="unknown", checks=checks, detail="no unit generators supplied"
        )

    one_coords = algebra.one_coords
    one_int = _pack(one_coords)

    # Structure constants as integers: product[i][j] = coords(e_i e_j).
    product_table = [_columns(algebra.left[i]) for i in range(dimension)]

    def multiply(a: int, b: int) -> int:
        out = 0
        for i in _bits(a):
            row = product_table[i]
            for j in _bits(b):
                out ^= row[j]
        return out

    def inverse(a: int) -> int | None:
        matrix = algebra.left_matrix(_unpack(a, dimension))
        solution = solve_gf2(matrix, one_coords)
        return None if solution is None else _pack(np.asarray(solution).reshape(-1))

    def sigma_of(value: int) -> int:
        return _pack((sigma @ _unpack(value, dimension)) % 2)

    # a -> sigma(g) a g is linear in a: one precomputed matrix per generator,
    # so an orbit step is a GF(2) matrix product, not an algebra product.
    action_matrices: list[np.ndarray] = []
    generator_ints: list[int] = []
    for g in generators:
        left = algebra.left_matrix((sigma @ g) % 2)
        right = algebra.right_matrix(g)
        action_matrices.append((left @ right) % 2)
        generator_ints.append(_pack(g))
    transposed = [np.ascontiguousarray(matrix.T) for matrix in action_matrices]

    one_row = np.asarray(one_coords, dtype=np.uint8).reshape(1, -1)
    if dimension <= 63:
        weights = (1 << np.arange(dimension, dtype=np.uint64)).astype(np.uint64)

        def keys_of(rows: np.ndarray) -> list:
            return (rows.astype(np.uint64) @ weights).tolist()

    else:  # pragma: no cover - only for algebras wider than any solved here

        def keys_of(rows: np.ndarray) -> list:
            return [row.tobytes() for row in np.packbits(rows, axis=1)]

    # --- the orbit of 1 --------------------------------------------------
    # Bulk breadth-first search: one GF(2) matrix product per (level,
    # generator).  Only the dedup keys and a breadth-first *tree* (four bytes
    # of parent plus two of generator per point) are retained; the element at
    # a point is recovered from the tree on demand, so the memory cost of a
    # million-point orbit is the dict, not a million algebra elements.
    index_of = {keys_of(one_row)[0]: 0}
    parent = array("i", [-1])
    parent_gen = array("h", [-1])
    total = 1
    frontier = one_row
    frontier_index = np.zeros(1, dtype=np.int64)
    while frontier.shape[0]:
        blocks: list[np.ndarray] = []
        block_index: list[np.ndarray] = []
        for gen_index, matrix in enumerate(transposed):
            images = (frontier @ matrix) & 1
            fresh: list[int] = []
            for position, key in enumerate(keys_of(images)):
                if key in index_of:
                    continue
                index_of[key] = total
                parent.append(int(frontier_index[position]))
                parent_gen.append(gen_index)
                fresh.append(position)
                total += 1
                if total > orbit_cap:
                    return UnitaryGroupResult(
                        status="unknown",
                        orbit_size=None,
                        checks=checks,
                        detail=(
                            f"congruence orbit of 1 exceeded the cap of "
                            f"{orbit_cap} points (|A^x| = {unit_order}); the "
                            "gate group is small inside a very large unit group"
                        ),
                    )
            if fresh:
                # a fancy-index take copies, so a level cannot pin the whole
                # image array it was sliced from
                blocks.append(images[np.asarray(fresh, dtype=np.int64)])
                block_index.append(
                    np.arange(total - len(fresh), total, dtype=np.int64)
                )
        if not blocks:
            break
        frontier = np.concatenate(blocks, axis=0)
        frontier_index = np.concatenate(block_index)
    orbit_size = total
    checks["orbit_closed"] = True

    if unit_order % orbit_size:
        return UnitaryGroupResult(
            status="unknown",
            orbit_size=orbit_size,
            checks=checks,
            detail=(
                f"|A^x| = {unit_order} is not divisible by the orbit size "
                f"{orbit_size}; orbit-stabilizer violated, refusing to report"
            ),
        )
    order = unit_order // orbit_size
    checks["orbit_stabilizer_divides"] = True

    # --- Schreier generators of the stabilizer ---------------------------
    words: dict[int, int] = {0: one_int}

    def transversal_of(index: int) -> int:
        """The element carrying 1 to orbit point ``index``, from the BFS tree."""

        cached = words.get(index)
        if cached is not None:
            return cached
        path: list[int] = []
        cursor = index
        while cursor not in words:
            path.append(cursor)
            cursor = parent[cursor]
        element = words[cursor]
        for node in reversed(path):
            element = multiply(element, generator_ints[parent_gen[node]])
            words[node] = element
        return element

    def row_of(index: int) -> np.ndarray:
        witness = transversal_of(index)
        return _unpack(multiply(sigma_of(witness), witness), dimension)

    # Visit the whole transversal when it is affordable (Schreier's lemma then
    # certifies generation); otherwise walk an evenly spaced sample of it, which
    # spreads the generators over the orbit instead of hugging the identity.
    if orbit_size <= max(1, schreier_cap):
        positions = range(orbit_size)
        exhaustive = True
    else:
        stride = -(-orbit_size // max(1, schreier_cap))
        positions = range(0, orbit_size, stride)
        exhaustive = False

    collected: list[np.ndarray] = []
    seen: set[int] = {one_int}
    inverses: dict[int, int] = {0: one_int}
    complete = False
    visited = 0
    for position in positions:
        witness = transversal_of(position)
        point = row_of(position)
        for gen_index in range(len(generator_ints)):
            image = (point @ transposed[gen_index]) & 1
            target = index_of[keys_of(image.reshape(1, -1))[0]]
            if target not in inverses:
                back = inverse(transversal_of(target))
                if back is None:
                    return UnitaryGroupResult(
                        status="unknown",
                        orbit_size=orbit_size,
                        checks=checks,
                        detail="a transversal element was not invertible",
                    )
                inverses[target] = back
            element = multiply(
                multiply(witness, generator_ints[gen_index]), inverses[target]
            )
            if element in seen:
                continue
            seen.add(element)
            if multiply(sigma_of(element), element) != one_int:
                return UnitaryGroupResult(
                    status="unknown",
                    orbit_size=orbit_size,
                    checks=checks,
                    detail=(
                        "a Schreier generator failed sigma(w) w = 1; the "
                        "involution or the transversal is inconsistent"
                    ),
                )
            collected.append(_unpack(element, dimension))
        visited += 1
        if order_probe is not None and collected and visited % probe_every == 0:
            if order_probe(collected) == order:  # it already fills G
                complete = True
                break
    else:
        complete = exhaustive  # Schreier's lemma needs the whole transversal
    checks["schreier_generators_verified"] = True
    checks["transversal_points_visited"] = visited
    checks["generators_complete"] = bool(complete)
    return UnitaryGroupResult(
        status="exact",
        order=order,
        generators=collected,
        generators_complete=complete,
        orbit_size=orbit_size,
        checks=checks,
        detail=(
            f"|G| = |A^x| / |orbit of 1| = {unit_order} / {orbit_size}; "
            + (
                "generators complete by Schreier's lemma"
                if complete
                else (
                    "generating set is a certified subgroup only: "
                    f"{visited} of {orbit_size} transversal points visited"
                )
            )
        ),
    )
