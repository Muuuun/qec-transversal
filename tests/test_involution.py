r"""The involution ``sigma(M) = Omega M^T Omega`` and the sigma-stable algebra.

The gate group on a partition is the *unitary group* of an algebra with
involution,

    ``G = {M in A^x : sigma(M) M = 1}``,

but that description is only available if ``sigma`` maps the algebra to
itself.  It does not for the naive preservation algebra ``A(S)``: a short
computation gives ``sigma(A(S)) = A(N)`` with ``N`` the normalizer, so the
sigma-stable object is the intersection ``A'(S) = A(S) cap A(N)``.

Three claims are pinned here, in increasing order of what they cost if wrong:

1. ``A(S)`` really is *not* sigma-stable in general (otherwise the refinement
   would be pointless and someone would eventually delete it);
2. ``A'(S)`` *is* sigma-stable, on every code tested;
3. the refinement changes **no gate group** -- which is the correctness
   claim, since ``A'`` is strictly smaller and a bug here would silently
   shrink an answer.
"""

import numpy as np
import pytest

from qec_transversal import REGISTRY, CSSCode, StabilizerCode, five_qubit_code
from qec_transversal.algebra.preservation import (
    _local_symplectic_form,
    partition_algebra,
    symplectic_involution,
)
from qec_transversal.ansatz.partition import PartitionCliffordAnalysis
from qec_transversal.utils.gf2 import rank
from qec_transversal.utils.symplectic import symplectic_product


def _pairs(n: int) -> list[tuple[int, ...]]:
    cells: list[tuple[int, ...]] = [(2 * i, 2 * i + 1) for i in range(n // 2)]
    if n % 2:
        cells.append((n - 1,))
    return cells


def _singletons(n: int) -> list[tuple[int, ...]]:
    return [(q,) for q in range(n)]


def _in_span(basis: np.ndarray, vector: np.ndarray) -> bool:
    if basis.shape[0] == 0:
        return not vector.any()
    return rank(np.vstack([basis, vector[None, :]])) == basis.shape[0]


def _sigma_escapes(basis: np.ndarray, layout) -> int:
    return sum(
        1 for row in basis if not _in_span(basis, symplectic_involution(row, layout))
    )


def _stabilizer(name: str) -> StabilizerCode:
    return CSSCode(*REGISTRY[name].build()).to_stabilizer_code()


def _symplectic_units(algebra: np.ndarray, layout) -> set:
    """Every blockwise-symplectic element, by brute force (validation only)."""

    dim = algebra.shape[0]
    assert dim <= 22, "refusing an unaffordable brute force"
    forms = {width: _local_symplectic_form(width) for _, width in layout}
    found = set()
    for mask in range(1 << dim):
        entries = np.zeros(algebra.shape[1], dtype=np.uint8)
        for bit in range(dim):
            if (mask >> bit) & 1:
                entries ^= algebra[bit]
        for start, width in layout:
            size = 2 * width
            block = entries[start : start + size * size].reshape(size, size)
            if not np.array_equal((block @ forms[width] @ block.T) & 1, forms[width]):
                break
        else:
            found.add(entries.tobytes())
    return found


# ---------------------------------------------------------------------------
# 1. sigma is an involution, and it is a genuine anti-automorphism
# ---------------------------------------------------------------------------


def test_sigma_is_an_involution_and_reverses_products() -> None:
    rng = np.random.default_rng(20260820)
    layout = [(0, 2), (16, 1)]  # one pair cell then one singleton
    width = 16 + 4

    def multiply(a, b):
        out = np.zeros(width, dtype=np.uint8)
        for start, cell in layout:
            size = 2 * cell
            A = a[start : start + size * size].reshape(size, size)
            B = b[start : start + size * size].reshape(size, size)
            out[start : start + size * size] = (A @ B % 2).reshape(-1)
        return out

    for _ in range(50):
        a = rng.integers(0, 2, size=width, dtype=np.uint8)
        b = rng.integers(0, 2, size=width, dtype=np.uint8)
        assert np.array_equal(symplectic_involution(symplectic_involution(a, layout), layout), a)
        assert np.array_equal(
            symplectic_involution(multiply(a, b), layout),
            multiply(symplectic_involution(b, layout), symplectic_involution(a, layout)),
        )


def test_symplectic_elements_are_exactly_the_sigma_unitary_ones() -> None:
    """``M`` blockwise symplectic  <=>  ``sigma(M) M = 1``."""

    code = _stabilizer("c4-22")
    cells = _pairs(code.n)
    algebra, layout = partition_algebra(code, cells)
    forms = {width: _local_symplectic_form(width) for _, width in layout}
    one = np.zeros(algebra.shape[1], dtype=np.uint8)
    for start, width in layout:
        size = 2 * width
        one[start : start + size * size] = np.eye(size, dtype=np.uint8).reshape(-1)

    def multiply(a, b):
        out = np.zeros_like(a)
        for start, width in layout:
            size = 2 * width
            A = a[start : start + size * size].reshape(size, size)
            B = b[start : start + size * size].reshape(size, size)
            out[start : start + size * size] = (A @ B % 2).reshape(-1)
        return out

    checked = 0
    for mask in range(1 << algebra.shape[0]):
        entries = np.zeros(algebra.shape[1], dtype=np.uint8)
        for bit in range(algebra.shape[0]):
            if (mask >> bit) & 1:
                entries ^= algebra[bit]
        symplectic = True
        for start, width in layout:
            size = 2 * width
            block = entries[start : start + size * size].reshape(size, size)
            if not np.array_equal((block @ forms[width] @ block.T) & 1, forms[width]):
                symplectic = False
                break
        unitary = np.array_equal(
            multiply(symplectic_involution(entries, layout), entries), one
        )
        assert symplectic == unitary
        checked += 1
    assert checked == 1 << algebra.shape[0]


# ---------------------------------------------------------------------------
# 2. the naive algebra is not sigma-stable; the refined one is
# ---------------------------------------------------------------------------


def test_naive_algebra_is_not_sigma_stable_on_multi_qubit_cells() -> None:
    """If this ever passes with zero escapes, the refinement lost its reason."""

    escapes_seen = 0
    for name in ("c4-22", "c6-22"):
        code = _stabilizer(name)
        naive, layout = partition_algebra(code, _pairs(code.n), refine=False)
        escapes_seen += _sigma_escapes(naive, layout)
    assert escapes_seen > 0


@pytest.mark.parametrize(
    "name", ["steane", "c4-22", "c6-22", "toric-4", "iceberg-8", "cube-832", "surface-5"]
)
def test_refined_algebra_is_sigma_stable(name: str) -> None:
    code = _stabilizer(name)
    for cells in (_singletons(code.n), _pairs(code.n)):
        refined, layout = partition_algebra(code, cells)
        assert _sigma_escapes(refined, layout) == 0, (name, len(cells))


def test_refined_algebra_is_sigma_stable_on_random_codes() -> None:
    rng = np.random.default_rng(515)
    checked = 0
    for n in (2, 3, 4, 5, 6):
        for _ in range(4):
            rows = np.zeros((0, 2 * n), dtype=np.uint8)
            for _ in range(40):
                candidate = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
                if not candidate.any():
                    continue
                if rows.shape[0] and symplectic_product(
                    rows, candidate[None, :], qubits=n
                ).any():
                    continue
                stacked = np.vstack([rows, candidate])
                if rank(stacked) > rows.shape[0]:
                    rows = stacked
            if rows.shape[0] == 0:
                continue
            code = StabilizerCode(rows)
            for cells in (_singletons(n), _pairs(n)):
                refined, layout = partition_algebra(code, cells)
                assert _sigma_escapes(refined, layout) == 0
            checked += 1
    assert checked >= 15


def test_five_qubit_code_refined_algebra_is_sigma_stable() -> None:
    code = five_qubit_code()
    refined, layout = partition_algebra(code, _singletons(5))
    assert _sigma_escapes(refined, layout) == 0


# ---------------------------------------------------------------------------
# 3. the refinement changes no gate group -- the correctness claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cells_kind",
    [
        ("steane", "singletons"),
        ("c4-22", "singletons"),
        ("c4-22", "pairs"),
        ("c6-22", "singletons"),
        ("toric-4", "singletons"),
        ("toric-4", "pairs"),
        ("cube-832", "singletons"),
        ("surface-5", "singletons"),
    ],
)
def test_refinement_preserves_the_symplectic_unit_set(name, cells_kind) -> None:
    """Same gate set, from a strictly smaller algebra.

    Restricted to instances where the *naive* algebra is small enough to sweep
    -- that is exactly the regime in which the two can be compared at all.
    """

    code = _stabilizer(name)
    cells = _singletons(code.n) if cells_kind == "singletons" else _pairs(code.n)
    naive, layout = partition_algebra(code, cells, refine=False)
    refined, _ = partition_algebra(code, cells)
    if naive.shape[0] > 20:
        pytest.skip("naive algebra too large to sweep for comparison")
    assert refined.shape[0] <= naive.shape[0]
    assert _symplectic_units(naive, layout) == _symplectic_units(refined, layout)


@pytest.mark.parametrize(
    "name,expected_order",
    [
        ("steane", 48),
        ("c4-22", 384),
        ("c6-22", 3072),
        pytest.param("cube-832", 6144, marks=pytest.mark.slow),
        pytest.param("iceberg-8", 24576, marks=pytest.mark.slow),
    ],
)
def test_pair_partitions_now_certify(name: str, expected_order: int) -> None:
    """Instances the pre-refinement algebra could not reach.

    ``c6-22`` (dim 28), ``cube-832`` (27) and ``iceberg-8`` (36) all exceeded
    the enumeration cap of 24 before the sigma-refinement and reported
    ``unknown``; they now come out exactly.  The numbers are the contract.
    """

    code = _stabilizer(name)
    analysis = PartitionCliffordAnalysis(code, _pairs(code.n))
    assert analysis.status == "exact"
    assert analysis.group_order == expected_order
    assert analysis.certified
