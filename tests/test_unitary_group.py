"""The phi-index symplectic cut: an orbit instead of a sweep of ``A^x``.

The claim under test is ``|G| = |A^x| / |orbit of 1|`` for the congruence
action ``a . u = sigma(u) a u``, with generators of ``G`` from Schreier's
lemma.  Where the old sweep also runs, the two must agree exactly; where it
cannot, the orbit route must still be sound (every generator verified
symplectic and code-preserving) and its order must divide ``|A^x|``.
"""

import itertools

import numpy as np
import pytest

from qec_transversal.algebra import AlgebraF2, unit_group
from qec_transversal.algebra.preservation import (
    _local_symplectic_form,
    _partition_multiply,
    partition_algebra,
)
from qec_transversal.algebra.unitary_group import unitary_group
from qec_transversal.ansatz.partition import (
    _involution_matrix,
    partition_units_via_structure,
)
from qec_transversal.api import (
    Completeness,
    _as_stabilizer_code,
    partition_clifford_group,
)
from qec_transversal.codes.css import CSSCode
from qec_transversal.codes.registry import REGISTRY


def _code(name: str):
    return _as_stabilizer_code(CSSCode(*REGISTRY[name].build()))


def _partitions(n: int, width: int):
    """Every partition of ``range(n)`` into cells of size at most ``width``."""

    def rec(rest, acc):
        if not rest:
            yield [tuple(cell) for cell in acc]
            return
        first, tail = rest[0], rest[1:]
        for size in range(1, min(width, len(rest)) + 1):
            for extra in itertools.combinations(tail, size - 1):
                cell = (first,) + extra
                remaining = [q for q in tail if q not in extra]
                yield from rec(remaining, acc + [cell])

    yield from rec(list(range(n)), [])


def _algebra_of(code, cells):
    basis, layout = partition_algebra(code, cells)
    width = basis.shape[1]
    one = np.zeros(width, dtype=np.uint8)
    for start, cell_width in layout:
        size = 2 * cell_width
        one[start : start + size * size] = np.eye(size, dtype=np.uint8).reshape(-1)
    algebra = AlgebraF2(basis, _partition_multiply(cells, layout, width), one)
    return algebra, layout


# ---------------------------------------------------------------------------
# agreement with the route it replaces
# ---------------------------------------------------------------------------


def test_phi_route_matches_the_sweep_on_every_small_partition() -> None:
    """Both routes are exact here, so they must agree element for element."""

    checked = 0
    for name, width in (("c4-22", 2), ("cube-832", 1), ("steane", 2)):
        code = _code(name)
        for cells in _partitions(code.n, width):
            sweep = partition_units_via_structure(code, cells, method="enumeration")
            if sweep["status"] != "exact":
                continue
            orbit = partition_units_via_structure(code, cells, method="phi")
            assert orbit["status"] == "exact", (name, cells, orbit["detail"])
            assert orbit["symplectic_group_order"] == sweep["symplectic_group_order"]
            assert orbit["logical_group"].get("order") == sweep["logical_group"].get(
                "order"
            )
            checked += 1
    assert checked > 100


def test_api_methods_agree_on_a_pair_partition() -> None:
    code = CSSCode(*REGISTRY["c6-22"].build())
    cells = [(0, 1), (2, 3), (4, 5)]
    results = {
        method: partition_clifford_group(code, cells, method=method)
        for method in ("auto", "enumeration", "structure", "phi")
    }
    for method, result in results.items():
        assert result.completeness is Completeness.COMPLETE, method
    orders = {r.group_order for r in results.values()}
    logical = {r.logical_group_order for r in results.values()}
    assert len(orders) == 1 and len(logical) == 1


def test_unknown_method_is_rejected() -> None:
    code = CSSCode(*REGISTRY["c4-22"].build())
    with pytest.raises(ValueError):
        partition_clifford_group(code, [(0,), (1,), (2,), (3,)], method="nope")
    with pytest.raises(ValueError):
        partition_units_via_structure(
            _code("c4-22"), [(0,), (1,), (2,), (3,)], method="nope"
        )


# ---------------------------------------------------------------------------
# what the sweep could not do at all
# ---------------------------------------------------------------------------


def test_three_qubit_cells_are_now_decidable() -> None:
    """``|A^x| = 14155776`` on this partition: far past any affordable sweep."""

    code = _code("c6-22")
    cells = [(0, 1, 2), (3, 4, 5)]
    swept = partition_units_via_structure(
        code, cells, method="enumeration", group_enumeration_cap=2_000_000
    )
    assert swept["status"] == "unknown"  # the wall this module removes

    summary = partition_units_via_structure(code, cells, method="phi")
    assert summary["status"] == "exact"
    assert summary["unit_group_order"] == 14_155_776
    assert summary["orbit"]["orbit_size"] == 384
    assert summary["symplectic_group_order"] == 36_864
    assert 14_155_776 == 36_864 * 384
    assert summary["logical_group"]["order"] == 48
    assert summary["logical_group"]["exact"] is True
    assert summary["orbit"]["generators_complete"] is True


def test_auto_reaches_for_the_index_when_the_sweep_is_too_big() -> None:
    code = CSSCode(*REGISTRY["c6-22"].build())
    result = partition_clifford_group(code, [(0, 1, 2), (3, 4, 5)])
    assert result.completeness is Completeness.COMPLETE
    assert result.group_order == 36_864
    assert result.metadata["generator_records"].startswith("certified generating set")


# ---------------------------------------------------------------------------
# soundness of what comes back
# ---------------------------------------------------------------------------


def test_every_generator_is_symplectic_and_preserves_the_code() -> None:
    code = _code("c6-22")
    for cells in ([(0, 1, 2), (3, 4, 5)], [(0, 1, 2, 3), (4,), (5,)]):
        summary = partition_units_via_structure(code, cells, method="phi")
        assert summary["status"] == "exact"
        analysis = summary["analysis"]
        assert analysis.generators
        for generator in analysis.generators:
            assert all(generator.certificate.values())
            matrix = generator.matrix
            for (start, cell_width), cell in zip(analysis._layout, analysis.cells):
                index = np.asarray(cell, dtype=int)
                coords = np.concatenate([index, code.n + index])
                block = matrix[np.ix_(coords, coords)]
                form = _local_symplectic_form(cell_width)
                assert np.array_equal((block @ form @ block.T) % 2, form)


def test_orbit_stabilizer_identity_holds_exactly() -> None:
    code = _code("c6-22")
    cells = [(0, 1, 2, 3), (4,), (5,)]
    algebra, layout = _algebra_of(code, cells)
    units = unit_group(algebra)
    assert units.status == "exact"
    sigma = _involution_matrix(algebra, layout)
    assert sigma is not None
    result = unitary_group(
        algebra,
        sigma,
        unit_order=units.order,
        unit_generators=units.generators,
    )
    assert result.status == "exact"
    assert result.order * result.orbit_size == units.order
    assert result.checks["sigma_square_is_identity"] is True
    assert result.checks["sigma_fixes_one"] is True
    assert result.checks["orbit_stabilizer_divides"] is True


def test_a_wrong_involution_is_refused_rather_than_used() -> None:
    """A linear map that is not an involution of ``A`` must not yield a number."""

    code = _code("c6-22")
    cells = [(0, 1), (2, 3), (4, 5)]
    algebra, layout = _algebra_of(code, cells)
    units = unit_group(algebra)
    bogus = np.eye(algebra.dim, dtype=np.uint8)
    bogus[[0, 1]] = bogus[[1, 0]]  # an involution of coordinates, not of A
    result = unitary_group(
        algebra,
        bogus,
        unit_order=units.order,
        unit_generators=units.generators,
    )
    assert result.status == "unknown"
    assert result.order is None


def test_orbit_cap_declines_instead_of_truncating() -> None:
    code = _code("c6-22")
    cells = [(0, 1, 2, 3), (4,), (5,)]
    algebra, layout = _algebra_of(code, cells)
    units = unit_group(algebra)
    sigma = _involution_matrix(algebra, layout)
    result = unitary_group(
        algebra,
        sigma,
        unit_order=units.order,
        unit_generators=units.generators,
        orbit_cap=64,
    )
    assert result.status == "unknown"
    assert result.order is None
    assert "exceeded the cap" in result.detail


@pytest.mark.slow
def test_five_qubit_cell_certifies_a_full_logical_clifford_group() -> None:
    """[[6,2,2]] with one width-5 cell: 1.17e15 units, decided by a 1.4e6 orbit.

    This is the regime the index formula exists for -- the sweep would need
    ``10^15`` steps -- and it is the upper end of the fixed-partition staircase
    for this code: width 5 reaches all of ``Sp(4,2)``.
    """

    code = _code("c6-22")
    summary = partition_units_via_structure(
        code, [(0, 1, 2, 3, 4), (5,)], method="phi", orbit_cap=4_000_000
    )
    assert summary["status"] == "exact"
    assert summary["unit_group_order"] == 1_168_918_299_279_360
    assert summary["orbit"]["orbit_size"] == 1_376_256
    assert summary["symplectic_group_order"] == 849_346_560
    assert (
        summary["symplectic_group_order"] * summary["orbit"]["orbit_size"]
        == summary["unit_group_order"]
    )
    assert summary["logical_group"]["order"] == 720  # |Sp(4,2)|
