import numpy as np
import pytest

from qec_transversal.codes import doubled_color_41, steane_code, surface_code, toric_code
from qec_transversal.concatenation import (
    concatenate,
    inner_block_cells,
    lift_pauli,
    shor_code,
)
from qec_transversal.css import CSSCode
from qec_transversal.faulttolerance import (
    block_diagonal,
    encoded_lift,
    logical_action,
    matching_cells,
    partition_distance,
    preserves_code,
    singleton_cells,
    spread,
)
from qec_transversal.gf2 import is_symplectic
from qec_transversal.stabilizer import StabilizerCode, five_qubit_code

H_LOGICAL = np.array([[0, 1], [1, 0]], dtype=np.uint8)
S_LOGICAL = np.array([[1, 1], [0, 1]], dtype=np.uint8)


def _stacked(css: CSSCode) -> StabilizerCode:
    return StabilizerCode(
        np.vstack(
            [
                np.hstack([css.c_x, np.zeros_like(css.c_x)]),
                np.hstack([np.zeros_like(css.c_z), css.c_z]),
            ]
        )
    )


def _weight(vector: np.ndarray, n: int) -> int:
    return int((vector[:n] | vector[n:]).sum())


def _brute_force_distance(code: StabilizerCode) -> int:
    """Minimum weight of a nontrivial logical, by exhaustion over N(S)/S."""

    n, k = code.n, code.k
    basis = np.vstack([code.h, code.logical])
    best = n + 1
    for mask in range(1, 1 << basis.shape[0]):
        bits = [(mask >> i) & 1 for i in range(basis.shape[0])]
        if not any(bits[code.rank :]):
            continue  # a stabilizer element, not a logical
        vector = np.zeros(2 * n, dtype=np.uint8)
        for bit, row in zip(bits, basis):
            if bit:
                vector ^= row
        best = min(best, _weight(vector, n))
    assert k  # only meaningful for k >= 1
    return best


def test_singleton_partition_distance_is_the_code_distance() -> None:
    for code in (_stacked(CSSCode(*steane_code())), five_qubit_code()):
        result = partition_distance(code, singleton_cells(code.n), max_blocks=3)
        assert result.value == _brute_force_distance(code) == 3
        assert result.single_fault_correctable is True
        assert result.correctable_faults == 1


def test_detection_only_code_is_not_single_fault_correctable() -> None:
    h = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
    code = StabilizerCode(h)
    strict = partition_distance(code, singleton_cells(4))
    assert strict.value == 2 and strict.single_fault_correctable is False
    # a logical XX sits inside a single cell of the paired partition
    paired = partition_distance(code, [(0, 1), (2, 3)])
    assert paired.value == 1 and paired.witness_cells == (0,)


def test_witness_is_a_genuine_logical_of_the_claimed_block_weight() -> None:
    code = _stacked(CSSCode(*toric_code(4)))
    cells = matching_cells([q ^ 1 for q in range(code.n)])
    result = partition_distance(code, cells, max_blocks=2)
    # d = 4 and cells of size 2, so 2l = d: the sufficient condition 2l < d just
    # fails, and here the layer really is not single-fault correctable.
    assert result.value == 2 and result.single_fault_correctable is False
    touched = {
        index
        for index, cell in enumerate(cells)
        for q in cell
        if result.witness[q] or result.witness[code.n + q]
    }
    assert touched == set(result.witness_cells)
    assert len(touched) == result.value
    # in N(S): commutes with every check; outside S: moves some logical
    swapped = np.hstack([code.h[:, code.n :], code.h[:, : code.n]])
    assert not (result.witness @ swapped.T % 2).any()
    pairing = np.hstack([code.logical[:, code.n :], code.logical[:, : code.n]])
    assert (result.witness @ pairing.T % 2).any()


def test_certificate_is_two_sided_at_the_searched_depth() -> None:
    code = _stacked(CSSCode(*surface_code(5)))
    result = partition_distance(code, singleton_cells(code.n), max_blocks=2)
    assert result.value is None  # no logical of weight <= 2 exists
    assert result.lower_bound == 3 and result.single_fault_correctable is True


def test_spread_is_not_the_cell_size() -> None:
    n = 5
    permutation = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i, image in enumerate([1, 2, 3, 4, 0]):
        permutation[i, image] = 1
        permutation[n + i, n + image] = 1
    assert is_symplectic(permutation, qubits=n)
    assert spread(permutation, qubits=n) == 1  # any cell size, spread one


def test_encoded_lift_realizes_any_logical_clifford() -> None:
    inner = shor_code()
    for target in (H_LOGICAL, S_LOGICAL):
        lift = encoded_lift(inner, target)
        assert preserves_code(inner, lift)
        assert np.array_equal(logical_action(inner, lift), target)


def test_large_cells_are_compatible_with_fault_tolerance() -> None:
    """Steane o Shor = [[63,1,9]]: cell size 9 >= d = 9, yet d_P = 3.

    The whole implication "cell size >= distance, therefore no fault-tolerant
    depth-one implementation" fails here: the layer is depth one on 9-qubit
    cells, one entire faulty cell is correctable, and it implements the full
    single-qubit logical Clifford group.
    """

    inner = shor_code()
    outer = _stacked(CSSCode(*steane_code()))
    code = concatenate(outer, inner)
    assert (code.n, code.k) == (63, 1)

    cells = inner_block_cells(outer.n, inner.n)
    assert max(len(cell) for cell in cells) == 9

    # d = 9: >= d_outer * d_inner by concatenation, and the lifted outer Xbar
    # attains it.
    assert _weight(lift_pauli(outer.logical[0], outer.n, inner.logical[0], inner.logical[1]), code.n) == 9

    result = partition_distance(code, cells, max_blocks=3)
    assert result.value == 3  # cell size 9 >= d = 9, and still correctable
    assert result.single_fault_correctable is True
    assert result.correctable_faults == 1

    for target in (H_LOGICAL, S_LOGICAL):
        block = encoded_lift(inner, target)
        assert spread(block, qubits=inner.n) <= inner.n
        layer = block_diagonal([block] * outer.n, cells, code.n)
        assert is_symplectic(layer, qubits=code.n)
        assert preserves_code(code, layer)
        assert np.array_equal(logical_action(code, layer), target)


def test_full_logical_clifford_at_cell_size_one_and_distance_nine() -> None:
    """The converse reading also fails: [[41,1,9]] is full at l = 1."""

    css = CSSCode(*doubled_color_41())
    analysis = css.analyze_transversal().to_dict()
    assert analysis["logical_group"]["is_full_logical_clifford"] is True
    code = _stacked(css)
    result = partition_distance(code, singleton_cells(code.n), max_blocks=2)
    assert result.single_fault_correctable is True


def test_cells_must_partition_the_qubits() -> None:
    code = five_qubit_code()
    with pytest.raises(ValueError):
        partition_distance(code, [(0, 1), (1, 2), (3,), (4,)])
