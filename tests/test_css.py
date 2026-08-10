import numpy as np
import pytest

from qec_transversal import CSSCode
from qec_transversal.gf2 import is_in_rowspace, nullspace
from qec_transversal.group import symplectic_group_order


STEANE = np.asarray(
    [
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def test_steane_transversal_generators_are_complete_for_one_logical_qubit() -> None:
    code = CSSCode(STEANE, STEANE)
    analysis = code.analyze_transversal()
    report = analysis.to_dict(group_cap=100)

    assert (code.n, code.k) == (7, 1)
    assert analysis.a_z.dimension == 1
    assert analysis.a_x.dimension == 1
    assert analysis.a_z.basis[0].tolist() == [1] * 7
    assert analysis.a_x.basis[0].tolist() == [1] * 7
    assert analysis.certified
    assert report["logical_group"]["exact"]
    assert report["logical_group"]["order"] == symplectic_group_order(1) == 6
    assert report["logical_group"]["is_full_logical_clifford"] is True

    capped_report = analysis.to_dict(group_cap=5)
    assert capped_report["logical_group"]["exact"] is True
    assert capped_report["logical_group"]["order"] == 6
    assert capped_report["logical_group"]["method"] == "symplectic ambient-order bound"


def test_dependent_checks_do_not_change_code_dimension() -> None:
    duplicated = np.vstack([STEANE, STEANE[0]])
    code = CSSCode(duplicated, STEANE)
    assert code.rank_x == 3
    assert code.k == 1
    assert code.analyze_transversal().certified


def test_noncommuting_css_checks_are_rejected() -> None:
    with pytest.raises(ValueError, match="H_X H_Z"):
        CSSCode([[1, 0]], [[1, 0]])


def test_zero_logical_qubits() -> None:
    code = CSSCode([[1, 0]], [[0, 1]])
    report = code.analyze_transversal().to_dict(group_cap=10)
    assert code.k == 0
    assert report["logical_group"]["order"] == 1
    assert report["certificate"]["certified"]


def _span(basis: np.ndarray) -> set[bytes]:
    rows = basis.shape[0]
    result: set[bytes] = set()
    for mask in range(1 << rows):
        vector = np.zeros(basis.shape[1], dtype=np.uint8)
        for row in range(rows):
            if mask & (1 << row):
                vector ^= basis[row]
        result.add(vector.tobytes())
    return result


def _brute_parameters(source: np.ndarray, target: np.ndarray) -> set[bytes]:
    result: set[bytes] = set()
    n = source.shape[1]
    for mask in range(1 << n):
        parameter = np.asarray([(mask >> qubit) & 1 for qubit in range(n)], dtype=np.uint8)
        if all(is_in_rowspace(parameter * check, target) for check in source):
            result.add(parameter.tobytes())
    return result


def test_parameter_nullspaces_match_exhaustive_search_on_small_random_codes() -> None:
    rng = np.random.default_rng(260805688)
    for n in range(2, 7):
        for _ in range(6):
            h_x = rng.integers(0, 2, size=(rng.integers(0, n), n), dtype=np.uint8)
            x_perp = nullspace(h_x)
            z_rows = rng.integers(0, 2, size=(rng.integers(0, n), x_perp.shape[0]), dtype=np.uint8)
            h_z = (z_rows @ x_perp) & 1
            code = CSSCode(h_x, h_z, n=n)
            analysis = code.analyze_transversal()

            assert _span(analysis.a_z.basis) == _brute_parameters(h_x, h_z)
            assert _span(analysis.a_x.basis) == _brute_parameters(h_z, h_x)
            assert analysis.certified
