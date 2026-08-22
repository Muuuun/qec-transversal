"""General (non-CSS) strict-transversal Clifford solver tests."""

import itertools

import numpy as np

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.algebra.preservation import _SL22, _block_action_matrix
from qec_transversal.ansatz.strict import analyze_local_clifford
from qec_transversal.codes.stabilizer import StabilizerCode, five_qubit_code
from qec_transversal.utils.gf2 import rank, reduce_rows, rref
from qec_transversal.utils.symplectic import symplectic_product


def _brute_force_order(code: StabilizerCode) -> int:
    reduced, pivots = rref(code.h)
    count = 0
    for combo in itertools.product(range(6), repeat=code.n):
        entries = np.zeros(4 * code.n, dtype=np.uint8)
        for i, g in enumerate(combo):
            entries[4 * i : 4 * i + 4] = _SL22[g]
        matrix = _block_action_matrix(entries, code.n)
        image = (code.h.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(np.uint8)
        if not reduce_rows(image, reduced, pivots).any():
            count += 1
    return count


def _random_stabilizer(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    rows = np.zeros((0, 2 * n), dtype=np.uint8)
    tries = 0
    while rows.shape[0] < m and tries < 200:
        tries += 1
        candidate = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
        if not candidate.any():
            continue
        if rows.shape[0] and symplectic_product(rows, candidate[None, :], qubits=n).any():
            continue
        stacked = np.vstack([rows, candidate])
        if rank(stacked) > rows.shape[0]:
            rows = stacked
    return rows


def test_five_qubit_code_has_the_cyclic_sh_gate() -> None:
    code = five_qubit_code()
    assert (code.n, code.k) == (5, 1)
    analysis = analyze_local_clifford(code)
    report = analysis.to_dict()
    assert report["physical_group_order"] == 3  # C_3 = <(SH)^tensor5> mod Pauli
    assert report["logical_group"]["order"] == 3
    assert report["enumeration_complete"] and report["certified"]
    names = {g.gate_names for g in analysis.generators}
    assert ("SH",) * 5 in names and ("HS",) * 5 in names


def test_solver_matches_brute_force_on_random_general_codes() -> None:
    rng = np.random.default_rng(515)
    checked = 0
    for n in range(2, 5):
        for _ in range(5):
            h = _random_stabilizer(rng, n, int(rng.integers(1, n + 1)))
            if h.shape[0] == 0:
                continue
            code = StabilizerCode(h)
            analysis = analyze_local_clifford(code)
            assert analysis.group_order == _brute_force_order(code)
            assert analysis.certified
            checked += 1
    assert checked >= 10


def test_general_solver_agrees_with_css_shear_theorem() -> None:
    # Albert's theorem: for CSS codes the diagonal shear families generate
    # the full strict-transversal Clifford group.  The general SL(2,2)^n
    # solver knows nothing about shears, so agreement is an independent
    # computational proof on each code.
    for name in ["steane", "c4-22", "qrm15", "surface-5"]:
        css = CSSCode(*REGISTRY[name].build())
        h = np.vstack(
            [
                np.hstack([css.c_x, np.zeros_like(css.c_x)]),
                np.hstack([np.zeros_like(css.c_z), css.c_z]),
            ]
        )
        general = analyze_local_clifford(StabilizerCode(h)).to_dict()
        strict = css.analyze_transversal().to_dict()
        assert general["logical_group"]["order"] == strict["logical_group"]["order"]


def test_noncommuting_rows_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="commute"):
        StabilizerCode(np.array([[1, 0], [0, 1]], dtype=np.uint8))  # X and Z on one qubit


def test_partition_solver_matches_brute_force_on_a_pair() -> None:
    from qec_transversal.algebra.preservation import _local_symplectic_form
    from qec_transversal.ansatz.partition import analyze_partition_clifford

    form = _local_symplectic_form(2)
    sp42 = []
    for bits in range(1 << 16):
        m = np.array(
            [[(bits >> (4 * r + c)) & 1 for c in range(4)] for r in range(4)],
            dtype=np.uint8,
        )
        if np.array_equal((m @ form @ m.T) & 1, form):
            sp42.append(m)
    assert len(sp42) == 720  # |Sp(4,2)|

    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(6):
        h = _random_stabilizer(rng, 2, int(rng.integers(1, 3)))
        if h.shape[0] == 0:
            continue
        code = StabilizerCode(h)
        analysis = analyze_partition_clifford(code, [(0, 1)])
        reduced, pivots = rref(code.h)
        want = 0
        for m in sp42:
            matrix = np.zeros((4, 4), dtype=np.uint8)
            coords = np.array([0, 1, 2, 3])
            matrix[np.ix_(coords, coords)] = m
            image = (code.h.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(
                np.uint8
            )
            if not reduce_rows(image, reduced, pivots).any():
                want += 1
        assert analysis.group_order == want and analysis.certified
        checked += 1
    assert checked >= 3


def test_partition_singletons_reproduce_strict_group() -> None:
    from qec_transversal.ansatz.partition import analyze_partition_clifford

    css = CSSCode(*REGISTRY["steane"].build())
    h = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    code = StabilizerCode(h)
    part = analyze_partition_clifford(code, [(i,) for i in range(7)])
    assert part.group_order == analyze_local_clifford(code).group_order == 6


def test_two_local_pairing_expands_c422_gate_group() -> None:
    from qec_transversal.ansatz.partition import analyze_partition_clifford

    css = CSSCode(*REGISTRY["c4-22"].build())
    h = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    report = analyze_partition_clifford(StabilizerCode(h), [(0, 1), (2, 3)]).to_dict()
    assert report["logical_group"]["order"] == 48  # vs 6 for strict layers
    assert report["certified"]


def _random_stabilizer_import_guard():
    return analyze_local_clifford
