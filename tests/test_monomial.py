"""Monomial (permutation x local-Clifford) automorphism group tests."""

import numpy as np
import pytest

pytest.importorskip("igraph")

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.monomial import analyze_monomial
from qec_transversal.stabilizer import (
    StabilizerCode,
    analyze_local_clifford,
    five_qubit_code,
)


def _stacked(css: CSSCode) -> StabilizerCode:
    return StabilizerCode(
        np.vstack(
            [
                np.hstack([css.c_x, np.zeros_like(css.c_x)]),
                np.hstack([np.zeros_like(css.c_z), css.c_z]),
            ]
        )
    )


def test_steane_monomial_group_recovers_pgl32() -> None:
    code = _stacked(CSSCode(*REGISTRY["steane"].build()))
    report = analyze_monomial(code).to_dict()
    # 1008 = 6 local-Clifford x 168 = |PGL(3,2)| permutations; the pure
    # Tanner-graph route sees only 6 permutations because the RREF check
    # basis breaks the symmetry - the all-elements row set restores it.
    assert report["monomial_group_order"] == 1008
    assert report["row_set_complete"] and report["certified"]
    assert report["logical_group"]["order"] == 6


def test_five_qubit_code_monomial_group_gives_full_logical_clifford() -> None:
    report = analyze_monomial(five_qubit_code()).to_dict()
    assert report["monomial_group_order"] == 360
    assert report["logical_group"]["order"] == 6  # full Clifford mod Pauli, k=1
    assert report["certified"]


def test_monomial_group_contains_strict_group() -> None:
    for name in ["steane", "c4-22", "cube-832"]:
        code = _stacked(CSSCode(*REGISTRY[name].build()))
        monomial = analyze_monomial(code).to_dict()["monomial_group_order"]
        strict = analyze_local_clifford(code).group_order
        assert monomial % strict == 0
