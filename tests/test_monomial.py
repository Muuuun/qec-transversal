"""Monomial (permutation x local-Clifford) automorphism group tests."""

import numpy as np
import pytest

pytest.importorskip("igraph")

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.ansatz import monomial as monomial_module
from qec_transversal.ansatz.monomial import analyze_monomial, strict_cross_check
from qec_transversal.codes.stabilizer import StabilizerCode, five_qubit_code


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
    assert report["permutation_image_order"] == 168
    assert report["row_set_complete"] and report["certified"]
    assert report["logical_group"]["order"] == 6


def test_five_qubit_code_monomial_group_gives_full_logical_clifford() -> None:
    report = analyze_monomial(five_qubit_code()).to_dict()
    assert report["monomial_group_order"] == 360
    assert report["logical_group"]["order"] == 6  # full Clifford mod Pauli, k=1
    assert report["certified"]


def test_monomial_group_contains_strict_group() -> None:
    # full-scope identity: the kernel of the projection onto S_n is exactly
    # the strict-transversal group, so |strict| = |monomial| / |image in S_n|
    names = [
        "steane",
        "c4-22",
        "c6-22",
        "cube-832",
        "iceberg-8",
        "iceberg-12",
        "tesseract",
    ]
    codes = [(name, _stacked(CSSCode(*REGISTRY[name].build()))) for name in names]
    codes.append(("five-qubit", five_qubit_code()))
    for name, code in codes:
        report = strict_cross_check(code)
        assert report["applicable"], name
        assert report["mode"] == "equality", name
        assert report["consistent"], name
        assert report["kernel_order"] == report["strict_order"], name
        assert report["monomial_order"] % report["strict_order"] == 0, name


def test_generator_scope_kernel_is_only_a_lower_bound(monkeypatch) -> None:
    # with the row set restricted to the RREF generator basis the kernel of
    # the permutation projection is merely a subgroup of the strict group
    # (row-SET preservation is stronger than stabilizer-GROUP preservation);
    # on steane the bound is strict: kernel 2 < strict 6
    monkeypatch.setattr(monomial_module, "_FULL_GROUP_RANK", 0)
    code = _stacked(CSSCode(*REGISTRY["steane"].build()))
    report = strict_cross_check(code)
    assert report["applicable"]
    assert report["mode"] == "lower_bound_only"
    assert report["consistent"]
    assert report["kernel_order"] <= report["strict_order"]
    assert report["kernel_order"] == 2
    assert report["strict_order"] == 6
