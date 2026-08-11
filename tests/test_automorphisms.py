"""Tanner-graph automorphism and duality-discovery tests."""

import numpy as np
import pytest

igraph = pytest.importorskip("igraph")

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.automorphisms import (
    analyze_automorphisms,
    permutation_group_order,
)
from qec_transversal.matching import analyze_matching


def test_permutation_group_order_known_groups() -> None:
    # S_4 from a transposition and a 4-cycle
    s4 = [np.array([1, 0, 2, 3]), np.array([1, 2, 3, 0])]
    assert permutation_group_order(s4) == 24
    # cyclic group of order 6
    c6 = [np.array([1, 2, 3, 4, 5, 0])]
    assert permutation_group_order(c6) == 6
    assert permutation_group_order([]) == 1
    assert permutation_group_order([np.arange(5)]) == 1


def test_c422_automorphisms_are_full_symmetric_group() -> None:
    code = CSSCode(*REGISTRY["c4-22"].build())
    analysis = analyze_automorphisms(code)
    report = analysis.to_dict()
    assert report["qubit_group_order"] == 24  # S_4 on the iceberg qubits
    assert report["duality_exists"] is True
    assert report["certified"]


def test_generators_preserve_row_spaces() -> None:
    for name in ["steane", "toric-4", "bb72", "gb48"]:
        analysis = analyze_automorphisms(CSSCode(*REGISTRY[name].build()))
        for generator in analysis.generators:
            assert generator.certificate["preserves_C_X"]
            assert generator.certificate["preserves_C_Z"]


def test_toric_translations_present_and_duality_found() -> None:
    code = CSSCode(*REGISTRY["toric-4"].build())
    report = analyze_automorphisms(code).to_dict()
    assert report["qubit_group_order"] % 16 == 0  # contains the d^2 = 16 translations
    assert report["duality_exists"] is True


def test_cube_code_has_no_duality() -> None:
    # [[8,3,2]] has 1 X-check and 4 Z-checks: no duality can exist.
    report = analyze_automorphisms(CSSCode(*REGISTRY["cube-832"].build())).to_dict()
    assert report["duality_exists"] is False


def test_kasai_duality_discovered_and_fold_certifies() -> None:
    code = CSSCode(*REGISTRY["kasai-binary-294"].build())
    analysis = analyze_automorphisms(code)
    assert analysis.duality_certified
    tau = analysis.involutive_duality()
    assert tau is not None
    fold = analyze_matching(code, tau)
    report = fold.to_dict()
    assert report["is_zx_duality"]
    assert report["nontrivial_generator_count"] >= 1
    assert report["fold_hadamard_nontrivial"]
    assert report["certified"]


def test_disconnected_tanner_graph_leaves_duality_undecided() -> None:
    # two disjoint [[4,2,2]] blocks: a duality exists (identity), but the
    # union-orbit witness only maps one component, so the verdict must be
    # None (undecided), never a false "nonexistent".
    block = np.zeros((2, 8), dtype=np.uint8)
    block[0, :4] = 1
    block[1, 4:] = 1
    code = CSSCode(block, block)
    report = analyze_automorphisms(code).to_dict()
    assert report["tanner_connected"] is False
    assert report["duality_exists"] is None
    assert report["duality_decided"] is False
