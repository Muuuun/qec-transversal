"""Blind structural discovery, validated against an external certified census.

The fixture ``tests/data/albert_census_extract.json`` carries five codes from
the arXiv:2608.05688 companion repository (valbert4/two-fold-transversal)
with the paper's certified depth-one two-local logical group orders — codes
this repository has NO registry entry for, so every assertion here exercises
the blind path.  Scientific outcomes asserted (each certified end to end):

* [[54,4,6]] and [[60,4,8]] Kasai codes -> blind one-block order 60, exactly
  the census value (these were order 1 before discovery existed);
* [[56,18,4]] Kasai -> blind exact order 1,524,096 at k = 18, matching the
  census (exercises the raised Schreier-Sims k-limit as well) — slow;
* [[54,8,4]] Kasai -> discovery reproduces every one of the census's four
  CZ-matching involutions natively;
* [[32,2,4]] (the toric-4 code under another flag) -> blind order 48, the
  census value, where the registry path previously certified only 12.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from qec_transversal import CSSCode
from qec_transversal.codes import REGISTRY
from qec_transversal.discovery import (
    certified_shift_structure,
    discover_involutions,
    structural_permutations,
)
from qec_transversal.matching import analyze_matching, logical_group_summary
from qec_transversal.oneblock import analyze_one_block

FIXTURE = Path(__file__).parent / "data" / "albert_census_extract.json"


def _census():
    return json.loads(FIXTURE.read_text())["codes"]


def _census_code(label: str) -> tuple[dict, CSSCode]:
    entry = next(c for c in _census() if c["label"] == label)
    h_x = np.array([[int(x) for x in row] for row in entry["H_X"]], dtype=np.uint8)
    h_z = np.array([[int(x) for x in row] for row in entry["H_Z"]], dtype=np.uint8)
    return entry, CSSCode(h_x, h_z)


# -- structure inference ----------------------------------------------------


def test_shift_structure_certifies_only_true_symmetries() -> None:
    _, code = _census_code("[[54,8,4]]kasai:ec88c3e3")
    from qec_transversal.gf2 import rowspace_residues

    structure = certified_shift_structure(code)
    assert 9 in structure and 1 in structure[9]  # the Kasai lift
    # every reported shift must actually be a rowspace symmetry
    for block, strides in structure.items():
        for stride in strides:
            index = np.arange(code.n)
            tau = (index // block) * block + ((index % block) + stride) % block
            assert not rowspace_residues(code.c_x[:, tau], code.c_x).any()
            assert not rowspace_residues(code.c_z[:, tau], code.c_z).any()


def test_discovered_involutions_are_involutions_and_certify() -> None:
    _, code = _census_code("[[54,4,6]]kasai:e4d756a1")
    found = discover_involutions(code)
    assert found, "discovery must find the Kasai negation matchings"
    identity = np.arange(code.n)
    for label, tau in found:
        assert np.array_equal(tau[tau], identity), label
        analysis = analyze_matching(code, tau)
        if label.startswith("discovered-duality"):
            assert analysis.is_zx_duality, label


def test_structural_permutations_certify_downstream() -> None:
    from qec_transversal.automorphisms import describe_permutation

    _, code = _census_code("[[60,4,8]]kasai:98c88188")
    perms = structural_permutations(code)
    assert perms, "certified shift structure must yield permutation gates"
    for perm in perms:
        record = describe_permutation(code, perm)
        assert all(record.certificate.values())


def test_discovery_reproduces_the_census_kasai54_matchings() -> None:
    entry, code = _census_code("[[54,8,4]]kasai:ec88c3e3")
    found_keys = {tau.tobytes() for _, tau in discover_involutions(code)}
    for pairs in entry["cz_matchings"]:
        tau = np.arange(code.n)
        for a, b in pairs:
            tau[a], tau[b] = tau[b], tau[a]
        assert tau.tobytes() in found_keys, "census matching missed by discovery"


# -- blind census agreement -------------------------------------------------


@pytest.mark.parametrize(
    "label", ["[[54,4,6]]kasai:e4d756a1", "[[60,4,8]]kasai:98c88188"]
)
def test_blind_one_block_matches_census_small_kasai(label: str) -> None:
    entry, code = _census_code(label)
    analysis = analyze_one_block(code, name=None, time_budget_s=60.0, involution_cap=48)
    assert analysis.logical_order_exact
    assert analysis.logical_order == int(entry["depth1_image"])


def test_blind_one_block_matches_census_toric_flag() -> None:
    entry, code = _census_code("[[32,2,4]]mm:f63d251a")
    analysis = analyze_one_block(code, name=None, time_budget_s=60.0, involution_cap=48)
    assert analysis.logical_order_exact
    assert analysis.logical_order == int(entry["depth1_image"]) == 48


@pytest.mark.slow
def test_blind_one_block_matches_census_kasai56_at_k18() -> None:
    entry, code = _census_code("[[56,18,4]]kasai:b824a37f")
    analysis = analyze_one_block(code, name=None, time_budget_s=180.0, involution_cap=48)
    assert analysis.logical_order_exact
    assert analysis.logical_order == int(entry["depth1_image"]) == 1524096


# -- registry improvements the cross-check produced -------------------------


def test_toric_4_registry_reaches_the_census_order() -> None:
    analysis = analyze_one_block(
        CSSCode(*REGISTRY["toric-4"].build()), name="toric-4", involution_cap=48
    )
    assert analysis.logical_order_exact
    assert analysis.logical_order == 48


@pytest.mark.slow
def test_gross_registry_exceeds_both_censuses() -> None:
    # 707,788,800 = the certified union of this repo's previous 11,059,200
    # and the census's 460,800 — discovery now reaches it natively.
    analysis = analyze_one_block(
        CSSCode(*REGISTRY["gross"].build()), name="gross", time_budget_s=60.0, involution_cap=48
    )
    assert analysis.logical_order_exact
    assert analysis.logical_order == 707788800


# -- big-k exact orders (matching.logical_group_summary) --------------------


@pytest.mark.slow
def test_group_summary_certifies_census_group_at_k18() -> None:
    entry, code = _census_code("[[56,18,4]]kasai:b824a37f")
    generators = []
    for pairs in entry["cz_matchings"]:
        tau = np.arange(code.n)
        for a, b in pairs:
            tau[a], tau[b] = tau[b], tau[a]
        analysis = analyze_matching(code, tau)
        for generator in analysis.generators:
            if not generator.is_logical_identity and all(generator.certificate.values()):
                generators.append(generator.logical_symplectic)
        fold_h = analysis.fold_hadamard
        if fold_h is not None and not fold_h.is_logical_identity and all(
            fold_h.certificate.values()
        ):
            generators.append(fold_h.logical_symplectic)
    summary = logical_group_summary(generators, code.k)
    assert summary["exact"] and summary["order"] == 1524096
