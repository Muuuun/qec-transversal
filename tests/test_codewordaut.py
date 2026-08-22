"""Characteristic-set (row-space) permutation automorphisms.

Scientific outcomes asserted (each certified, never guessed):

* brute-force oracle equality on every n <= 8 registry code — steane 168
  (= PGL(3,2)), c4-22 24, c6-22 48, cube-832 1344 — the engine's exactness
  claim checked against exhaustive S_n search;
* census Kasai [[54,8,4]]: exact row-space permutation group of order 108,
  containing all four permutation generators recorded by the
  arXiv:2608.05688 census — the group the Tanner (row-set) route reports as
  trivial, and the capability other tools obtain only through MAGMA/GAP;
* honest degradation: over-cap ranks raise, truncated class selections
  certify a subgroup and say so.
"""

import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from qec_transversal import CSSCode
from qec_transversal.ansatz.codeword_permutation import analyze_codeword_automorphisms
from qec_transversal.codes import REGISTRY
from qec_transversal.utils.gf2 import rowspace_residues

FIXTURE = Path(__file__).parent / "data" / "albert_census_extract.json"


def _brute_force_order(code: CSSCode) -> int:
    count = 0
    for image in permutations(range(code.n)):
        tau = np.array(image)
        if not rowspace_residues(code.c_x[:, tau], code.c_x).any() and not rowspace_residues(
            code.c_z[:, tau], code.c_z
        ).any():
            count += 1
    return count


@pytest.mark.parametrize(
    "name,expected",
    [("steane", 168), ("c4-22", 24), ("c6-22", 48), ("cube-832", 1344)],
)
def test_matches_brute_force_on_small_codes(name: str, expected: int) -> None:
    code = CSSCode(*REGISTRY[name].build())
    analysis = analyze_codeword_automorphisms(code)
    assert analysis.exact and analysis.certified
    assert analysis.group_order == expected == _brute_force_order(code)


def test_census_kasai54_group_is_exactly_108_and_contains_the_census_gens() -> None:
    entry = next(
        c
        for c in json.loads(FIXTURE.read_text())["codes"]
        if c["label"] == "[[54,8,4]]kasai:ec88c3e3"
    )
    h_x = np.array([[int(x) for x in row] for row in entry["H_X"]], dtype=np.uint8)
    h_z = np.array([[int(x) for x in row] for row in entry["H_Z"]], dtype=np.uint8)
    code = CSSCode(h_x, h_z)
    analysis = analyze_codeword_automorphisms(code)
    assert analysis.exact and analysis.certified
    assert analysis.group_order == 108
    ours = PermutationGroup(
        [Permutation(list(perm)) for perm in analysis.qubit_generators]
    )
    for cycles in entry["perm_generators"]:
        image = list(range(code.n))
        for cycle in re.findall(r"\(([\d,]+)\)", cycles):
            indices = [int(x) for x in cycle.split(",")]
            for position, value in enumerate(indices):
                image[value] = indices[(position + 1) % len(indices)]
        assert ours.contains(Permutation(image)), "census generator outside our group"


def test_bounded_path_agrees_with_full_enumeration() -> None:
    # Same group from both routes on codes where the full path is exact:
    # the bounded-weight information-set enumeration must reproduce it.
    for name, expected in [("steane", 168), ("c4-22", 24)]:
        code = CSSCode(*REGISTRY[name].build())
        bounded = analyze_codeword_automorphisms(code, full_enum_rank=1)
        assert bounded.exact and bounded.group_order == expected, name


def test_large_rank_codes_certify_exactly() -> None:
    # These ranks are far beyond 2^rank enumeration; the information-set
    # route certifies exact groups (values cross-validated 2026-08-20:
    # gross's 144 equals its independently certified Tanner qubit group).
    for name, expected in [("bb72", 432), ("gross", 144)]:
        code = CSSCode(*REGISTRY[name].build())
        analysis = analyze_codeword_automorphisms(code)
        assert analysis.exact and analysis.certified
        assert analysis.group_order == expected, name


def test_census_kasai_ranks_25_and_28_certify_exactly() -> None:
    # The two fixture codes the 2^rank path had to decline.
    expected = {"[[54,4,6]]kasai:e4d756a1": 27, "[[60,4,8]]kasai:98c88188": 30}
    for entry in json.loads(FIXTURE.read_text())["codes"]:
        if entry["label"] not in expected:
            continue
        h_x = np.array([[int(x) for x in row] for row in entry["H_X"]], dtype=np.uint8)
        h_z = np.array([[int(x) for x in row] for row in entry["H_Z"]], dtype=np.uint8)
        analysis = analyze_codeword_automorphisms(CSSCode(h_x, h_z))
        assert analysis.exact and analysis.certified
        assert analysis.group_order == expected[entry["label"]], entry["label"]


def test_out_of_budget_code_declines_honestly() -> None:
    # two-gross's minimum-weight classes sit past the h*t completeness bound
    # at the default work cap: the verdict must degrade, never guess.
    code = CSSCode(*REGISTRY["two-gross"].build())
    analysis = analyze_codeword_automorphisms(code)
    assert not analysis.exact
    assert analysis.notes


def test_truncated_selection_degrades_to_certified_subgroup() -> None:
    code = CSSCode(*REGISTRY["steane"].build())
    analysis = analyze_codeword_automorphisms(code, size_cap=1)
    assert not analysis.exact
    assert analysis.notes  # says why
    assert analysis.certified  # whatever survived is still certified
