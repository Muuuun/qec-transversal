"""The pre-0.2 flat module paths still resolve, to the very same objects.

The 0.2 refactor reorganised ``qec_transversal`` by concept.  Thin alias
modules keep every 0.1 import path working; this file is the contract for
that promise, and it checks object *identity*, not merely importability -- an
alias that re-implemented something would be worse than one that broke.

See ``docs/refactor_report.md`` for the full old -> new mapping.
"""

import importlib

import pytest

#: (legacy module, symbol, new module) for every publicly documented name.
ALIASES = [
    ("gf2", "rref", "utils.gf2"),
    ("gf2", "nullspace", "utils.gf2"),
    ("gf2", "gf2_inverse", "utils.gf2"),
    ("gf2", "gf2_matmul", "utils.gf2"),
    ("gf2", "quotient_complement", "utils.gf2"),
    ("gf2", "rowspace_residues", "utils.gf2"),
    ("gf2", "symplectic_form", "utils.symplectic"),
    ("gf2", "symplectic_product", "utils.symplectic"),
    ("gf2", "is_symplectic", "utils.symplectic"),
    ("group", "schreier_sims_order", "logical.group"),
    ("group", "generated_group_order", "logical.group"),
    ("group", "logical_group_summary", "logical.group"),
    ("group", "symplectic_group_order", "utils.symplectic"),
    ("css", "CSSCode", "codes.css"),
    ("css", "shear_matrix", "ansatz.strict_css"),
    ("css", "shear_images", "ansatz.strict_css"),
    ("css", "ParameterSpace", "ansatz.strict_css"),
    ("css", "TransversalAnalysis", "ansatz.strict_css"),
    ("css", "TransversalGenerator", "ansatz.strict_css"),
    ("stabilizer", "StabilizerCode", "codes.stabilizer"),
    ("stabilizer", "five_qubit_code", "codes.stabilizer"),
    ("stabilizer", "local_clifford_algebra", "algebra.preservation"),
    ("stabilizer", "partition_algebra", "algebra.preservation"),
    ("stabilizer", "analyze_local_clifford", "ansatz.strict"),
    ("stabilizer", "LocalCliffordAnalysis", "ansatz.strict"),
    ("stabilizer", "analyze_partition_clifford", "ansatz.partition"),
    ("stabilizer", "PartitionCliffordAnalysis", "ansatz.partition"),
    ("stabilizer", "partition_units_via_structure", "ansatz.partition"),
    ("stabilizer", "symplectic_gram_schmidt", "utils.symplectic"),
    ("matching", "analyze_matching", "ansatz.matching"),
    ("matching", "MatchingAnalysis", "ansatz.matching"),
    ("matching", "sigma_matrix", "ansatz.matching"),
    ("matching", "involution_pairs", "ansatz.matching"),
    ("matching", "logical_group_summary", "logical.group"),
    ("monomial", "analyze_monomial", "ansatz.monomial"),
    ("monomial", "strict_cross_check", "ansatz.monomial"),
    ("automorphisms", "analyze_automorphisms", "ansatz.permutation"),
    ("automorphisms", "describe_permutation", "ansatz.permutation"),
    ("automorphisms", "permutation_group_order", "utils.permutations"),
    ("codewordaut", "analyze_codeword_automorphisms", "ansatz.codeword_permutation"),
    ("axes", "diagonal_kernel_general", "hierarchy.general"),
    ("axes", "diagonal_kernel_general_exact", "hierarchy.general"),
    ("axes", "axis_frame_group", "hierarchy.frames"),
    ("axes", "frame_conjugated_code", "hierarchy.frames"),
    ("twofold", "two_fold_group", "ansatz.twofold"),
    ("twofold", "levi_logical_generators", "ansatz.twofold"),
    ("twofold", "automorphism_involutions", "ansatz.twofold"),
    ("dualities", "candidates_for", "ansatz.dualities"),
    ("dualities", "two_block_inversion", "ansatz.dualities"),
    ("discovery", "discover_involutions", "ansatz.discovery"),
    ("discovery", "certified_shift_structure", "ansatz.discovery"),
    ("discovery", "structural_permutations", "ansatz.discovery"),
    ("oneblock", "analyze_one_block", "logical.generated"),
    ("oneblock", "single_matching_fullness", "logical.generated"),
    ("oneblock", "factor_target", "logical.generated"),
    ("oneblock", "recognize_full_symplectic", "logical.recognition"),
    ("oneblock", "RecognitionReport", "logical.recognition"),
    ("oneblock", "WordBSGS", "logical.words"),
    ("oneblock", "symplectic_transvection", "utils.symplectic"),
    ("synthesis", "verify_logical_gate", "logical.synthesis"),
    ("synthesis", "logical_target", "logical.synthesis"),
    ("witness", "export_strict_witness", "certificates.witness"),
    ("witness", "export_stabilizer_witness", "certificates.witness"),
    ("witness", "write_witness", "certificates.witness"),
    ("signed", "SignedStabilizer", "certificates.signed"),
    ("signed", "verify_sign_exact", "certificates.signed"),
    ("phase", "verify_phases", "certificates.phase"),
    ("unitgroup", "AlgebraF2", "algebra.finite_algebra"),
    ("unitgroup", "unit_group", "algebra.unit_group"),
    ("unitgroup", "UnitGroupResult", "algebra.unit_group"),
]


@pytest.mark.parametrize("legacy,symbol,new", ALIASES, ids=lambda v: str(v))
def test_legacy_import_path_resolves_to_the_same_object(legacy, symbol, new) -> None:
    old_module = importlib.import_module(f"qec_transversal.{legacy}")
    new_module = importlib.import_module(f"qec_transversal.{new}")
    assert hasattr(old_module, symbol), f"qec_transversal.{legacy}.{symbol} vanished"
    assert getattr(old_module, symbol) is getattr(new_module, symbol)


def test_package_level_names_from_0_1_survive() -> None:
    import qec_transversal

    for name in (
        "REGISTRY",
        "CSSCode",
        "NamedCode",
        "ParameterSpace",
        "TransversalAnalysis",
        "TransversalGenerator",
        "gf2_inverse",
        "nullspace",
        "rank",
        "row_basis",
        "rref",
        "symplectic_form",
    ):
        assert hasattr(qec_transversal, name), name


def test_codes_module_still_exposes_every_constructor() -> None:
    """``qec_transversal.codes`` became a package; its facade must be complete."""

    from qec_transversal import codes

    for name in (
        "REGISTRY",
        "NamedCode",
        "bivariate_bicycle",
        "circulant",
        "cyclic_shift",
        "generalized_bicycle",
        "hypergraph_product",
        "iceberg",
        "kasai_binary_pair",
        "kasai_nonbinary",
        "middle_reed_muller",
        "quantum_reed_muller_15",
        "quantum_tanner_lift",
        "self_dual_bicycle",
        "steane_code",
        "surface_code",
        "toric_code",
    ):
        assert hasattr(codes, name), name


def test_hierarchy_package_keeps_the_0_1_surface() -> None:
    """``hierarchy.py`` became a package; ``analyze_hierarchy`` must survive."""

    from qec_transversal import hierarchy
    from qec_transversal.hierarchy.css import analyze_hierarchy

    assert hierarchy.analyze_hierarchy is analyze_hierarchy


def test_css_code_keeps_its_convenience_method() -> None:
    from qec_transversal import REGISTRY, CSSCode

    code = CSSCode(*REGISTRY["steane"].build())
    analysis = code.analyze_transversal()
    assert analysis.certified
    assert analysis.a_z.dimension == 1 and analysis.a_x.dimension == 1
