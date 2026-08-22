"""The public API surface: one result shape, honest completeness values."""

import numpy as np
import pytest

import qec_transversal
from qec_transversal import (
    REGISTRY,
    Completeness,
    CSSCode,
    GateSearchResult,
    StabilizerCode,
    css_strict_transversal_clifford,
    diagonal_transversal_gates,
    five_qubit_code,
    matching_clifford_group,
    monomial_clifford_group,
    one_block_clifford_group,
    partition_clifford_group,
    permutation_automorphism_group,
    strict_transversal_clifford,
)

REQUIRED_FIELDS = (
    "method",
    "ansatz",
    "completeness",
    "generators",
    "logical_generators",
    "group_order",
    "logical_group_order",
    "logical_group_order_is_exact",
    "certificate",
    "metadata",
)


def _steane() -> CSSCode:
    return CSSCode(*REGISTRY["steane"].build())


def _assert_result_shape(result: GateSearchResult) -> None:
    assert isinstance(result, GateSearchResult)
    for field in REQUIRED_FIELDS:
        assert hasattr(result, field), field
    assert isinstance(result.completeness, Completeness)
    assert result.complete is (result.completeness is Completeness.COMPLETE)
    assert isinstance(result.method, str) and result.method
    assert isinstance(result.ansatz, str) and result.ansatz
    assert isinstance(result.certificate, dict)
    assert isinstance(result.metadata, dict)
    for matrix in result.logical_generators:
        assert isinstance(matrix, np.ndarray)
    # a group order is either absent or a positive integer -- never 0, and
    # never a float
    for order in (result.group_order, result.logical_group_order):
        assert order is None or (isinstance(order, int) and order >= 1)


def test_every_entry_point_returns_the_same_result_shape() -> None:
    css = _steane()
    results = [
        strict_transversal_clifford(css),
        strict_transversal_clifford(css, method="general"),
        css_strict_transversal_clifford(css),
        partition_clifford_group(css, [(0, 1), (2, 3), (4, 5), (6,)]),
        matching_clifford_group(css, np.arange(css.n)),
        diagonal_transversal_gates(css, level=3),
        one_block_clifford_group(css, name="steane", time_budget_s=20.0),
        strict_transversal_clifford(five_qubit_code()),
    ]
    for result in results:
        _assert_result_shape(result)


def test_results_serialise_to_json_safe_dicts() -> None:
    import json

    css = _steane()
    for result in (
        strict_transversal_clifford(css),
        partition_clifford_group(css, [(0, 1), (2, 3), (4, 5), (6,)]),
        diagonal_transversal_gates(css, level=3),
    ):
        payload = result.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["completeness"] in {c.value for c in Completeness}
        assert payload["complete"] is (payload["completeness"] == "COMPLETE")


def test_completeness_is_a_string_enum_with_exactly_three_values() -> None:
    assert {c.value for c in Completeness} == {
        "COMPLETE",
        "INCOMPLETE_LOWER_BOUND",
        "UNKNOWN",
    }
    assert Completeness.COMPLETE == "COMPLETE"  # JSON-friendly


def test_auto_method_dispatch() -> None:
    css = _steane()
    assert strict_transversal_clifford(css).method == "css-shear-kernel"
    assert "preservation-algebra" in strict_transversal_clifford(
        css.to_stabilizer_code()
    ).method
    with pytest.raises(TypeError):
        strict_transversal_clifford(five_qubit_code(), method="css")
    with pytest.raises(ValueError):
        strict_transversal_clifford(css, method="nonsense")


def test_css_to_stabilizer_bridge_is_the_same_code() -> None:
    css = _steane()
    stabilizer = css.to_stabilizer_code()
    assert isinstance(stabilizer, StabilizerCode)
    assert (stabilizer.n, stabilizer.k) == (css.n, css.k)


def test_partition_must_cover_every_qubit() -> None:
    css = _steane()
    with pytest.raises(ValueError, match="partition"):
        partition_clifford_group(css, [(0, 1), (2, 3)])


def test_ansatz_string_names_the_gate_class() -> None:
    """A result must be self-describing: no bare 'transversal'."""

    css = _steane()
    strict = strict_transversal_clifford(css)
    assert "single-qubit Clifford per qubit" in strict.ansatz
    partition = partition_clifford_group(css, [(0, 1), (2, 3), (4, 5), (6,)])
    assert "partition cell" in partition.ansatz
    matching = matching_clifford_group(css, np.arange(css.n))
    assert "Levi" in matching.ansatz  # the excluded factor is stated up front


def test_one_block_completeness_is_one_sided() -> None:
    """Fullness is a certificate; falling short is a lower bound, not a no-go."""

    css = _steane()
    full = one_block_clifford_group(css, name="steane", time_budget_s=30.0)
    assert full.completeness is Completeness.COMPLETE
    assert full.certificate["is_full_symplectic"] is True

    toric = CSSCode(*REGISTRY["toric-4"].build())
    short = one_block_clifford_group(toric, name="toric-4", time_budget_s=30.0)
    assert short.completeness is not Completeness.COMPLETE
    assert short.certificate["is_full_symplectic"] is not False  # True or None
    assert "one_sided" in short.metadata


@pytest.mark.parametrize("name", ["steane", "c4-22"])
def test_permutation_and_monomial_report_their_scope(name: str) -> None:
    pytest.importorskip("igraph")
    css = CSSCode(*REGISTRY[name].build())

    tanner = permutation_automorphism_group(css, method="tanner")
    # the Tanner graph is row-SET scoped, so it can never claim completeness
    assert tanner.completeness is Completeness.INCOMPLETE_LOWER_BOUND
    assert "row-SET" in tanner.metadata["scope"]

    codewords = permutation_automorphism_group(css, method="codewords")
    assert codewords.completeness in (
        Completeness.COMPLETE,
        Completeness.INCOMPLETE_LOWER_BOUND,
    )
    # the row-space group contains the Tanner group
    assert codewords.group_order >= tanner.group_order

    mono = monomial_clifford_group(css)
    assert isinstance(mono.metadata["row_set_complete"], bool)
    assert mono.complete is mono.metadata["row_set_complete"]


def test_diagonal_gates_route_by_code_class() -> None:
    """One entry point, two solvers, and each says which one ran."""

    pytest.importorskip("stim")
    css = _steane()
    css_result = diagonal_transversal_gates(css, level=3)
    assert css_result.method == "Z_{2^L} coset-phase module kernel"
    assert css_result.completeness is Completeness.COMPLETE

    general = diagonal_transversal_gates(css.to_stabilizer_code(), level=3)
    assert "general stabilizer" in general.method
    # the general engine is sound always and complete here (small code)
    assert general.completeness in (
        Completeness.COMPLETE,
        Completeness.INCOMPLETE_LOWER_BOUND,
    )
    assert general.certificate["sound"] is True
    assert general.metadata["kernel_generators"] >= 1

    perfect = diagonal_transversal_gates(five_qubit_code(), level=3)
    assert perfect.completeness is Completeness.COMPLETE
    # the [[5,1,3]] code has no diagonal gate beyond the Paulis
    assert perfect.metadata["beyond_pauli_generators"] == 0

    with pytest.raises(ValueError, match="family"):
        diagonal_transversal_gates(five_qubit_code(), family="X")


def test_package_exports_are_importable_and_documented() -> None:
    for name in qec_transversal.__all__:
        assert hasattr(qec_transversal, name), name
    assert qec_transversal.__version__ == "0.2.1"
    assert qec_transversal.__doc__
