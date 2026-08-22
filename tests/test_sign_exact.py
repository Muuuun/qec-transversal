"""Sign-exact certification across every backend.

The symplectic layer works modulo Pauli and global phase.  That is sound for
group orders, but it is not a circuit-level claim, and 0.1 only closed the gap
automatically for the strict *diagonal* generators.  Now every backend's
records carry a dense physical matrix, and :func:`certify_signs` lifts them to
exact Stim tableaux.

What is pinned here:

* every generator of every ansatz carries a physical matrix that really is
  symplectic and really preserves the code;
* :func:`certify_signs` certifies all of them;
* the one-block engine is *correctly skipped* rather than silently misread --
  its records hold logical actions, and the ``2k x 2k`` shape must not be
  mistaken for a physical ``2n x 2n`` matrix;
* a deliberately broken matrix is rejected.
"""

import numpy as np
import pytest

pytest.importorskip("stim")

from qec_transversal import (  # noqa: E402
    REGISTRY,
    CSSCode,
    certify_signs,
    five_qubit_code,
    matching_clifford_group,
    monomial_clifford_group,
    one_block_clifford_group,
    partition_clifford_group,
    permutation_automorphism_group,
    strict_transversal_clifford,
    transversal_clifford_across_blocks,
)
from qec_transversal.api import _physical_symplectic  # noqa: E402
from qec_transversal.utils.gf2 import gf2_matmul, rowspace_residues  # noqa: E402
from qec_transversal.utils.symplectic import is_symplectic  # noqa: E402


def _steane_css() -> CSSCode:
    return CSSCode(*REGISTRY["steane"].build())


def _backends():
    css = _steane_css()
    stab = css.to_stabilizer_code()
    yield "strict/css", css, strict_transversal_clifford(css)
    yield "strict/general", stab, strict_transversal_clifford(stab, method="general")
    yield "partition", stab, partition_clifford_group(
        stab, [(0, 1), (2, 3), (4, 5), (6,)], method="enumeration"
    )
    yield "matching", css, matching_clifford_group(css, np.arange(css.n))
    yield "five-qubit", five_qubit_code(), strict_transversal_clifford(five_qubit_code())


def test_every_backend_certifies_sign_exactly() -> None:
    for label, code, result in _backends():
        certificate = certify_signs(code, result)
        assert certificate.checked == len(result.generators), label
        assert certificate.skipped == 0, label
        assert certificate.all_preserved, label
        assert certificate.all_signs_plus, label
        assert certificate.certified, label
        assert len(certificate.corrections) == certificate.checked


def test_graph_backends_certify_sign_exactly() -> None:
    pytest.importorskip("igraph")
    css = _steane_css()
    stab = css.to_stabilizer_code()
    for label, code, result in (
        ("monomial", stab, monomial_clifford_group(stab)),
        ("permutation", css, permutation_automorphism_group(css)),
    ):
        certificate = certify_signs(code, result)
        assert certificate.certified, label
        assert certificate.skipped == 0, label


def test_cross_block_gates_certify_sign_exactly() -> None:
    from qec_transversal.codes.stabilizer import tensor_power

    single = _steane_css().to_stabilizer_code()
    result = transversal_clifford_across_blocks(single, blocks=2, method="enumeration")
    certificate = certify_signs(tensor_power(single, 2), result)
    assert certificate.certified
    assert certificate.skipped == 0


def test_every_physical_matrix_is_symplectic_and_preserves_the_code() -> None:
    """The matrices themselves, independently of the sign machinery."""

    for label, code, result in _backends():
        stab = code if hasattr(code, "h") else code.to_stabilizer_code()
        for record in result.generators:
            matrix = _physical_symplectic(record, stab.n)
            assert matrix is not None, label
            assert is_symplectic(matrix, qubits=stab.n), label
            assert not rowspace_residues(gf2_matmul(stab.h, matrix), stab.h).any(), label


def test_one_block_records_are_skipped_not_misread() -> None:
    """``OneBlockGenerator.matrix`` is a 2k x 2k logical action, not a gate."""

    css = _steane_css()
    result = one_block_clifford_group(css, name="steane", time_budget_s=20.0)
    assert result.generators
    for record in result.generators:
        assert record.matrix.shape == (2 * css.k, 2 * css.k)
        assert _physical_symplectic(record, css.n) is None
    certificate = certify_signs(css, result)
    assert certificate.checked == 0
    assert certificate.skipped == len(result.generators)
    assert not certificate.certified  # nothing checked is not a pass
    assert "skipped" in certificate.detail


def test_a_non_preserving_matrix_is_rejected() -> None:
    """The certificate must fail on a gate that does not fix the code."""

    from qec_transversal.certificates.signed import SignedStabilizer, verify_sign_exact

    css = _steane_css()
    stab = css.to_stabilizer_code()
    # a single-qubit Hadamard on qubit 0 alone: not code-preserving
    matrix = np.eye(2 * stab.n, dtype=np.uint8)
    matrix[0, 0] = matrix[stab.n, stab.n] = 0
    matrix[0, stab.n] = matrix[stab.n, 0] = 1
    outcome = verify_sign_exact(SignedStabilizer(stab.h), matrix)
    assert not outcome.preserved


def test_flipped_stabilizer_signs_change_the_correction() -> None:
    """The certificate is sensitive to the signed code, not just its rows."""

    css = _steane_css()
    stab = css.to_stabilizer_code()
    result = strict_transversal_clifford(stab, method="general")
    plain = certify_signs(stab, result)
    flipped_signs = [1] * stab.h.shape[0]
    flipped_signs[0] = -1
    flipped = certify_signs(stab, result, signs=flipped_signs)
    assert plain.certified and flipped.certified
    assert plain.corrections != flipped.corrections


def test_limit_bounds_the_work() -> None:
    css = _steane_css()
    stab = css.to_stabilizer_code()
    result = partition_clifford_group(
        stab, [(0, 1), (2, 3), (4, 5), (6,)], method="enumeration"
    )
    assert len(result.generators) > 3
    certificate = certify_signs(stab, result, limit=3)
    assert certificate.checked == 3
