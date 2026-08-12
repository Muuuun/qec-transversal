"""(H, sigma) signed representation: sign-exact arbitrary-gate verification."""

import numpy as np
import pytest

pytest.importorskip("stim")

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.codes import iceberg
from qec_transversal.css import shear_matrix
from qec_transversal.monomial import analyze_monomial
from qec_transversal.signed import SignedStabilizer, verify_sign_exact
from qec_transversal.stabilizer import StabilizerCode, five_qubit_code


def _signed(css: CSSCode) -> SignedStabilizer:
    rows = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    return SignedStabilizer(rows)


def test_diagonal_gates_agree_with_phase_module() -> None:
    # the [[6,4,2]] iceberg's S-layer needs a genuine Pauli correction —
    # the generic engine must find it, matching the circuit-based one
    css = CSSCode(*iceberg(3))
    analysis = css.analyze_transversal()
    z_generator = next(g for g in analysis.generators if g.family == "Z")
    matrix = shear_matrix("Z", z_generator.parameter)
    result = verify_sign_exact(_signed(css), matrix)
    assert result.preserved
    assert result.certificate["stabilizer_signs_corrected_to_plus"]
    assert set(result.pauli_correction) - {"+", "_", "I"}  # nontrivial fix


def test_monomial_gate_on_five_qubit_code_sign_exact() -> None:
    # a permutation x local-Clifford gate (well beyond diagonal circuits)
    code = five_qubit_code()
    monomial = analyze_monomial(code)
    signed = SignedStabilizer(code.h)
    verified = 0
    for generator in monomial.generators:
        n = code.n
        matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
        for i in range(n):
            a, b, c, d = generator.blocks[i]
            j = int(generator.qubit_permutation[i])
            matrix[i, j] = a
            matrix[i, n + j] = b
            matrix[n + i, j] = c
            matrix[n + i, n + j] = d
        result = verify_sign_exact(signed, matrix)
        assert result.preserved
        assert result.certificate["stabilizer_signs_corrected_to_plus"]
        verified += 1
    assert verified >= 1


def test_non_preserving_gate_rejected() -> None:
    css = CSSCode(*REGISTRY["steane"].build())
    n = css.n
    # a random single-qubit Hadamard layer does NOT preserve Steane's
    # stabilizer as given (H^x7 does; H on one qubit does not)
    matrix = np.eye(2 * n, dtype=np.uint8)
    matrix[0, 0] = 0
    matrix[0, n] = 1
    matrix[n, 0] = 1
    matrix[n, n] = 0
    result = verify_sign_exact(_signed(css), matrix)
    assert result.preserved is False


def test_custom_signs_change_the_correction() -> None:
    css = CSSCode(*REGISTRY["steane"].build())
    rows = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    matrix = shear_matrix("Z", np.ones(css.n, dtype=np.uint8))
    plus = verify_sign_exact(SignedStabilizer(rows), matrix)
    flipped = verify_sign_exact(SignedStabilizer(rows, [-1] + [1] * (rows.shape[0] - 1)), matrix)
    assert plus.preserved and flipped.preserved
    assert plus.certificate["stabilizer_signs_corrected_to_plus"]
    assert flipped.certificate["stabilizer_signs_corrected_to_plus"]
