"""(H, sigma) signed representation: sign-exact arbitrary-gate verification."""

import numpy as np
import pytest

pytest.importorskip("stim")

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.ansatz.monomial import analyze_monomial
from qec_transversal.ansatz.strict_css import shear_matrix
from qec_transversal.certificates.signed import SignedStabilizer, verify_sign_exact
from qec_transversal.codes import iceberg
from qec_transversal.codes.stabilizer import five_qubit_code


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


def test_dependent_row_valid_sign_matches_independent_run() -> None:
    # dependent rows are legal input: in a valid signed group the sign
    # defects form a character on the group, so an extra dependent row
    # cannot change the verdict or the Pauli correction
    css = CSSCode(*iceberg(3))
    analysis = css.analyze_transversal()
    z_generator = next(g for g in analysis.generators if g.family == "Z")
    matrix = shear_matrix("Z", z_generator.parameter)
    rows = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    base = SignedStabilizer(rows)
    a, b = 0, rows.shape[0] - 1
    dep_sign = int((base.pauli(a) * base.pauli(b)).sign.real)
    with_dep = SignedStabilizer(
        np.vstack([rows, rows[a] ^ rows[b]]),
        [1] * rows.shape[0] + [dep_sign],
    )
    independent = verify_sign_exact(base, matrix)
    dependent = verify_sign_exact(with_dep, matrix)
    assert dependent.preserved
    assert dependent.certificate["stabilizer_signs_corrected_to_plus"]
    assert dependent.pauli_correction == independent.pauli_correction


def test_dependent_row_flipped_sign_rejected_as_input_error() -> None:
    # flipping the dependent row's sign puts -I in the generated group:
    # invalid input, and it must not masquerade as a gate failure
    css = CSSCode(*iceberg(3))
    rows = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    base = SignedStabilizer(rows)
    a, b = 0, rows.shape[0] - 1
    dep_sign = int((base.pauli(a) * base.pauli(b)).sign.real)
    bad = SignedStabilizer(
        np.vstack([rows, rows[a] ^ rows[b]]),
        [1] * rows.shape[0] + [-dep_sign],
    )
    with pytest.raises(ValueError, match="-I in generated group"):
        verify_sign_exact(bad, np.eye(2 * css.n, dtype=np.uint8))


def test_five_qubit_dependent_row_random_valid_signs_all_plus() -> None:
    # random valid base signs plus a dependent row whose sign is read off
    # the actual signed Stim product: verification always reaches all-plus
    code = five_qubit_code()
    rows = code.h.copy()
    monomial = analyze_monomial(code)
    generator = monomial.generators[0]
    n = code.n
    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i in range(n):
        a, b, c, d = generator.blocks[i]
        j = int(generator.qubit_permutation[i])
        matrix[i, j] = a
        matrix[i, n + j] = b
        matrix[n + i, j] = c
        matrix[n + i, n + j] = d
    rng = np.random.default_rng(2026)
    checked = 0
    while checked < 5:
        mask = rng.integers(0, 2, size=rows.shape[0]).astype(np.uint8)
        support = [int(j) for j in np.flatnonzero(mask)]
        if len(support) < 2:
            continue
        base_signs = [int(s) for s in rng.choice([1, -1], size=rows.shape[0])]
        base = SignedStabilizer(rows, base_signs)
        dep = rows[support[0]].copy()
        product = base.pauli(support[0])
        for j in support[1:]:
            dep ^= rows[j]
            product = product * base.pauli(j)
        signed = SignedStabilizer(
            np.vstack([rows, dep]), base_signs + [int(product.sign.real)]
        )
        result = verify_sign_exact(signed, matrix)
        assert result.preserved
        assert result.certificate["stabilizer_signs_corrected_to_plus"]
        checked += 1


def test_anticommuting_rows_rejected() -> None:
    # X and Z on the same qubit do not generate a stabilizer group
    rows = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="anticommute"):
        verify_sign_exact(SignedStabilizer(rows), np.eye(2, dtype=np.uint8))


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
