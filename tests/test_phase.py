"""Sign-exact (Stim) phase verification tests."""

import pytest

pytest.importorskip("stim")

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.certificates.phase import verify_phases
from qec_transversal.codes import iceberg


@pytest.mark.parametrize("name", ["steane", "c4-22", "qrm15", "cube-832", "grid-4x6"])
def test_phase_verification_certifies_registry_codes(name) -> None:
    verification = verify_phases(CSSCode(*REGISTRY[name].build()))
    assert verification.certified


def test_weight_two_mod_four_check_forces_a_pauli_correction() -> None:
    # [[6,4,2]]: the global X-stabilizer has weight 6 = 2 mod 4, so the
    # transversal S layer conjugates it to MINUS itself times Z's — the
    # sign defect must be found, corrected by an explicit Pauli, and the
    # corrected gate certified.
    code = CSSCode(*iceberg(3))
    verification = verify_phases(code)
    assert verification.certified
    z_generator = next(g for g in verification.generators if g.family == "Z")
    assert set(z_generator.pauli_correction) - {"+", "_", "I"}  # nontrivial fix


def test_sign_exact_logical_phase_recorded() -> None:
    # Steane sqrt(X) layer flips the sign of the logical Z representative -
    # invisible at the symplectic level, captured here.
    verification = verify_phases(CSSCode(*REGISTRY["steane"].build()))
    x_generator = next(g for g in verification.generators if g.family == "X")
    assert x_generator.logical_diagonal_phases == (2,)
