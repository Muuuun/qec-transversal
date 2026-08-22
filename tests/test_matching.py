"""Fold-transversal solver tests: brute force, dualities, known folds."""

import numpy as np
import pytest

from qec_transversal import CSSCode
from qec_transversal.ansatz.dualities import candidates_for, hgp_transpose, two_block_inversion
from qec_transversal.ansatz.matching import analyze_matching, sigma_matrix
from qec_transversal.codes import steane_code, surface_code, toric_code
from qec_transversal.utils.gf2 import nullspace, rowspace_residues


def _random_involution(rng: np.random.Generator, n: int) -> np.ndarray:
    tau = np.arange(n)
    order = rng.permutation(n)
    used = np.zeros(n, dtype=bool)
    for index in range(0, n - 1, 2):
        i, j = order[index], order[index + 1]
        if not used[i] and not used[j] and rng.random() < 0.7:
            tau[i], tau[j] = j, i
            used[i] = used[j] = True
    return tau


def _legal_by_brute_force(code: CSSCode, tau, pairs, family: str) -> set[bytes]:
    """All parameter vectors whose Sigma-shear preserves the stabilizer."""

    n = code.n
    width = n + len(pairs)
    source = code.h_x if family == "Z" else code.h_z
    target = code.c_z if family == "Z" else code.c_x
    legal = set()
    for mask in range(1 << width):
        parameter = np.array([(mask >> b) & 1 for b in range(width)], dtype=np.uint8)
        sigma = sigma_matrix(parameter, pairs, n)
        image = (source.astype(np.int64) @ sigma.astype(np.int64) % 2).astype(np.uint8)
        if not rowspace_residues(image, target).any():
            legal.add(parameter.tobytes())
    return legal


def _span(basis: np.ndarray, width: int) -> set[bytes]:
    out = set()
    for mask in range(1 << basis.shape[0]):
        vector = np.zeros(width, dtype=np.uint8)
        for row in range(basis.shape[0]):
            if (mask >> row) & 1:
                vector ^= basis[row]
        out.add(vector.tobytes())
    return out


def test_matching_kernels_match_brute_force_on_random_codes() -> None:
    rng = np.random.default_rng(4711)
    checked = 0
    for n in range(3, 7):
        for _ in range(6):
            rows_x = rng.integers(0, n + 1)
            h_x = rng.integers(0, 2, size=(rows_x, n), dtype=np.uint8)
            x_perp = nullspace(h_x)
            rows_z = rng.integers(0, n + 1)
            z = rng.integers(0, 2, size=(rows_z, x_perp.shape[0]), dtype=np.uint8)
            h_z = (z.astype(np.int64) @ x_perp.astype(np.int64) % 2).astype(np.uint8)
            code = CSSCode(h_x, h_z, n=n)
            tau = _random_involution(rng, n)
            analysis = analyze_matching(code, tau)
            width = n + len(analysis.pairs)
            assert _span(analysis.z_basis, width) == _legal_by_brute_force(
                code, tau, analysis.pairs, "Z"
            )
            assert _span(analysis.x_basis, width) == _legal_by_brute_force(
                code, tau, analysis.pairs, "X"
            )
            assert analysis.certified
            checked += 1
    assert checked == 24


def test_matching_kernel_contains_strict_space() -> None:
    # With c = 0 the matching kernel must reproduce the strict A_Z exactly.
    code = CSSCode(*steane_code())
    tau = np.array([1, 0, 3, 2, 5, 4, 6])  # arbitrary involution
    analysis = analyze_matching(code, tau)
    strict = code.analyze_transversal()
    width = code.n + len(analysis.pairs)
    fold_span = _span(analysis.z_basis, width)
    strict_span = {
        np.concatenate([v, np.zeros(len(analysis.pairs), dtype=np.uint8)]).tobytes()
        for v in (
            np.zeros(code.n, dtype=np.uint8),
            *strict.a_z.basis,
        )
    }
    assert strict_span <= fold_span


def test_toric_and_surface_transpose_folds_are_dualities_with_full_gates() -> None:
    code = CSSCode(*surface_code(3))
    analysis = analyze_matching(code, hgp_transpose(3, 2))
    report = analysis.to_dict()
    assert report["is_zx_duality"]
    assert report["fold_hadamard_nontrivial"]
    # folded surface code: S and H folds generate the full logical Clifford
    assert report["logical_group"]["order"] == 6
    assert report["certified"]

    toric = CSSCode(*toric_code(3))
    toric_report = analyze_matching(toric, hgp_transpose(3, 3)).to_dict()
    assert toric_report["is_zx_duality"]
    assert toric_report["logical_group"]["order"] == 6


def test_bb_tau0_is_a_duality_and_gross_matching_is_fixed_point_free() -> None:
    from qec_transversal.codes import bivariate_bicycle

    h_x, h_z = bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    code = CSSCode(h_x, h_z)
    analysis = analyze_matching(code, two_block_inversion(12, 6))
    report = analysis.to_dict()
    assert report["is_zx_duality"]
    assert report["fixed_points"] == 0  # the gross duality is fixed-point-free
    assert report["dim_S_MZ"] >= 1  # the CZ-matching fold layer exists
    assert report["nontrivial_generator_count"] >= 1
    assert report["certified"]


def test_invalid_candidates_are_rejected_not_crashed() -> None:
    code = CSSCode(*steane_code())
    tau = np.arange(7)  # identity: a duality only for self-dual codes
    analysis = analyze_matching(code, tau)
    assert analysis.is_zx_duality  # Steane IS self-dual
    swapped = np.array([1, 0, 2, 3, 4, 5, 6])
    assert analyze_matching(code, swapped).certified


def test_registry_candidates_all_run() -> None:
    from qec_transversal import REGISTRY

    for name in ["coprime30", "trivariate30", "gb46"]:
        code = CSSCode(*REGISTRY[name].build())
        for _, tau in candidates_for(name):
            assert analyze_matching(code, tau).certified


def test_non_involution_rejected() -> None:
    code = CSSCode(*steane_code())
    with pytest.raises(ValueError):
        analyze_matching(code, np.array([1, 2, 0, 3, 4, 5, 6]))
