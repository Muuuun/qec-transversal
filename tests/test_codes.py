"""Constructor and registry tests against published code parameters."""

import numpy as np
import pytest

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.codes import (
    bipartite_grid,
    bivariate_bicycle,
    doubled_color_41,
    generalized_bicycle,
    hypergraph_product,
    kasai_binary_pair,
    kasai_nonbinary,
    la_cross,
    middle_reed_muller,
    quantum_reed_muller_15,
    subset_inclusion,
    surface_code,
    toric_code,
)

# Instances small enough to build in the default test run; the large ones
# (bb756, lifted-b1, kasai-*) are exercised by the marked slow test below.
FAST_NAMES = [
    name
    for name, entry in REGISTRY.items()
    if entry.n <= 500
]


@pytest.mark.parametrize("name", FAST_NAMES)
def test_registry_matches_published_parameters(name: str) -> None:
    entry = REGISTRY[name]
    h_x, h_z = entry.build()
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (entry.n, entry.k)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name", [name for name in REGISTRY if name not in FAST_NAMES]
)
def test_registry_matches_published_parameters_large(name: str) -> None:
    entry = REGISTRY[name]
    h_x, h_z = entry.build()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (entry.n, entry.k)


def test_bivariate_bicycle_gross_code_has_trivial_strict_transversal_group() -> None:
    h_x, h_z = bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    analysis = CSSCode(h_x, h_z).analyze_transversal()
    assert analysis.a_z.dimension == 0
    assert analysis.a_x.dimension == 0
    assert analysis.certified


def test_quantum_reed_muller_15_finds_transversal_s() -> None:
    analysis = CSSCode(*quantum_reed_muller_15()).analyze_transversal()
    assert analysis.a_z.dimension == 5
    assert analysis.a_x.dimension == 0
    nontrivial = [g for g in analysis.generators if not g.is_logical_identity]
    assert len(nontrivial) == 5
    shear = np.asarray([[1, 1], [0, 1]], dtype=np.uint8)
    assert all(np.array_equal(g.logical_symplectic, shear) for g in nontrivial)
    report = analysis.to_dict()
    assert report["structure"]["all_ones_in_A_Z"] is True
    assert report["structure"]["logically_trivial_dimension_A_Z"] == 4
    assert report["logical_group"]["order"] == 2


def test_grid_code_is_self_dual_with_nontrivial_diagonal_gates() -> None:
    analysis = CSSCode(*bipartite_grid(4, 6)).analyze_transversal()
    report = analysis.to_dict()
    assert report["code"] == {"n": 24, "k": 8, "rank_X": 8, "rank_Z": 8}
    assert report["structure"]["self_dual"] is True
    assert report["structure"]["all_ones_in_A_Z"] is True
    assert report["structure"]["logically_nontrivial_rank_A_Z"] >= 1


def test_toric_and_surface_codes_have_no_strict_transversal_gates() -> None:
    for h_x, h_z in [toric_code(3), surface_code(3), la_cross(7, 3)]:
        analysis = CSSCode(h_x, h_z).analyze_transversal()
        assert analysis.a_z.dimension == 0
        assert analysis.a_x.dimension == 0


def test_generalized_bicycle_dimension_formula() -> None:
    # k = 2 * deg gcd(a, b, x^l - 1); for the PK A3 code this is 6.
    h_x, h_z = generalized_bicycle(24, [0, 2, 8, 15], [0, 2, 12, 17])
    assert CSSCode(h_x, h_z).k == 6


def test_hypergraph_product_of_full_rank_seeds() -> None:
    seed = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    h_x, h_z = hypergraph_product(seed, seed)
    code = CSSCode(h_x, h_z)
    assert code.n == 3 * 3 + 2 * 2
    assert code.k == 1  # k1*k2 + k1^T*k2^T = 1*1 + 0*0


def test_kasai_binary_pair_is_orthogonal_for_any_lift() -> None:
    for width, lift in [(6, 7), (6, 49), (8, 20)]:
        h_x, h_z = kasai_binary_pair(width, lift)
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        code = CSSCode(h_x, h_z)
        assert code.n == width * lift
        assert code.k == (width - 4) * lift + 2


def test_kasai_nonbinary_expansion_preserves_orthogonality() -> None:
    h_x, h_z = kasai_nonbinary(6, 7, extension=4, seed=3)
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert code.n == 4 * 6 * 7
    assert code.k == 4 * ((6 - 4) * 7 + 2)


def test_middle_reed_muller_tesseract() -> None:
    code = CSSCode(*middle_reed_muller(4))
    assert (code.n, code.k) == (16, 6)
    report = code.analyze_transversal().to_dict()
    assert report["structure"]["self_dual"] is True


def test_gf2_matmul_large_matches_integer_path() -> None:
    from qec_transversal.gf2 import gf2_matmul

    rng = np.random.default_rng(11)
    left = rng.integers(0, 2, size=(700, 900), dtype=np.uint8)
    right = rng.integers(0, 2, size=(900, 800), dtype=np.uint8)
    # Accelerate's sgemm pollutes FP flags (underflow); the BLAS path must
    # stay silent even under raise-on-everything.
    with np.errstate(all="raise"):
        fast = gf2_matmul(left, right)  # large: takes the float32 BLAS path
    exact = (left.astype(np.int64) @ right.astype(np.int64)) % 2
    assert np.array_equal(fast, exact.astype(np.uint8))


def test_schreier_sims_matches_symplectic_orders_and_closure() -> None:
    from qec_transversal.group import (
        generated_group_order,
        schreier_sims_order,
        symplectic_group_order,
    )

    def transvection(vector: np.ndarray, qubits: int) -> np.ndarray:
        size = 2 * qubits
        form = np.zeros((size, size), dtype=np.uint8)
        form[:qubits, qubits:] = np.eye(qubits, dtype=np.uint8)
        form[qubits:, :qubits] = np.eye(qubits, dtype=np.uint8)
        matrix = np.eye(size, dtype=np.uint8)
        for index in range(size):
            basis = np.zeros(size, dtype=np.uint8)
            basis[index] = 1
            coefficient = int(basis @ form @ vector) & 1
            matrix[index] = (basis ^ (coefficient * vector)) & 1
        return matrix

    for qubits in (1, 2, 3):
        size = 2 * qubits
        generators = []
        for index in range(size):
            vector = np.zeros(size, dtype=np.uint8)
            vector[index] = 1
            generators.append(transvection(vector, qubits))
        for index in range(size - 1):
            vector = np.zeros(size, dtype=np.uint8)
            vector[index] = vector[index + 1] = 1
            generators.append(transvection(vector, qubits))
        assert schreier_sims_order(generators) == symplectic_group_order(qubits)

    rng = np.random.default_rng(5)
    for _ in range(5):
        generators = []
        for _ in range(3):
            symmetric = rng.integers(0, 2, size=(2, 2), dtype=np.uint8)
            symmetric = (symmetric + symmetric.T) % 2
            generators.append(
                np.block(
                    [
                        [np.eye(2, dtype=np.uint8), symmetric],
                        [np.zeros((2, 2), dtype=np.uint8), np.eye(2, dtype=np.uint8)],
                    ]
                ).astype(np.uint8)
            )
        closure = generated_group_order(generators, cap=100_000)
        assert closure.exact
        assert schreier_sims_order(generators) == closure.order


def test_subset_inclusion_reproduces_published_qlrc_parameters() -> None:
    h_x, h_z = subset_inclusion(6, 3, 2)
    assert h_x.shape == (6, 20)
    assert (h_x == h_z).all()
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (20, 8)


def test_doubled_color_41_certifies_full_k1_clifford() -> None:
    h_x, h_z = doubled_color_41()
    assert (h_x == h_z).all()
    assert h_x.shape == (20, 41)
    assert not (h_x.sum(axis=1) % 4).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (41, 1)
    report = code.analyze_transversal().to_dict()
    assert report["structure"]["self_dual"]
    assert report["logical_group"]["order"] == 6
    assert report["logical_group"]["is_full_logical_clifford"]
