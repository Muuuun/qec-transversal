"""Constructor and registry tests against published code parameters."""

import numpy as np
import pytest

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.codes import (
    bipartite_grid,
    bivariate_bicycle,
    doubled_color_41,
    generalized_bicycle,
    helix_code,
    helper_qss_css,
    hypergraph_product,
    kasai_binary_pair,
    kasai_nonbinary,
    la_cross,
    middle_reed_muller,
    quantum_reed_muller_15,
    self_dual_bicycle,
    subset_inclusion,
    surface_code,
    toric_code,
)
from qec_transversal.utils.gf2 import rank

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


@pytest.mark.parametrize(
    ("monomials", "basis_1", "basis_2", "expected"),
    [
        ([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 4), (2, 2), (16, 4)),
        ([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 8), (4, 4), (64, 8)),
        ([(0, 0), (1, 0), (2, 1), (-1, 1)], (0, 3), (5, 0), (30, 6)),
        ([(0, 0), (1, 0), (2, 1), (-1, 1)], (0, 7), (4, 3), (56, 6)),
        ([(0, 0), (1, 0), (2, 1), (-1, 1)], (0, 19), (4, 6), (152, 6)),
        ([(0, 0), (1, 0), (2, 2), (-1, 1)], (0, 10), (8, 0), (160, 8)),
    ],
)
def test_self_dual_bicycle_reproduces_liang_chen_parameters(
    monomials: list, basis_1: tuple, basis_2: tuple, expected: tuple
) -> None:
    """Tables I-II of arXiv:2510.05211, including the twisted-torus cases."""

    h_x, h_z = self_dual_bicycle(monomials, basis_1, basis_2)
    assert np.array_equal(h_x, h_z)  # self-dual: C_X = C_Z
    assert set(h_x.sum(axis=1)) == {8}  # weight-8, hence doubly even
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == expected


def test_self_dual_bicycle_is_the_registry_positive_ldpc_control() -> None:
    """These sparse codes carry a *strict* transversal gate -- the counterexample
    to reading the qLDPC census as a claim about check sparsity."""

    for name, entry in REGISTRY.items():
        if entry.family != "self-dual-bb":
            continue
        analysis = CSSCode(*entry.build()).analyze_transversal()
        assert analysis.certified, name
        assert analysis.a_z.dimension > 0 and analysis.a_x.dimension > 0, name
        structure = analysis.to_dict()["structure"]
        assert structure["self_dual"], name
        assert structure["logically_nontrivial_rank_A_Z"] > 0, name
        assert structure["logically_nontrivial_rank_A_X"] > 0, name


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
    from qec_transversal.utils.gf2 import gf2_matmul

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
    from qec_transversal.logical.group import generated_group_order, schreier_sims_order
    from qec_transversal.utils.symplectic import symplectic_group_order

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


def test_helper_qss_css_reproduces_the_papers_printed_generators() -> None:
    # arXiv:2609.00220 Ex. 6 prints the m = 3 and m = 5 members in full.
    steane = ["XIIXXXI", "IXIXXIX", "IIXXIXX"]
    h_x, h_z = helper_qss_css(3)
    assert (h_x == h_z).all()
    assert ["".join("X" if bit else "I" for bit in row) for row in h_x] == steane
    eleven = [
        "XIIIIXXXXXI",
        "IXIIIXXXXIX",
        "IIXIIXXXIXX",
        "IIIXIXXIXXX",
        "IIIIXXIXXXX",
    ]
    h_x, h_z = helper_qss_css(5)
    assert ["".join("X" if bit else "I" for bit in row) for row in h_x] == eleven
    for parties in (3, 5, 7, 9):
        h_x, h_z = helper_qss_css(parties)
        code = CSSCode(h_x, h_z)
        assert (code.n, code.k) == (2 * parties + 1, 1)
        report = code.analyze_transversal().to_dict()
        assert report["structure"]["self_dual"]
        assert report["logical_group"]["order"] == 6
        assert report["logical_group"]["is_full_logical_clifford"]
    with pytest.raises(ValueError):
        helper_qss_css(4)


def test_helix_codes_rebuild_the_published_concatenated_double_codes() -> None:
    # arXiv:2609.03194 Table I prints the [[20,2,6]] C4-Helix generators in
    # full (some lifted checks multiplied by an inner check, which leaves the
    # stabilizer group alone); the [[60,2,12]] and [[100,2,18]] members swap
    # the inner code.
    published = [
        (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19),
        (0, 3, 6, 7, 10, 11, 12, 15), (4, 7, 8, 9, 14, 15, 16, 19),
        (1, 2, 8, 11, 12, 13, 18, 19), (2, 3, 5, 6, 13, 14, 16, 17),
        (0, 1, 4, 5, 9, 10, 17, 18),
    ]
    table = np.zeros((len(published), 20), dtype=np.uint8)
    for row, support in zip(table, published):
        row[list(support)] = 1
    h_x, h_z = helix_code("c4")
    assert (h_x == h_z).all()
    for checks in (h_x, h_z):
        assert rank(np.vstack([checks, table])) == rank(checks) == rank(table)

    for inner, (n, k) in {"c4": (20, 2), "carbon": (60, 2), "c4-helix": (100, 2)}.items():
        h_x, h_z = helix_code(inner)
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        code = CSSCode(h_x, h_z)
        assert (code.n, code.k) == (n, k)
    with pytest.raises(ValueError):
        helix_code("c6")


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


def test_multi_agent_bicycle_codes_match_published_weight_classes() -> None:
    # arXiv:2608.08996 Table 1: overall weight w = max(stabilizer weight,
    # qubit degree); the three abelian instances have w = 7, 10, 10.
    for name, weight in [
        ("bb288-2608.08996", 7),
        ("bb234-2608.08996", 10),
        ("bb372-2608.08996", 10),
    ]:
        entry = REGISTRY[name]
        h_x, h_z = entry.build()
        code = CSSCode(h_x, h_z)
        assert (code.n, code.k) == (entry.n, entry.k)
        stabilizer = max(int(h_x.sum(axis=1).max()), int(h_z.sum(axis=1).max()))
        degree = int((h_x.sum(axis=0) + h_z.sum(axis=0)).max())
        assert max(stabilizer, degree) == weight


def test_apm_kasai_reproduces_published_parameters() -> None:
    from qec_transversal.codes import apm_kasai

    h_x, h_z = apm_kasai(
        96,
        [(5, 41), (85, 77), (73, 66), (1, 0), (1, 72), (37, 9)],
        [(61, 15), (1, 24), (89, 62), (25, 22), (85, 93), (25, 78)],
    )
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (1152, 580)
    # Every check row sums twelve permutation-matrix rows: weight 12.
    assert set(h_x.sum(axis=1)) == {12} and set(h_z.sum(axis=1)) == {12}


def test_gala_abelian_reproduces_published_parameters() -> None:
    """arXiv:2608.07431 Tables S3/S5: two rows beyond the registry's three.

    Every abelian row of those tables must come out at its published
    ``[[n, k]]`` and at stabilizer weight 12, so the block-circulant index
    convention of Definition 11 is pinned by more than the registry entries.
    """

    from qec_transversal.codes import gala_abelian

    rows = {
        # [[136,36,8]], L = 8, J = 3 over C_17 (Table S5).
        (136, 36): ([17], 8, 3,
                    [[[7]], [[0], [5]], [[6]], [[1], [2]]],
                    [[[10]], [[16], [15]], [[11]], [[0], [12]]]),
        # [[168,42,12]], L = 8, J = 3 over C_3 x C_7 (Table S5).
        (168, 42): ([3, 7], 8, 3,
                    [[[0, 1]], [[0, 3], [0, 2]], [[2, 4], [0, 4]], [[1, 3]]],
                    [[[0, 6]], [[2, 4]], [[1, 3], [0, 3]], [[0, 4], [0, 5]]]),
    }
    for (n, k), (moduli, rungs, active, f_terms, g_terms) in rows.items():
        h_x, h_z = gala_abelian(moduli, rungs, active, f_terms, g_terms)
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        code = CSSCode(h_x, h_z)
        assert (code.n, code.k) == (n, k)
        assert set(h_x.sum(axis=1)) == {12} and set(h_z.sum(axis=1)) == {12}


def test_gala_abelian_rejects_malformed_shapes() -> None:
    from qec_transversal.codes import gala_abelian

    with pytest.raises(ValueError):  # L must be even
        gala_abelian([5], 7, 2, [[[0]]] * 3, [[[0]]] * 3)
    with pytest.raises(ValueError):  # J <= L/2
        gala_abelian([5], 8, 5, [[[0]]] * 4, [[[0]]] * 4)
    with pytest.raises(ValueError):  # one exponent per cyclic factor
        gala_abelian([3, 5], 4, 1, [[[0]], [[1]]], [[[0, 0]], [[1, 1]]])


def test_cornucopia_reproduces_published_parameters() -> None:
    from qec_transversal.codes import cornucopia

    h_x, h_z = cornucopia(7, [2, 1, 1, 1, 4, 5], [5, 3, 0, 5, 2, 3])
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (252, 130)
    assert set(h_x.sum(axis=1)) == {12} and set(h_z.sum(axis=1)) == {12}
    # Each data qubit sits in exactly three X and three Z checks.
    assert set(h_x.sum(axis=0)) == {3} and set(h_z.sum(axis=0)) == {3}


def test_qt_local_codes_match_published_parameters() -> None:
    from qec_transversal.codes import qt_local_code
    from qec_transversal.utils.gf2 import rank

    for label, length, dimension in (("633", 6, 3), ("734", 7, 3), ("953", 9, 5)):
        check, generator = qt_local_code(label)
        assert check.shape[1] == length and generator.shape[1] == length
        assert rank(generator) == dimension
        assert rank(check) == length - dimension
        assert not ((generator.astype(np.int64) @ check.T.astype(np.int64)) % 2).any()


def test_quantum_tanner_lift_reproduces_worked_example() -> None:
    """Appendix A.1 of arXiv:2608.12509: the [[756,10,(<=9,<=42)]] instance."""

    from qec_transversal.codes import qt_local_code, quantum_tanner_lift

    multiset_a = [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]],
                  [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]]]
    multiset_b = [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]],
                  [[1, 4, 3, 2], [5, 7]], [[1, 4, 3, 2], [5, 6]],
                  [[1, 2, 3, 4], [6, 7]], [[1, 2, 3, 4], [5, 7]]]
    h_x, h_z = quantum_tanner_lift(
        7, multiset_a, multiset_b, qt_local_code("734"), qt_local_code("953"),
        [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7, 8, 9, 6],
    )
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (756, 10)
    # n = n_A n_B |G| with |C_3 x. C_4| = 12, and the paper's row weights.
    assert h_x.shape == (480, 756) and h_z.shape == (288, 756)
    assert h_x.sum(axis=1).max() == 20 and set(h_z.sum(axis=1)) == {20}


def test_quantum_tanner_lift_rejects_mismatched_local_length() -> None:
    from qec_transversal.codes import qt_local_code, quantum_tanner_lift

    with pytest.raises(ValueError):
        quantum_tanner_lift(
            5, [[], [[1, 2, 3, 4, 5]]], [[], [[1, 2, 3, 4, 5]]],
            qt_local_code("633"), qt_local_code("633"), [1, 2], [1, 2],
        )


def test_cpm_pair_partition_reproduces_published_parameters() -> None:
    from qec_transversal.codes import cpm_pair_partition

    # arXiv:2609.30069 App. D.2, Eqs. (76)-(77): (J, L, P) = (3, 8, 23).
    e_x = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 12, 8, 21, 6, 1, 19, 15], [0, 9, 18, 11, 7, 17, 10, 4]]
    e_z = [[0, 15, 7, 22, 7, 0, 22, 15], [0, 1, 2, 4, 0, 1, 2, 4], [0, 5, 17, 6, 6, 17, 5, 0]]
    h_x, h_z = cpm_pair_partition(23, e_x, e_z)
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (184, 50)
    # The paper's check ranks (67 each) and column/row weights J = 3, L = 8.
    assert rank(h_x) == rank(h_z) == 67
    assert set(h_x.sum(axis=0)) == {3} and set(h_x.sum(axis=1)) == {8}
    # Eq. (78): check (r, s) acts on qubit s - E[r, c] of block c.
    assert h_x[23 + 5, 1 * 23 + (5 - 12) % 23] == 1
    with pytest.raises(ValueError):
        cpm_pair_partition(23, e_x, [[0, 1, 0, 0, 0, 0, 0, 0]] + e_z[1:])


def test_cpm_pair_partition_f4_reproduces_published_parameters() -> None:
    from qec_transversal.codes import cpm_pair_partition_f4

    # arXiv:2609.35601 worked example, Eqs. (22)-(25): (J, L, P) = (3, 8, 20).
    e_x = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 16, 12, 13, 18, 10, 9], [0, 6, 2, 13, 18, 17, 1, 5]]
    e_z = [[0, 9, 18, 17, 18, 0, 17, 9], [0, 18, 3, 5, 0, 18, 3, 5], [0, 1, 17, 13, 13, 17, 1, 0]]
    c_x = [[0, 1, 2, 0, 1, 0, 0, 2], [1, 0, 0, 2, 0, 1, 2, 0], [2, 0, 0, 1, 0, 2, 1, 0]]
    c_z = [[0, 2, 1, 0, 2, 0, 0, 1], [1, 0, 0, 2, 0, 1, 2, 0], [2, 0, 0, 1, 0, 2, 1, 0]]
    h_x, h_z = cpm_pair_partition_f4(20, e_x, e_z, c_x, c_z)
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (320, 80)
    # Sec. 3.4 / Table 2: full check rank 120, every check row of weight 10,
    # 4P columns of weight 3 and 12P of weight 4.
    assert rank(h_x) == rank(h_z) == 120
    assert set(h_x.sum(axis=1)) == set(h_z.sum(axis=1)) == {10}
    assert sorted(h_x.sum(axis=0)).count(3) == 80 and sorted(h_x.sum(axis=0)).count(4) == 240
    with pytest.raises(ValueError):
        cpm_pair_partition_f4(20, e_x, e_z, c_x, [[1] + c_z[0][1:]] + c_z[1:])


def test_two_block_group_algebra_reproduces_published_parameters() -> None:
    from qec_transversal.codes import permutation_group_table, two_block_group_algebra

    # arXiv:2609.36213 Prop. 4.1: [[144,16,12]] over A_4 x C_6 with weight-eight
    # checks, both check ranks 64, and the all-ones vector in both row spaces.
    h_x, h_z = REGISTRY["bbga144-2609.36213"].build()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (144, 16)
    assert rank(h_x) == rank(h_z) == 64
    assert set(h_x.sum(axis=1)) == set(h_z.sum(axis=1)) == {8}
    assert set(h_x.sum(axis=0)) == set(h_z.sum(axis=0)) == {4}
    ones = np.ones((1, 144), dtype=np.uint8)
    assert rank(np.vstack([h_x, ones])) == rank(np.vstack([h_z, ones])) == 64
    # Over the abelian C_6 x C_6 the construction is the ordinary BB code:
    # A = x^3 + y + y^2, B = y^3 + x + x^2 is the [[72,12,6]] of arXiv:2308.07915.
    table, (x, y) = permutation_group_table([[[1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12]]], 12)

    def power(g: int, e: int) -> int:
        h = 0
        for _ in range(e):
            h = table[h][g]
        return h

    h_x, h_z = two_block_group_algebra(table, [power(x, 3), y, power(y, 2)], [power(y, 3), x, power(x, 2)])
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (72, 12)
    # A Latin square that is not a group table is rejected.
    with pytest.raises(ValueError):
        two_block_group_algebra([[(i - j) % 3 for j in range(3)] for i in range(3)], [1], [1])


def test_weighted_shift_c3_code_reproduces_published_parameters() -> None:
    import itertools

    from qec_transversal.codes import weighted_shift_c3_code

    # arXiv:2609.36213 Prop. 5.1: [[18,4,3]], check ranks 7, six rows of weight
    # six and three of weight eight per sector; Eq. (21) gives A and B explicitly.
    h_x, h_z = weighted_shift_c3_code()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (18, 4)
    assert rank(h_x) == rank(h_z) == 7
    assert sorted(h_x.sum(axis=1)) == sorted(h_z.sum(axis=1)) == [6] * 6 + [8] * 3
    c = (1 - np.eye(3, dtype=np.uint8)) % 2
    i, z = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    j = (i + c) % 2
    a = np.block([[c, z, i], [c, c, z], [z, c, c]])
    b = np.block([[z, c, j], [z, z, c], [c, z, z]])
    assert np.array_equal(h_x, np.concatenate([a, b], axis=1))
    assert np.array_equal(h_z, np.concatenate([b.T, a.T], axis=1))
    # Eq. (22): each row space has weight enumerator 1 + 19z^6 + 45z^8 + 42z^10 + 18z^12 + 3z^14.
    for h in (h_x, h_z):
        vectors = {tuple(np.bitwise_xor.reduce(h[list(rows)], axis=0)) if rows else (0,) * 18
                   for r in range(10) for rows in itertools.combinations(range(9), r)}
        enumerator = {}
        for v in vectors:
            enumerator[sum(v)] = enumerator.get(sum(v), 0) + 1
        assert enumerator == {0: 1, 6: 19, 8: 45, 10: 42, 12: 18, 14: 3}


def test_lifted_product_monomial_reproduces_published_parameters() -> None:
    from qec_transversal.codes import lifted_product_monomial

    # arXiv:2609.39874 Sec. SI.1: [[306,52]] over R_9 with both binary check
    # ranks 127 and weight-eight checks.
    h_x, h_z = REGISTRY["lp306-2609.39874"].build()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (306, 52)
    assert rank(h_x) == rank(h_z) == 127
    assert set(h_x.sum(axis=1)) == set(h_z.sum(axis=1)) == {8}
    # Sec. SII.1: the 2 x 4 seed with exponent rows (0,0,0,0), (0,1,3,7) at
    # lift 15 gives [[300,66]].
    h_x, h_z = REGISTRY["lp300-2609.39874"].build()
    assert h_x.shape == h_z.shape == (120, 300)
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (300, 66)
    # Sec. SII.1, Eq. (S56): the 3 x 5 seed over R_15 gives [[510,76]] with
    # both check ranks 217.
    h_x, h_z = lifted_product_monomial(15, [[0, 0, 0, 0, 0], [0, 8, 13, 7, 6], [0, 14, 6, 4, 13]])
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (510, 76)
    assert rank(h_x) == rank(h_z) == 217
    # A 1 x 1 protograph at lift 1 is the trivial [[2,0]] pair of checks.
    h_x, h_z = lifted_product_monomial(1, [[0]])
    assert np.array_equal(h_x, [[1, 1]]) and np.array_equal(h_z, [[1, 1]])
    with pytest.raises(ValueError):
        lifted_product_monomial(9, [[0, 1], [0]])


def test_trivariate_tricycle_reproduces_published_parameters() -> None:
    from qec_transversal.codes import trivariate_tricycle

    # arXiv:2508.08191 Table 1: [[72,6]] at (l, m, p) = (4, 3, 2) and
    # [[180,12]] at (5, 4, 3); weight-9 X checks and weight-6 Z checks.
    for name, n, k in (("tt72-2508.08191", 72, 6), ("tt180-2508.08191", 180, 12)):
        h_x, h_z = REGISTRY[name].build()
        assert h_x.shape == (n // 3, n) and h_z.shape == (n, n)
        assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (n, k)
        assert set(h_x.sum(axis=1)) == {9} and set(h_z.sum(axis=1)) == {6}
    # Table 3: the two-term CCZ instance [[21,3]] at (7, 1, 1), weight-6/4 checks.
    h_x, h_z = REGISTRY["tt21-2508.08191"].build()
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (21, 3)
    assert set(h_x.sum(axis=1)) == {6} and set(h_z.sum(axis=1)) == {4}
    # Table 5: [[432,12]] at (6, 6, 4); meta-checks [A^T | B^T | C^T] kill H_Z.
    a = [(0, 0, 0), (1, 1, 3), (3, 4, 2)]
    b = [(0, 0, 0), (3, 1, 2), (3, 2, 3)]
    c = [(0, 0, 0), (4, 3, 3), (5, 0, 2)]
    h_x, h_z = trivariate_tricycle(6, 6, 4, a, b, c)
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (432, 12)
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    # H_X^T is the meta-check map composed with a block relabelling: the
    # transposed blocks [A^T | B^T | C^T] annihilate H_Z from the left.
    meta = np.hstack([h_x[:, :144].T, h_x[:, 144:288].T, h_x[:, 288:].T])
    assert not ((meta.astype(np.int64) @ h_z.astype(np.int64)) % 2).any()
    # (1, 1, 1) with A = B = C = 1 is the trivial [[3,0]] triple.
    h_x, h_z = trivariate_tricycle(1, 1, 1, [(0, 0, 0)], [(0, 0, 0)], [(0, 0, 0)])
    assert np.array_equal(h_x, [[1, 1, 1]]) and CSSCode(h_x, h_z).k == 0
    with pytest.raises(ValueError):
        trivariate_tricycle(2, 2, 2, [(0, 0)], [(0, 0, 0)], [(0, 0, 0)])
    with pytest.raises(ValueError):
        trivariate_tricycle(2, 2, 2, [(0, 0, 0), (2, 0, 0)], [(0, 0, 0)], [(0, 0, 0)])


def test_depth_one_universal_reproduces_published_parameters() -> None:
    from qec_transversal.codes import depth_one_universal

    # arXiv:2610.06730 Table 1: lift:ghz37 is [[37,1,7]] and lift:width3-59-w8
    # is [[59,1,9]]; both have checks of weight at most eight.
    for label, n in (("ghz37", 37), ("width3-59-w8", 59)):
        h_x, h_z = depth_one_universal(label)
        assert h_x.shape == h_z.shape == ((n - 1) // 2, n)
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (n, 1)
        assert max(h_x.sum(axis=1).max(), h_z.sum(axis=1).max()) == 8
    # App. B.1: ghz37 is the 15-qubit Reed-Muller seed tensored with a kernel
    # of four GHZ triples and ten |+> qubits, pushed through one CNOT from
    # every kernel qubit onto the seed qubit of its set.  In the ancillary
    # file's numbering each seed qubit is followed by its kernel qubits.
    seed = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35]
    kernel = [q for q in range(37) if q not in seed]
    ghz = [(8, 18, 28), (9, 23, 33), (13, 19, 34), (14, 24, 29)]
    plus = [q for q in kernel if not any(q in triple for triple in ghz)]
    seed_x, seed_z = quantum_reed_muller_15()
    x = np.zeros((18, 37), dtype=np.uint8)
    z = np.zeros((18, 37), dtype=np.uint8)
    x[:4, seed], z[:10, seed] = seed_x, seed_z
    for row, triple in enumerate(ghz):
        x[4 + row, list(triple)] = 1
        z[10 + 2 * row, [triple[0], triple[2]]] = 1
        z[11 + 2 * row, [triple[1], triple[2]]] = 1
    for row, qubit in enumerate(plus):
        x[8 + row, qubit] = 1
    for control in kernel:
        target = max(s for s in seed if s < control)
        x[:, target] ^= x[:, control]
        z[:, control] ^= z[:, target]
    h_x, h_z = depth_one_universal("ghz37")
    assert rank(x) == rank(h_x) == rank(np.vstack([x, h_x])) == 18
    assert rank(z) == rank(h_z) == rank(np.vstack([z, h_z])) == 18
    with pytest.raises(ValueError):
        depth_one_universal("ghz38")


def test_cyclic_triorthogonal_reproduces_published_parameters() -> None:
    from qec_transversal.codes import cyclic_triorthogonal

    # arXiv:2610.08012 App. C: spectral supports of the qubit construction-A
    # records [[15,1,3]], [[85,1,5]], [[127,1,7]] and [[223,1,9]].
    supports = {
        15: [1, 2, 4, 8],
        85: [1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 24, 32, 43, 44, 48, 64],
        127: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64,
              65, 66, 68, 72, 80, 96],
        223: [1, 2, 4, 7, 8, 14, 15, 16, 17, 28, 30, 32, 33, 34, 41, 49, 56, 60, 64, 66, 68,
              82, 98, 105, 112, 115, 119, 120, 128, 132, 136, 164, 169, 171, 196, 197, 210],
    }
    for n, support in supports.items():
        h_x, h_z = cyclic_triorthogonal(n, support)
        assert h_x.shape == (len(support), n) and h_z.shape == (n - len(support) - 1, n)
        assert rank(h_x) == len(support) and rank(h_z) == n - len(support) - 1
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        code = CSSCode(h_x, h_z)
        assert (code.n, code.k) == (n, 1)

    def weight_enumerator(rows):
        rows = rows.astype(np.int64)
        coefficients = np.array(
            [[(i >> r) & 1 for r in range(rows.shape[0])] for i in range(1 << rows.shape[0])],
            dtype=np.int64,
        )
        return np.bincount(((coefficients @ rows) % 2).sum(axis=1), minlength=rows.shape[1] + 1)

    # Theorem 16: the X-check code is triply even -- every one of the 2^16
    # codewords at n = 85 has weight divisible by eight.
    h_x, _ = cyclic_triorthogonal(85, supports[85])
    assert not (np.flatnonzero(weight_enumerator(h_x)) % 8).any()
    # S = {1, 2, 4, 8} at n = 15 is the punctured Reed-Muller code: both check
    # codes share their weight enumerators with qrm15 (the [15,4,8] simplex
    # code and the [15,10,4] even-weight subcode of the Hamming code).
    h_x, h_z = cyclic_triorthogonal(15, supports[15])
    q_x, q_z = quantum_reed_muller_15()
    assert weight_enumerator(h_x).tolist() == weight_enumerator(q_x).tolist()
    assert weight_enumerator(h_z).tolist() == weight_enumerator(q_z).tolist()
    with pytest.raises(ValueError):
        cyclic_triorthogonal(15, [1, 2, 4])  # not a union of cyclotomic cosets
    with pytest.raises(ValueError):
        cyclic_triorthogonal(15, [1, 2, 4, 8, 3, 6, 12, 9])  # 3 + 4 + 8 = 15: not triply even
    with pytest.raises(ValueError):
        cyclic_triorthogonal(16, [1, 2, 4, 8])  # even length


def test_projective_ccz_48_reproduces_published_parameters() -> None:
    from qec_transversal.codes import projective_ccz_48
    from qec_transversal.utils.gf2 import nullspace

    # arXiv:2610.09341 Sec. 3: Q_48 is the FG48 check matrix of Eq. (7) with
    # the Table 2 logical rows; the paper states d_X = 16 and d_Z = 3.
    h_x, h_z = projective_ccz_48()
    assert h_x.shape == (6, 48) and h_z.shape == (39, 48)
    assert rank(h_x) == 6 and rank(h_z) == 39
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    code = CSSCode(h_x, h_z)
    assert (code.n, code.k) == (48, 3)

    def codewords(rows):
        rows = rows.astype(np.int64)
        coefficients = np.array(
            [[(i >> r) & 1 for r in range(rows.shape[0])] for i in range(1 << rows.shape[0])],
            dtype=np.int64,
        )
        return (coefficients @ rows) % 2

    # Prop. 2.3: FG48 is two-weight and 8-divisible -- 60 words of weight 24, 3 of weight 32.
    weights = np.bincount(codewords(h_x).sum(axis=1), minlength=49)
    assert weights[24] == 60 and weights[32] == 3 and weights.sum() == 64
    # d_X = 16: C_1 = ker H_Z has 512 words and its lightest word outside C_2 = rowspan H_X
    # weighs 16.  d_Z = 3: the columns of H_X are distinct and nonzero (Lemma 2.1), and the
    # three points with u = 0 are collinear, so they support a weight-3 Z logical.
    c_1 = codewords(nullspace(h_z))
    outside = np.array([rank(np.vstack([h_x, v])) == 7 for v in c_1])
    assert c_1[outside].sum(axis=1).min() == 16
    columns = {tuple(c) for c in h_x.T}
    assert len(columns) == 48 and (0,) * 6 not in columns
    v = np.zeros(48, dtype=np.uint8)
    v[[0, 16, 32]] = 1
    assert not ((h_x.astype(np.int64) @ v) % 2).any() and rank(np.vstack([h_z, v])) == 40

    # Prop. 3.9: the flip vector gamma(u, w) = u3 + w1 Q1(u) + w2 Q2(u) has weight 22, and
    # its T/T-dagger layer has signed weight 4 x1 x2 x3 (mod 8) on every codeword of C_1:
    # exactly the 64 words of one coset of C_2 give 4, all others 0 -- logical CCZ.
    def gamma(t, w):
        u1, u2, u3, u4 = ((t >> i) & 1 for i in range(4))
        q1 = u1 + u2 + u3 + u1 * u3 + u2 * u3 + u2 * u4
        q2 = u3 + u4 + u1 * u2 + u1 * u4 + u3 * u4
        return (u3 + w[0] * q1 + w[1] * q2) % 2

    flips = np.array([gamma(t, w) for w in ((1, 0), (0, 1), (1, 1)) for t in range(16)])
    assert flips.sum() == 22
    signed = (c_1 @ (1 - 2 * flips)) % 8
    assert set(signed.tolist()) == {0, 4} and int((signed == 4).sum()) == 64


def test_quadcycle_reproduces_published_parameters() -> None:
    from qec_transversal.codes import quadcycle, trivariate_tricycle

    # arXiv:2610.10731 Table 2: [[54,6]] over Z_3 x Z_3, [[72,6]] over
    # Z_3 x Z_4 and [[162,6]] over Z_3 x Z_9; 4|G| checks of each type.
    for name, n, k, weights in (
        ("quad54-2610.10731", 54, 6, {6}),
        ("quad72-2610.10731", 72, 6, {6, 8}),
        ("quad162-2610.10731", 162, 6, {6, 8}),
    ):
        h_x, h_z = REGISTRY[name].build()
        assert h_x.shape == h_z.shape == (2 * n // 3, n)
        assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
        assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (n, k)
        assert set(h_x.sum(axis=1)) == weights and set(h_z.sum(axis=1)) == weights
    # The [[228,6]] row over Z_2 x Z_19 and the trivariate [[420,6]] row over
    # Z_2 x Z_5 x Z_7.
    a, b, c = [(1, 0), (0, 2)], [(1, 0), (0, 3)], [(1, 0), (0, 5)]
    h_x, h_z = quadcycle((2, 19), a, b, c, [(0, 0), (1, 1), (0, 2), (0, 9)])
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (228, 6)
    a, b, c = [(0, 0, 0), (0, 1, 1)], [(1, 0, 0), (0, 0, 2)], [(1, 1, 0), (0, 0, 3)]
    d = [(0, 0, 0), (1, 1, 0), (1, 2, 0), (0, 3, 1)]
    h_x, h_z = quadcycle((2, 5, 7), a, b, c, d)
    assert (CSSCode(h_x, h_z).n, CSSCode(h_x, h_z).k) == (420, 6)
    assert not ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any()
    # Methods: the meta-checks [A^T | B^T | C^T | D^T] and [D | B | A | C]
    # annihilate H_X and H_Z from the left; the factors a, b, c alone give
    # the paired [[210,3]] tricycle code, so k_4D = 2 k_3D.
    size = 70
    block = lambda h, i, j: h[i * size : (i + 1) * size, j * size : (j + 1) * size]
    a_t, b_t, c_t, d_t = block(h_x, 2, 0), block(h_x, 2, 1), block(h_x, 0, 0), block(h_x, 0, 3)
    meta_x = np.hstack([a_t, b_t, c_t, d_t])
    meta_z = np.hstack([d_t.T, b_t.T, a_t.T, c_t.T])
    assert not ((meta_x.astype(np.int64) @ h_x.astype(np.int64)) % 2).any()
    assert not ((meta_z.astype(np.int64) @ h_z.astype(np.int64)) % 2).any()
    tricycle = CSSCode(*trivariate_tricycle(2, 5, 7, a, b, c))
    assert (tricycle.n, tricycle.k) == (210, 3)
    # One cyclic factor of order 1 with a = b = c = d = 1 is the trivial [[6,0]] code.
    h_x, h_z = quadcycle((1,), [(0,)], [(0,)], [(0,)], [(0,)])
    assert h_x.shape == (4, 6) and CSSCode(h_x, h_z).k == 0
    with pytest.raises(ValueError):
        quadcycle((2, 2), [(0,)], [(0, 0)], [(0, 0)], [(0, 0)])
    with pytest.raises(ValueError):
        quadcycle((2, 2), [(0, 0), (2, 0)], [(0, 0)], [(0, 0)], [(0, 0)])
