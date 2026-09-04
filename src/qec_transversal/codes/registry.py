"""The named-code registry.

``REGISTRY`` maps a short name to a :class:`NamedCode` record carrying the
published ``[[n, k, d]]`` parameters, the source reference, and a builder
returning ``(H_X, H_Z)``.  It is the input set for the census in ``docs/zoo``
and for most of the test suite; it is data, not algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..utils.gf2 import BinaryMatrix
from .families import (
    apm_kasai,
    bipartite_grid,
    bivariate_bicycle,
    cornucopia,
    doubled_color_41,
    gala_abelian,
    generalized_bicycle,
    hamming_7_4,
    helix_code,
    helper_qss_css,
    hypergraph_product,
    iceberg,
    kasai_binary_pair,
    kasai_nonbinary,
    la_cross,
    lifted_product_b1,
    middle_reed_muller,
    qt_local_code,
    quantum_reed_muller_15,
    quantum_reed_muller_31,
    quantum_tanner_lift,
    self_dual_bicycle,
    steane_code,
    subset_inclusion,
    surface_code,
    toric_code,
)


@dataclass(frozen=True)
class NamedCode:
    """A registry entry: a constructor plus its published parameters."""

    name: str
    family: str
    build: Callable[[], tuple[BinaryMatrix, BinaryMatrix]]
    n: int
    k: int
    d: int | None = None
    d_is_upper_bound: bool = False
    source: str = ""


def _bb(l: int, m: int, a: list, b: list) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: bivariate_bicycle(l, m, a, b)


def _sdbb(
    monomials: list, basis_1: Sequence[int], basis_2: Sequence[int]
) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: self_dual_bicycle(monomials, basis_1, basis_2)


def _qt(
    degree: int,
    multiset_a: list,
    multiset_b: list,
    label_a: str,
    label_b: str,
    pi_a: Sequence[int],
    pi_b: Sequence[int],
) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: quantum_tanner_lift(
        degree,
        multiset_a,
        multiset_b,
        qt_local_code(label_a),
        qt_local_code(label_b),
        pi_a,
        pi_b,
    )


REGISTRY: dict[str, NamedCode] = {
    code.name: code
    for code in [
        # Positive controls: codes with known nontrivial strict-transversal gates.
        NamedCode("steane", "color", steane_code, 7, 1, 3, source="self-dual doubly-even"),
        NamedCode(
            "c4-22", "small", lambda: (np.ones((1, 4), np.uint8), np.ones((1, 4), np.uint8)),
            4, 2, 2, source="[[4,2,2]] iceberg code",
        ),
        NamedCode(
            "c6-22", "small",
            lambda: (
                np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]], np.uint8),
                np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]], np.uint8),
            ),
            6, 2, 2, source="Albert running example",
        ),
        NamedCode(
            "cube-832", "small",
            lambda: (
                np.ones((1, 8), np.uint8),
                np.asarray(
                    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]],
                    np.uint8,
                ),
            ),
            8, 3, 2, source="[[8,3,2]] cube code, transversal CCZ",
        ),
        NamedCode("iceberg-8", "iceberg", lambda: iceberg(4), 8, 6, 2, source="[[2m,2m-2,2]] error-detection family"),
        NamedCode("iceberg-12", "iceberg", lambda: iceberg(6), 12, 10, 2, source="[[2m,2m-2,2]] error-detection family"),
        NamedCode("qrm15", "reed-muller", quantum_reed_muller_15, 15, 1, 3, source="arXiv:1403.2734"),
        NamedCode("qrm31", "reed-muller", quantum_reed_muller_31, 31, 1, 3, source="punctured RM(1,5); level-4 transversal gate"),
        NamedCode("tesseract", "reed-muller", lambda: middle_reed_muller(4), 16, 6, 4, source="arXiv:2608.05688"),
        NamedCode("rm64", "reed-muller", lambda: middle_reed_muller(6), 64, 20, 8, source="arXiv:2608.05688"),
        NamedCode("rm256", "reed-muller", lambda: middle_reed_muller(8), 256, 70, 16, source="arXiv:2608.05688"),
        NamedCode("grid-4x6", "grid", lambda: bipartite_grid(4, 6), 24, 8, 4, source="arXiv:2608.05688"),
        NamedCode("grid-6x8", "grid", lambda: bipartite_grid(6, 8), 48, 24, 4, source="arXiv:2608.05688"),
        # Topological negative controls.
        NamedCode("toric-4", "toric", lambda: toric_code(4), 32, 2, 4),
        NamedCode("toric-10", "toric", lambda: toric_code(10), 200, 2, 10),
        NamedCode("surface-5", "surface", lambda: surface_code(5), 41, 1, 5),
        # Bivariate bicycle codes, Bravyi et al. arXiv:2308.07915 Table 3.
        NamedCode("bb72", "bivariate-bicycle", _bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 72, 12, 6, source="arXiv:2308.07915"),
        NamedCode("bb90", "bivariate-bicycle", _bb(15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)]), 90, 8, 10, source="arXiv:2308.07915"),
        NamedCode("bb108", "bivariate-bicycle", _bb(9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 108, 8, 10, source="arXiv:2308.07915"),
        NamedCode("gross", "bivariate-bicycle", _bb(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 144, 12, 12, source="arXiv:2308.07915"),
        NamedCode("two-gross", "bivariate-bicycle", _bb(12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]), 288, 12, 18, source="arXiv:2308.07915"),
        NamedCode("bb360", "bivariate-bicycle", _bb(30, 6, [(9, 0), (0, 1), (0, 2)], [(0, 3), (25, 0), (26, 0)]), 360, 12, 24, d_is_upper_bound=True, source="arXiv:2308.07915"),
        NamedCode("bb756", "bivariate-bicycle", _bb(21, 18, [(3, 0), (0, 10), (0, 17)], [(0, 5), (3, 0), (19, 0)]), 756, 16, 34, d_is_upper_bound=True, source="arXiv:2308.07915"),
        NamedCode("bb54", "bivariate-bicycle", _bb(3, 9, [(0, 0), (0, 2), (0, 4)], [(0, 3), (1, 0), (2, 0)]), 54, 8, 6, source="arXiv:2408.10001"),
        # Symmetric BB codes with rich fold-transversal groups.
        NamedCode("bb98-symmetric", "bivariate-bicycle", _bb(7, 7, [(1, 0), (0, 3), (0, 4)], [(0, 1), (3, 0), (4, 0)]), 98, 6, 12, source="arXiv:2407.03973"),
        NamedCode("bb162-symmetric", "bivariate-bicycle", _bb(9, 9, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 162, 8, 12, source="arXiv:2407.03973"),
        # Coprime bivariate bicycle codes, Wang-Mueller arXiv:2408.10001.
        NamedCode("coprime30", "coprime-bb", _bb(3, 5, [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 2), (1, 2)]), 30, 4, 6, source="arXiv:2408.10001"),
        NamedCode("coprime42", "coprime-bb", _bb(3, 7, [(0, 0), (2, 2), (0, 3)], [(0, 0), (2, 2), (1, 3)]), 42, 6, 6, source="arXiv:2408.10001"),
        NamedCode("coprime70", "coprime-bb", _bb(5, 7, [(0, 0), (1, 1), (0, 5)], [(0, 0), (1, 1), (2, 5)]), 70, 6, 8, source="arXiv:2408.10001"),
        NamedCode("coprime126", "coprime-bb", _bb(7, 9, [(0, 0), (1, 1), (2, 4)], [(0, 0), (6, 4), (6, 5)]), 126, 12, 10, source="arXiv:2408.10001"),
        NamedCode("coprime154", "coprime-bb", _bb(7, 11, [(0, 0), (1, 1), (3, 9)], [(0, 0), (5, 8), (4, 9)]), 154, 6, 16, source="arXiv:2408.10001"),
        # Trivariate bicycle, arXiv:2406.19151 (weight-5 checks).
        NamedCode("trivariate30", "trivariate-bicycle", _bb(3, 5, [(1, 0), (1, 4)], [(1, 0), (0, 2), (2, 2)]), 30, 4, 5, source="arXiv:2406.19151"),
        # Generalized bicycle codes, Panteleev-Kalachev arXiv:1904.02703 App. B.
        NamedCode("gb48", "generalized-bicycle", lambda: generalized_bicycle(24, [0, 2, 8, 15], [0, 2, 12, 17]), 48, 6, 8, source="arXiv:1904.02703 A3"),
        NamedCode("gb46", "generalized-bicycle", lambda: generalized_bicycle(23, [0, 5, 8, 12], [0, 1, 5, 7]), 46, 2, 9, source="arXiv:1904.02703 A4"),
        NamedCode("gb126", "generalized-bicycle", lambda: generalized_bicycle(63, [0, 1, 14, 16, 22], [0, 3, 13, 20, 42]), 126, 28, 8, source="arXiv:1904.02703 A2"),
        NamedCode("gb66-2608.09115", "generalized-bicycle", lambda: generalized_bicycle(33, [1, 2, 3, 5, 10, 27, 32], [0, 1, 3, 4, 5, 7, 13, 15, 22, 24, 30, 32]), 66, 20, 7, source="arXiv:2608.09115 Table I"),
        NamedCode("gb46-2608.09115", "generalized-bicycle", lambda: generalized_bicycle(23, [0, 1, 12, 13, 17, 19], [0, 1, 4, 5, 8, 9, 12, 13]), 46, 2, 8, source="arXiv:2608.09115 Table I"),
        # Multi-agent search bicycle codes, Qian-Li arXiv:2608.08996 (SI construction data).
        # Abelian coset-orbit balanced products: bb288 is the paper's [[288,16,18]] over
        # Z12 x Z48 with normal K = <y^12>, reduced (as the paper states) to its lifted
        # product over Z12 x Z12; the other two have K = {e} and are two-block codes as given.
        NamedCode("bb288-2608.08996", "bivariate-bicycle", _bb(12, 12, [(1, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0), (5, 9)]), 288, 16, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("bb234-2608.08996", "bivariate-bicycle", _bb(13, 9, [(0, 0), (0, 2), (0, 8), (4, 4), (6, 8)], [(2, 8), (5, 4), (10, 2), (11, 5), (12, 0)]), 234, 28, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("bb372-2608.08996", "bivariate-bicycle", _bb(31, 6, [(5, 1), (5, 3), (7, 2), (18, 2), (30, 2)], [(10, 1), (10, 3), (21, 5), (24, 5), (26, 5)]), 372, 44, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("wzl20-2608.10912", "subset-inclusion", lambda: subset_inclusion(6, 3, 2), 20, 8, 4, source="arXiv:2608.10912 Sec. IV"),
        NamedCode("wzl120-2608.10912", "subset-inclusion", lambda: subset_inclusion(10, 3, 2), 120, 100, 4, source="arXiv:2608.10912 Sec. IV"),
        NamedCode("doubled41-2608.11160", "doubled", doubled_color_41, 41, 1, 9, source="arXiv:2608.11160 Ex. III.5"),
        # Blind-helper QSS codes, arXiv:2609.00220 Ex. 6: the explicit
        # [[2m+1, 1]] CSS family on odd m parties.  m = 3 is the Steane code
        # (the paper says so, and the rebuild confirms it), so the registry
        # takes the next three members.  The paper publishes no distance.
        NamedCode("helper11-2609.00220", "helper-qss", lambda: helper_qss_css(5), 11, 1, None, source="arXiv:2609.00220 Ex. 6"),
        NamedCode("helper15-2609.00220", "helper-qss", lambda: helper_qss_css(7), 15, 1, None, source="arXiv:2609.00220 Ex. 6"),
        NamedCode("helper19-2609.00220", "helper-qss", lambda: helper_qss_css(9), 19, 1, None, source="arXiv:2609.00220 Ex. 6"),
        # Self-dual bivariate bicycle codes: sparse (weight-8, doubly even) codes
        # that DO carry strict transversal H and S -- the registry's positive LDPC
        # control against reading the qLDPC census as a statement about sparsity.
        # f(x, y) per Eq. (14) on the twisted torus of Tables I-II; distances are
        # the paper's exact integer-programming values.
        NamedCode("sdbb16-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 4), (2, 2)), 16, 4, 4, source="arXiv:2510.05211 Table I"),
        NamedCode("sdbb64-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 8), (4, 4)), 64, 8, 8, source="arXiv:2510.05211 Table I, Fig. 1"),
        NamedCode("sdbb152-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (2, 1), (-1, 1)], (0, 19), (4, 6)), 152, 6, 16, source="arXiv:2510.05211 Table II"),
        NamedCode("sdbb160-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (2, 2), (-1, 1)], (0, 10), (8, 0)), 160, 8, 16, source="arXiv:2510.05211 Table II"),
        NamedCode("apm1152-2604.16209", "apm-kasai", lambda: apm_kasai(96, [(5, 41), (85, 77), (73, 66), (1, 0), (1, 72), (37, 9)], [(61, 15), (1, 24), (89, 62), (25, 22), (85, 93), (25, 78)]), 1152, 580, 12, d_is_upper_bound=True, source="arXiv:2604.16209 Table A1"),
        NamedCode("apm2304-2604.16209", "apm-kasai", lambda: apm_kasai(192, [(71, 127), (97, 80), (67, 117), (163, 165), (25, 60), (187, 33)], [(163, 165), (55, 183), (167, 79), (139, 41), (109, 78), (31, 27)]), 2304, 1156, 14, d_is_upper_bound=True, source="arXiv:2604.16209 Table A1"),
        NamedCode("cornucopia252-2608.02773", "cornucopia", lambda: cornucopia(7, [2, 1, 1, 1, 4, 5], [5, 3, 0, 5, 2, 3]), 252, 130, 6, source="arXiv:2608.02773 Ext. Tab. 1"),
        NamedCode("cornucopia1044-2608.02773", "cornucopia", lambda: cornucopia(29, [2, 22, 20, 22, 18, 6], [27, 11, 12, 18, 21, 26]), 1044, 526, 12, source="arXiv:2608.02773 Ext. Tab. 1"),
        NamedCode("cornucopia2844-2608.02773", "cornucopia", lambda: cornucopia(79, [6, 49, 55, 18, 40, 7], [24, 41, 78, 53, 68, 21]), 2844, 1426, 18, source="arXiv:2608.02773 Ext. Tab. 1"),
        # Lifted quantum Tanner codes, Mian et al. arXiv:2608.12509.  Only the
        # instances whose two local codes are among the three the paper prints
        # in full are rebuildable; the [8,4,4] and [7,4,3] rows are named by
        # parameters only, so their column order -- which the lift depends on --
        # is not published.  d is min(d_X, d_Z) of the paper's sQetch bounds.
        # qt504 and qt756 additionally reproduce the rank and row-weight
        # multiset of the authors' own H_X, H_Z (Zenodo 10.5281/zenodo.21904804).
        NamedCode("qt720-2608.12509", "quantum-tanner", _qt(5, [[], [[2, 4, 5, 3]], [[1, 5, 4, 3, 2]], [[1, 5, 2, 3]], [[1, 3, 4, 2]], [[1, 2, 3, 4, 5]]], [[], [[2, 4, 5, 3]], [[1, 5, 4, 3, 2]], [[1, 5, 2, 3]], [[1, 3, 4, 2]], [[1, 2, 3, 4, 5]]], "633", "633", [1, 2, 3, 4, 5, 6], [1, 2, 5, 4, 6, 3]), 720, 6, 30, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 3, 10, 11"),
        NamedCode("qt504-2608.12509", "quantum-tanner", _qt(7, [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]]], [[], [[1, 3], [2, 4], [5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]], [[1, 4, 3, 2], [5, 6]], [[1, 2, 3, 4], [6, 7]]], "734", "633", [1, 2, 3, 4, 5, 6, 7], [1, 2, 6, 3, 4, 5]), 504, 4, 12, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 1, 9, 12"),
        NamedCode("qt756-2608.12509", "quantum-tanner", _qt(7, [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]]], [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]], [[1, 4, 3, 2], [5, 6]], [[1, 2, 3, 4], [6, 7]], [[1, 2, 3, 4], [5, 7]]], "734", "953", [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7, 8, 9, 6]), 756, 10, 9, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 1, 9, 12"),
        # GALA codes, arXiv:2608.07431.  Only the abelian-lift instances are
        # rebuilt: the paper's non-abelian rows name their S_3 / S_4 elements
        # by sigma/tau labels and word syntax whose permutation realization is
        # not printed alongside the tables.  Distances are exact -- the paper
        # certifies every one by exhaustive exclusion plus a weight-d witness.
        NamedCode("gala132-2608.07431", "gala", lambda: gala_abelian([11], 12, 5, [[[2]], [[4]], [[3]], [[6]], [[3]], [[9]]], [[[9]], [[2]], [[8]], [[5]], [[8]], [[7]]]), 132, 30, 12, source="arXiv:2608.07431 Table S5"),
        NamedCode("gala136-2608.07431", "gala", lambda: gala_abelian([17], 8, 3, [[[2]], [[1]], [[3], [16]], [[13], [12]]], [[[15]], [[4], [5]], [[14], [1]], [[16]]]), 136, 34, 12, source="arXiv:2608.07431 Table S5"),
        NamedCode("gala672-2608.07431", "gala", lambda: gala_abelian([2, 3, 14], 8, 2, [[[1, 1, 5]], [[0, 2, 7], [0, 0, 10]], [[1, 2, 5]], [[1, 0, 6], [0, 0, 8]]], [[[0, 1, 13]], [[1, 0, 10], [0, 2, 10]], [[0, 0, 6]], [[1, 0, 8], [0, 1, 12]]]), 672, 336, 12, source="arXiv:2608.07431 Table S3"),
        # Concatenated symplectic double ("Helix") codes, arXiv:2609.03194
        # App. B: the [[10,2,3]] twisted toric code concatenated along its
        # ZX-duality with a larger k = 2 inner code than the [[4,2,2]] of the
        # published [[20,2,6]] C4-Helix (arXiv:2510.18753).  Distances are the
        # paper's 3 x d_inner; the rebuilt [[20,2,6]] reproduces its Table I
        # stabilizer group exactly, which is what fixes the lift.
        NamedCode("helix60-2609.03194", "csd-helix", lambda: helix_code("carbon"), 60, 2, 12, source="arXiv:2609.03194 App. B"),
        NamedCode("helix100-2609.03194", "csd-helix", lambda: helix_code("c4-helix"), 100, 2, 18, source="arXiv:2609.03194 App. B"),
        # Hypergraph and lifted products.
        NamedCode("hgp-hamming", "hypergraph-product", lambda: hypergraph_product(hamming_7_4(), hamming_7_4()), 58, 16, 3, source="arXiv:0903.0566"),
        NamedCode("lacross65", "la-cross", lambda: la_cross(7, 3), 65, 9, 4, source="arXiv:2404.13010"),
        NamedCode("lacross400", "la-cross", lambda: la_cross(16, 4), 400, 16, 8, source="arXiv:2404.13010"),
        NamedCode("lifted-b1", "lifted-product", lifted_product_b1, 882, 24, 24, d_is_upper_bound=True, source="arXiv:1904.02703 B1"),
        # Kasai-style quasi-cyclic CSS codes.
        NamedCode("kasai-binary-294", "kasai", lambda: kasai_binary_pair(6, 49), 294, 100, None, source="arXiv:2501.13444"),
        NamedCode("kasai-binary-1104", "kasai", lambda: kasai_binary_pair(8, 138), 1104, 554, None, source="arXiv:2501.13444"),
        NamedCode("kasai-gf256-2352", "kasai", lambda: kasai_nonbinary(6, 49), 2352, 800, None, source="arXiv:2412.21171 (CSA labels)"),
    ]
}
