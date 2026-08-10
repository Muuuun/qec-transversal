import numpy as np

from qec_transversal.gf2 import gf2_inverse, nullspace, quotient_complement, rank, rref


def test_rref_and_nullspace() -> None:
    matrix = np.asarray([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=np.uint8)
    reduced, pivots = rref(matrix)
    kernel = nullspace(matrix)

    assert len(pivots) == rank(matrix) == 2
    assert reduced.shape == (2, 4)
    assert kernel.shape == (2, 4)
    assert not ((matrix @ kernel.T) & 1).any()


def test_inverse() -> None:
    matrix = np.asarray([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    inverse = gf2_inverse(matrix)
    assert np.array_equal((matrix @ inverse) & 1, np.eye(3, dtype=np.uint8))


def test_quotient_complement() -> None:
    ambient = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    subspace = np.asarray([[1, 1, 0]], dtype=np.uint8)
    complement = quotient_complement(ambient, subspace)
    assert complement.shape == (2, 3)
    assert rank(np.vstack([subspace, complement])) == 3

