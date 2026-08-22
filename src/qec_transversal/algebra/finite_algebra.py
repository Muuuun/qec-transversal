"""Finite-dimensional unital algebras over ``F_2`` in coordinates."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..utils.gf2 import rank, row_basis
from ..utils.polynomials import _solve_coords


class AlgebraF2:
    """A finite-dimensional unital algebra over F_2 in coordinates.

    ``basis``: rows spanning the algebra as flat GF(2) vectors.
    ``multiply``: bilinear product on flat vectors.
    ``one``: the unit element (must lie in the span).
    """

    def __init__(self, basis: np.ndarray, multiply: Callable, one: np.ndarray):
        self.basis = row_basis(basis.astype(np.uint8))
        self.multiply = multiply
        self.dim = self.basis.shape[0]
        one_coords = _solve_coords(self.basis, one.astype(np.uint8))
        if one_coords is None:
            raise ValueError("unit element is not in the algebra span")
        self.one_coords = one_coords
        # structure constants via left-multiplication matrices L_i (d x d)
        self.left = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint8)
        self.right = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint8)
        for i in range(self.dim):
            for j in range(self.dim):
                product = multiply(self.basis[i], self.basis[j]) % 2
                coords = _solve_coords(self.basis, product)
                if coords is None:
                    raise ValueError("basis is not multiplicatively closed")
                self.left[i, :, j] = 0  # filled below
                for k in range(self.dim):
                    if coords[k]:
                        self.left[i, k, j] = 1  # (L_i)_{kj}: e_i e_j = sum_k c e_k
                        self.right[j, k, i] = 1  # (R_j)_{ki}: e_i e_j
        # rows of self.left[i] act on coordinate columns: L_i @ coords(x) = coords(e_i x)

    def coords_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Product in coordinates."""

        out = np.zeros(self.dim, dtype=np.uint8)
        for i in np.flatnonzero(a):
            out ^= (self.left[i] @ b % 2).astype(np.uint8)
        return out

    def left_matrix(self, a: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.dim, self.dim), dtype=np.uint8)
        for i in np.flatnonzero(a):
            matrix ^= self.left[i]
        return matrix

    def right_matrix(self, a: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.dim, self.dim), dtype=np.uint8)
        for j in np.flatnonzero(a):
            matrix ^= self.right[j]
        return matrix

    def is_unit(self, a: np.ndarray) -> bool:
        return rank(self.left_matrix(a)) == self.dim

    def basis_coords(self, index: int) -> np.ndarray:
        """The ``index``-th basis vector in algebra coordinates."""

        vector = np.zeros(self.dim, dtype=np.uint8)
        vector[index] = 1
        return vector
