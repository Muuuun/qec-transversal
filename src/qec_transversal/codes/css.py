"""The CSS code object.

A CSS code is the special case ``S = rowspan([H_X | 0]) + rowspan([0 | H_Z])``
with ``H_X H_Z^T = 0``.  Storing the two check families separately is what
makes the specialised CSS solvers (strict shear families, fixed-matching fold
layers, the coset-phase diagonal ladder) cheaper than the general
preservation-algebra route -- see ``docs/mathematics.md``.

This module owns the *representation* only.  ``analyze_transversal`` is kept
as a convenience method for backwards compatibility and delegates to
:mod:`qec_transversal.ansatz.strict_css`.
"""

from __future__ import annotations

import numpy as np

from ..utils.gf2 import (
    as_binary_matrix,
    gf2_inverse,
    nullspace,
    quotient_complement,
    rank,
    row_basis,
    rref,
)


def _infer_width(matrix: object) -> int | None:
    array = np.asarray(matrix)
    if array.ndim == 2:
        return int(array.shape[1])
    if array.ndim == 1 and array.size:
        return int(array.shape[0])
    return None


class CSSCode:
    """A binary CSS stabilizer code specified by X- and Z-check rows."""

    def __init__(self, h_x: object, h_z: object, *, n: int | None = None):
        inferred_x = _infer_width(h_x)
        inferred_z = _infer_width(h_z)
        widths = {width for width in (inferred_x, inferred_z, n) if width is not None}
        if not widths:
            raise ValueError("cannot infer n from two empty check matrices")
        if len(widths) != 1:
            raise ValueError(f"inconsistent physical-qubit counts: {sorted(widths)}")
        self.n = widths.pop()
        if self.n <= 0:
            raise ValueError("n must be positive")

        self.h_x = as_binary_matrix(h_x, ncols=self.n)
        self.h_z = as_binary_matrix(h_z, ncols=self.n)
        if ((self.h_x @ self.h_z.T) & 1).any():
            raise ValueError("invalid CSS checks: H_X H_Z^T is nonzero over GF(2)")

        self.c_x = row_basis(self.h_x, ncols=self.n)
        self.c_z = row_basis(self.h_z, ncols=self.n)
        self.rank_x = rank(self.c_x)
        self.rank_z = rank(self.c_z)
        self.k = self.n - self.rank_x - self.rank_z
        if self.k < 0:
            raise ValueError("check ranks exceed n")

        normalizer_x = nullspace(self.h_z)
        normalizer_z = nullspace(self.h_x)
        self.logical_x = quotient_complement(normalizer_x, self.c_x)
        logical_z_unpaired = quotient_complement(normalizer_z, self.c_z)
        if self.logical_x.shape[0] != self.k or logical_z_unpaired.shape[0] != self.k:
            raise ValueError("failed to construct the expected number of logical operators")

        pairing = (self.logical_x @ logical_z_unpaired.T) & 1
        self.logical_z = (gf2_inverse(pairing).T @ logical_z_unpaired) & 1
        if not np.array_equal(
            (self.logical_x @ self.logical_z.T) & 1,
            np.eye(self.k, dtype=np.uint8),
        ):
            raise AssertionError("internal error: logical basis is not symplectically paired")

        zeros_x = np.zeros((self.c_x.shape[0], self.n), dtype=np.uint8)
        zeros_z = np.zeros((self.c_z.shape[0], self.n), dtype=np.uint8)
        self.stabilizer = np.vstack(
            [
                np.hstack([self.c_x, zeros_x]),
                np.hstack([zeros_z, self.c_z]),
            ]
        )
        self.logical = np.vstack(
            [
                np.hstack([self.logical_x, np.zeros_like(self.logical_x)]),
                np.hstack([np.zeros_like(self.logical_z), self.logical_z]),
            ]
        )
        self._stabilizer_rref = rref(self.stabilizer)

    def analyze_transversal(self):
        """The complete strict-transversal Clifford family of this code.

        Thin delegate to :func:`qec_transversal.ansatz.strict_css.analyze_strict_css`;
        imported lazily so that :mod:`qec_transversal.codes` stays free of any
        dependency on the ansatz layer.
        """

        from ..ansatz.strict_css import analyze_strict_css

        return analyze_strict_css(self)

    def to_stabilizer_code(self):
        """The same code as a general :class:`~.stabilizer.StabilizerCode`.

        ``S = rowspan([C_X | 0]) + rowspan([0 | C_Z])``.  The bridge lets the
        general preservation-algebra solvers run on a CSS code, which is how
        the specialised and general engines are cross-validated against each
        other in the test suite.
        """

        from .stabilizer import StabilizerCode

        return StabilizerCode(self.stabilizer)
