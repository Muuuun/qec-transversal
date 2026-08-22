"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.gf2`` is now :mod:`qec_transversal.utils.gf2` (linear
algebra) and :mod:`qec_transversal.utils.symplectic` (the symplectic form).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .utils.gf2 import *  # noqa: F403
from .utils.gf2 import (
    _PACKED_MIN_COLS,
    BinaryMatrix,
    _rref_packed,
    as_binary_matrix,
    gf2_inverse,
    gf2_matmul,
    is_in_rowspace,
    nullspace,
    quotient_complement,
    rank,
    reduce_rows,
    row_basis,
    rowspace_residues,
    rref,
    supports,
)
from .utils.symplectic import is_symplectic, symplectic_form, symplectic_product


__all__ = [
    "BinaryMatrix",
    "_PACKED_MIN_COLS",
    "_rref_packed",
    "as_binary_matrix",
    "gf2_inverse",
    "gf2_matmul",
    "is_in_rowspace",
    "is_symplectic",
    "nullspace",
    "quotient_complement",
    "rank",
    "reduce_rows",
    "row_basis",
    "rowspace_residues",
    "rref",
    "supports",
    "symplectic_form",
    "symplectic_product",
]
