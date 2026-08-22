"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.group`` is now :mod:`qec_transversal.logical.group`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .logical.group import (
    GroupOrder,
    _apply,
    _pack_rows,
    generated_group_order,
    logical_group_summary,
    schreier_sims_order,
)
from .utils.symplectic import symplectic_group_order


__all__ = [
    "GroupOrder",
    "_apply",
    "_pack_rows",
    "generated_group_order",
    "logical_group_summary",
    "schreier_sims_order",
    "symplectic_group_order",
]
