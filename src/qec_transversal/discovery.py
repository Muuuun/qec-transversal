"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.discovery`` is now :mod:`qec_transversal.ansatz.discovery`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.discovery import (
    certified_shift_structure,
    discover_involutions,
    structural_permutations,
)


__all__ = [
    "certified_shift_structure",
    "discover_involutions",
    "structural_permutations",
]
