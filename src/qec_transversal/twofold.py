"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.twofold`` is now :mod:`qec_transversal.ansatz.twofold`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.twofold import (
    TwoFoldResult,
    automorphism_involutions,
    levi_logical_generators,
    two_fold_group,
)


__all__ = [
    "TwoFoldResult",
    "automorphism_involutions",
    "levi_logical_generators",
    "two_fold_group",
]
