"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.matching`` is now :mod:`qec_transversal.ansatz.matching`;
``logical_group_summary`` moved to :mod:`qec_transversal.logical.group`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.matching import (
    FoldHadamard,
    MatchingAnalysis,
    MatchingGenerator,
    analyze_matching,
    involution_pairs,
    sigma_matrix,
)
from .logical.group import logical_group_summary


__all__ = [
    "FoldHadamard",
    "MatchingAnalysis",
    "MatchingGenerator",
    "analyze_matching",
    "involution_pairs",
    "logical_group_summary",
    "sigma_matrix",
]
