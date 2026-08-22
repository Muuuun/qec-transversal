"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.monomial`` is now :mod:`qec_transversal.ansatz.monomial`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.monomial import (
    MonomialAnalysis,
    MonomialGenerator,
    analyze_monomial,
    strict_cross_check,
)


__all__ = [
    "MonomialAnalysis",
    "MonomialGenerator",
    "analyze_monomial",
    "strict_cross_check",
]
