"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.phase`` is now :mod:`qec_transversal.certificates.phase`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .certificates.phase import (
    PhaseVerification,
    PhaseVerifiedGenerator,
    verify_phases,
)


__all__ = [
    "PhaseVerification",
    "PhaseVerifiedGenerator",
    "verify_phases",
]
