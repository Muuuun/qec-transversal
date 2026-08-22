"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.signed`` is now :mod:`qec_transversal.certificates.signed`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .certificates.signed import (
    SignedStabilizer,
    SignExactResult,
    tableau_from_symplectic,
    verify_sign_exact,
)


__all__ = [
    "SignExactResult",
    "SignedStabilizer",
    "tableau_from_symplectic",
    "verify_sign_exact",
]
