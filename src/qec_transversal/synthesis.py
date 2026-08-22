"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.synthesis`` is now :mod:`qec_transversal.logical.synthesis`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .logical.synthesis import (
    SynthesisResult,
    logical_target,
    verify_logical_gate,
)


__all__ = [
    "SynthesisResult",
    "logical_target",
    "verify_logical_gate",
]
