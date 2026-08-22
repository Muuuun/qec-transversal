"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.codewordaut`` is now
:mod:`qec_transversal.ansatz.codeword_permutation`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.codeword_permutation import (
    CodewordAutomorphisms,
    analyze_codeword_automorphisms,
)


__all__ = [
    "CodewordAutomorphisms",
    "analyze_codeword_automorphisms",
]
