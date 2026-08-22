"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.oneblock`` split into
:mod:`qec_transversal.logical.generated` (the analysis and driver),
:mod:`qec_transversal.logical.recognition` (the McLaughlin route), and
:mod:`qec_transversal.logical.words` (the word-tracking chain).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .logical.generated import (
    OneBlockAnalysis,
    OneBlockGenerator,
    analyze_one_block,
    factor_target,
    single_matching_fullness,
)
from .logical.recognition import (
    RecognitionReport,
    recognize_full_symplectic,
)
from .logical.words import WordBSGS
from .utils.symplectic import symplectic_transvection


__all__ = [
    "OneBlockAnalysis",
    "OneBlockGenerator",
    "RecognitionReport",
    "WordBSGS",
    "analyze_one_block",
    "factor_target",
    "recognize_full_symplectic",
    "single_matching_fullness",
    "symplectic_transvection",
]
