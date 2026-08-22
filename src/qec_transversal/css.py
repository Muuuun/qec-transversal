"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.css`` split into :mod:`qec_transversal.codes.css`
(the code object) and :mod:`qec_transversal.ansatz.strict_css` (the
strict-transversal solver).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.strict_css import (
    Family,
    ParameterSpace,
    TransversalAnalysis,
    TransversalGenerator,
    shear_images,
    shear_matrix,
)
from .codes.css import CSSCode


__all__ = [
    "CSSCode",
    "Family",
    "ParameterSpace",
    "TransversalAnalysis",
    "TransversalGenerator",
    "shear_images",
    "shear_matrix",
]
