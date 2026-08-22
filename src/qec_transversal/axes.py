"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.axes`` split into :mod:`qec_transversal.hierarchy.general`
(diagonal kernels for arbitrary stabilizer codes) and
:mod:`qec_transversal.hierarchy.frames` (the axis-frame sweep).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .hierarchy.frames import (
    AxisFrameResult,
    axis_frame_group,
    frame_conjugated_code,
)
from .hierarchy.general import (
    diagonal_kernel_general,
    diagonal_kernel_general_exact,
)


__all__ = [
    "AxisFrameResult",
    "axis_frame_group",
    "diagonal_kernel_general",
    "diagonal_kernel_general_exact",
    "frame_conjugated_code",
]
