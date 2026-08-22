"""Diagonal gates in the Clifford hierarchy.

A depth-one diagonal layer ``U(t) = diag(omega^{t . u})`` with
``omega = exp(2 pi i / 2^L)`` is not a symplectic object, so these solvers do
not use the preservation algebra: code preservation becomes a system of
congruences over ``Z_{2^L}``, solved exactly by module elimination
(:mod:`qec_transversal.utils.modular`).

:mod:`.css` is complete for CSS codes at any level; :mod:`.general` is sound
for arbitrary stabilizer codes and complete under stated conditions;
:mod:`.frames` sweeps per-qubit Pauli axis frames, which by the
Zeng-Cross-Chuang structure theorem closes the whole single-qubit transversal
class when the sweep is exhaustive and every frame reports complete.
"""

from .css import HierarchyAnalysis, analyze_hierarchy
from .frames import AxisFrameResult, axis_frame_group, frame_conjugated_code
from .general import diagonal_kernel_general, diagonal_kernel_general_exact

__all__ = [
    "AxisFrameResult",
    "HierarchyAnalysis",
    "analyze_hierarchy",
    "axis_frame_group",
    "diagonal_kernel_general",
    "diagonal_kernel_general_exact",
    "frame_conjugated_code",
]
