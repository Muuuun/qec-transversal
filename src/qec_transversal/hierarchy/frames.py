r"""Per-qubit Pauli axis frames: closing the single-qubit transversal class.

A diagonal solver only sees gates diagonal in the computational basis.  By the
Zeng-Cross-Chuang structure theorem every single-qubit transversal gate is
local-Clifford-equivalent to a *frame-diagonal* one: choose, per qubit, which
Pauli axis its rotations are diagonal in, conjugate the code into that frame,
and solve the diagonal problem there.

Sweeping all ``3^n`` frames therefore closes the entire single-qubit
transversal class at the given hierarchy level -- **provided** every reported
frame carries ``complete=True``.  Frames routed to the complete CSS ladder
(:mod:`.css`), to the capped exact engine, or covered by the ``z_dim == 0``
rule do; a frame beyond the exact engine's cap with ``z_dim > 0`` reports
``complete=False`` and contributes only a sound subgroup.  Beyond
``frame_limit`` frames the sweep is not exhaustive and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iter_product
from typing import Any

import numpy as np

from ..codes.stabilizer import StabilizerCode
from .general import diagonal_kernel_general_exact

#: Frame Cliffords as 2x2 symplectic blocks (row-vector convention): the
#: frame maps the chosen rotation axis onto Z.
_FRAME_TO_Z = {
    "Z": np.array([[1, 0], [0, 1]], dtype=np.uint8),
    "X": np.array([[0, 1], [1, 0]], dtype=np.uint8),  # H
    "Y": np.array([[1, 1], [1, 0]], dtype=np.uint8),  # maps Y = (1,1) -> Z = (0,1)
}


def frame_conjugated_code(code: StabilizerCode, axes: tuple[str, ...]) -> StabilizerCode:
    """The code conjugated so each qubit's chosen axis becomes Z."""

    n = code.n
    matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i, axis in enumerate(axes):
        block = _FRAME_TO_Z[axis]
        matrix[i, i] = block[0, 0]
        matrix[i, n + i] = block[0, 1]
        matrix[n + i, i] = block[1, 0]
        matrix[n + i, n + i] = block[1, 1]
    rows = (code.h.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(np.uint8)
    return StabilizerCode(rows)


@dataclass(frozen=True)
class AxisFrameResult:
    frames_tested: int
    exhaustive: bool
    nontrivial_frames: list[tuple[tuple[str, ...], int, bool]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_tested": self.frames_tested,
            "exhaustive": self.exhaustive,
            "nontrivial_frames": [
                {"axes": "".join(axes), "kernel_generators": count, "complete": complete}
                for axes, count, complete in self.nontrivial_frames
            ],
        }


def axis_frame_group(
    code: StabilizerCode, *, level: int = 3, frame_limit: int = 60_000
) -> AxisFrameResult:
    """Sweep per-qubit axis frames; report frames with nontrivial kernels.

    Exhaustive when ``3^n <= frame_limit`` — by Zeng-Cross-Chuang this then
    covers the complete single-qubit transversal class at the given level,
    *provided* every reported frame carries ``complete=True``: frames
    routed to the CSS ladder or the capped exact engine do, while a frame
    beyond the cap with ``z_dim > 0`` reports ``complete=False`` and its
    kernel is only a sound subgroup.
    """

    n = code.n
    exhaustive = 3**n <= frame_limit
    if exhaustive:
        frames = iter_product("ZXY", repeat=n)
        count = 3**n
    else:
        frames = iter(
            [tuple("Z" for _ in range(n)), tuple("X" for _ in range(n)), tuple("Y" for _ in range(n))]
        )
        count = 3
    nontrivial: list[tuple[tuple[str, ...], int, bool]] = []
    modulus = 1 << level
    for axes in frames:
        conjugated = frame_conjugated_code(code, tuple(axes))
        # NOTE: ``StabilizerCode.__init__`` row-reduces the conjugated
        # rows, and some frames CSS-split only after that reduction (e.g.
        # the Steane Y-frame).  A refactor feeding raw conjugated rows to
        # ``_css_split`` would silently reroute those frames from the
        # complete CSS ladder to the general engine.
        css_split = _css_split(conjugated)
        if css_split is not None:
            from ..codes.css import CSSCode
            from .css import analyze_hierarchy

            h_x, h_z = css_split
            css = CSSCode(h_x, h_z, n=n)
            kernel = analyze_hierarchy(css, "Z", level=level).kernel
            complete = True
        else:
            kernel, complete = diagonal_kernel_general_exact(conjugated, level=level)
        interesting = any((row % (modulus // 2)).any() for row in kernel)
        if interesting:
            nontrivial.append((tuple(axes), int(kernel.shape[0]), complete))
    return AxisFrameResult(count, exhaustive, nontrivial)


def _css_split(code: StabilizerCode):
    """Split the row basis into pure-X / pure-Z parts when possible."""

    n = code.n
    x_rows, z_rows = [], []
    for row in code.h:
        has_x, has_z = row[:n].any(), row[n:].any()
        if has_x and has_z:
            return None
        if has_x:
            x_rows.append(row[:n])
        elif has_z:
            z_rows.append(row[n:])
    h_x = np.asarray(x_rows, dtype=np.uint8) if x_rows else np.zeros((0, n), dtype=np.uint8)
    h_z = np.asarray(z_rows, dtype=np.uint8) if z_rows else np.zeros((0, n), dtype=np.uint8)
    return h_x, h_z
