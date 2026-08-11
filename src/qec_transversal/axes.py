"""Diagonal hierarchy gates on general stabilizer codes, and axis frames.

Extends the CSS coset-phase solver to arbitrary stabilizer codes: a
diagonal layer ``U(t) = diag(omega^{t.u})`` (``omega = e^{2 pi i / 2^L}``)
preserves the code space iff for every stabilizer generator ``X_a Z_b``
there is a Z-type stabilizer ``Z_c`` absorbing the conjugation residue:

    ``U X_a Z_b U^dag = omega^{t.a} X_a Z_b diag(omega^{-2 t.(u & a)})``

and the diagonal factor must equal ``(-1)^{?} Z_c`` for some ``c`` in the
Z-stabilizer subspace, giving entrywise congruences
``2 (t & a) + 2^{L-1} c = 0 (mod 2^L)`` plus a scalar phase congruence.
The mixed system in ``(t, c)`` is a Z_{2^L} module kernel with auxiliary
variables.  Scope, validated against dense projector brute force: the
operator-level condition implemented here is SOUND (every returned gate
preserves the code space) and complete except when phase agreement is only
needed on a proper support coset — the fully complete engine for that case
is the CSS coset ladder of :mod:`.hierarchy`, which
:func:`axis_frame_group` uses automatically whenever the frame-conjugated
code is CSS.

On top of this, :func:`axis_frame_group` searches per-qubit *axis frames*
(which Pauli axis each qubit's rotations are diagonal in): by the
Zeng-Cross-Chuang structure theorem every single-qubit transversal gate is
local-Clifford-equivalent to such a frame-diagonal gate, so exhausting
frames (feasible for small ``n``; sampled beyond) closes the entire
single-qubit transversal class.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iter_product
from typing import Any

import numpy as np

from .hierarchy import _valuation, module_kernel
from .stabilizer import StabilizerCode

#: Frame Cliffords as 2x2 symplectic blocks (row-vector convention): the
#: frame maps the chosen rotation axis onto Z.
_FRAME_TO_Z = {
    "Z": np.array([[1, 0], [0, 1]], dtype=np.uint8),
    "X": np.array([[0, 1], [1, 0]], dtype=np.uint8),  # H
    "Y": np.array([[1, 1], [1, 0]], dtype=np.uint8),  # maps Y = (1,1) -> Z = (0,1)
}


def _z_subspace_with_signs(code: StabilizerCode):
    """Basis of the Z-type stabilizer subspace with its sign character.

    Each basis element ``c`` arises as a product of the given generators;
    the product's sign ``epsilon(c) = (-1)^{lambda(c)}`` (Hermitian Pauli
    convention, computed exactly with Stim) defines a linear character
    ``lambda`` because diagonal Paulis multiply without extra signs.
    """

    import stim

    n = code.n
    from .gf2 import nullspace, row_basis

    x_part = code.h[:, :n]
    combos = nullspace(x_part.T)
    basis_rows: list[np.ndarray] = []
    lambdas: list[int] = []
    seen = None
    for combo in combos:
        z = (combo @ code.h[:, n:]) % 2
        z = z.astype(np.uint8)
        if not z.any():
            continue
        stack = np.vstack(basis_rows + [z]) if basis_rows else z[None, :]
        if row_basis(stack, ncols=n).shape[0] <= len(basis_rows):
            continue
        product = stim.PauliString(n)
        for g_index in np.flatnonzero(combo):
            row = code.h[g_index]
            ps = stim.PauliString(n)
            for i in range(n):
                xi, zi = int(row[i]), int(row[n + i])
                if xi and zi:
                    ps[i] = "Y"
                elif xi:
                    ps[i] = "X"
                elif zi:
                    ps[i] = "Z"
            product = product * ps
        sign = complex(product.sign)
        if sign == 1:
            lambdas.append(0)
        elif sign == -1:
            lambdas.append(1)
        else:  # pragma: no cover - diagonal products are Hermitian
            raise AssertionError("imaginary sign on a Z-type stabilizer")
        basis_rows.append(z)
    if not basis_rows:
        return np.zeros((0, n), dtype=np.uint8), np.zeros(0, dtype=np.int64)
    return np.asarray(basis_rows, dtype=np.uint8), np.asarray(lambdas, dtype=np.int64)


def diagonal_kernel_general(
    code: StabilizerCode, *, level: int = 3
) -> np.ndarray:
    """Generators of ``{t : U(t) preserves the code space}`` over Z_{2^L}.

    Homogeneous module system in ``(t, c-coefficients)``: for each
    stabilizer generator ``X_a Z_b`` there must exist ``c`` in the Z-type
    subspace with, entrywise, ``2 (t & a) + 2^{L-1} c = 0 (mod 2^L)`` and,
    scalar, ``t . a + 2^{L-1} lambda(c) = 0 (mod 2^L)``.  Validated against
    dense projector brute force in the tests.
    """

    n = code.n
    z_basis, z_lambda = _z_subspace_with_signs(code)
    z_dim = z_basis.shape[0]
    generators = code.h
    total = n + z_dim * generators.shape[0]

    constraints: list[tuple[np.ndarray, int]] = []
    for g_index, row in enumerate(generators):
        a = row[:n].astype(np.int64)
        aux0 = n + g_index * z_dim
        for i in np.flatnonzero(a):
            pattern = np.zeros(total, dtype=np.int64)
            pattern[i] = 2
            for j in range(z_dim):
                pattern[aux0 + j] = (1 << (level - 1)) * int(z_basis[j, i])
            constraints.append((pattern, 1))
        for i in range(n):
            if a[i]:
                continue
            # c must vanish outside supp(a): 2^{L-1} c_i = 0
            pattern = np.zeros(total, dtype=np.int64)
            for j in range(z_dim):
                pattern[aux0 + j] = (1 << (level - 1)) * int(z_basis[j, i])
            if pattern.any():
                constraints.append((pattern, 1))
        pattern = np.zeros(total, dtype=np.int64)
        pattern[:n] = a
        for j in range(z_dim):
            pattern[aux0 + j] = (1 << (level - 1)) * int(z_lambda[j])
        constraints.append((pattern, 1))

    kernel = module_kernel(constraints, total, exponent=level)
    modulus = 1 << level
    projected = [gen[:n] % modulus for gen in kernel if (gen[:n] % modulus).any()]
    if not projected:
        return np.zeros((0, n), dtype=np.int64)
    return np.asarray(projected, dtype=np.int64)


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
    covers the complete single-qubit transversal class at the given level.
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
        css_split = _css_split(conjugated)
        if css_split is not None:
            from .css import CSSCode
            from .hierarchy import analyze_hierarchy

            h_x, h_z = css_split
            css = CSSCode(h_x, h_z, n=n)
            kernel = analyze_hierarchy(css, "Z", level=level).kernel
            complete = True
        else:
            kernel = diagonal_kernel_general(conjugated, level=level)
            complete = False  # sound subgroup; support-coset ladder not applied
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
