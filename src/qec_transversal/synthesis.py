"""Constructive logical-target verification for strict-transversal gates.

The headline verifier interface: given a CSS code and a target logical
Clifford, either return an explicit transversal implementation — the
three-layer normal form ``U_Z(a) U_X(b) U_Z(a')`` guaranteed by Albert's
structure theorem (arXiv:2608.05688, Eq. 21) — or a *complete* NO: the
target is provably outside the logical image of the entire
strict-transversal Clifford group.

Synthesis is exact linear algebra.  Writing the target as blocks
``L = [[alpha, beta], [gamma, delta]]`` and the layers as shears
``[[I, S1], [0, I]] [[I, 0], [T, I]] [[I, S2], [0, I]]``, the product
matches ``L`` iff

    ``T = gamma``  (with ``T`` in the span of the X-family shears),
    ``S1 gamma = alpha + I``,
    ``gamma S2 = delta + I``,
    ``S1 + alpha S2 = beta``,

where the last equation — a priori bilinear through ``S1 gamma S2`` —
becomes linear after substituting ``S1 gamma = alpha + I``.  All four are
one joint GF(2) system over the parameter-space coefficients, so both the
YES (with witness) and the NO (complete, by the structure theorem) are
exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .css import CSSCode

_SINGLE = {
    "S": np.array([[1, 1], [0, 1]], dtype=np.uint8),
    "SQRT_X": np.array([[1, 0], [1, 1]], dtype=np.uint8),
    "H": np.array([[0, 1], [1, 0]], dtype=np.uint8),
}


def logical_target(k: int, name: str, *qubits: int) -> np.ndarray:
    """A named logical gate as a ``2k x 2k`` symplectic matrix.

    Supported: ``S i``, ``SQRT_X i``, ``H i``, ``CZ i j``, ``CNOT i j``
    (control ``i``), ``SWAP i j`` — row-vector ``(x | z)`` convention.
    """

    matrix = np.eye(2 * k, dtype=np.uint8)
    if name in _SINGLE:
        (i,) = qubits
        block = _SINGLE[name]
        matrix[i, i] = block[0, 0]
        matrix[i, k + i] = block[0, 1]
        matrix[k + i, i] = block[1, 0]
        matrix[k + i, k + i] = block[1, 1]
        return matrix
    i, j = qubits
    if name == "CZ":
        matrix[i, k + j] = 1
        matrix[j, k + i] = 1
        return matrix
    if name == "CNOT":  # X_i -> X_i X_j, Z_j -> Z_i Z_j
        matrix[i, j] = 1
        matrix[k + j, k + i] = 1
        return matrix
    if name == "SWAP":
        for a, b in ((i, j), (j, i)):
            matrix[a, a] = 0
            matrix[a, b] = 1
            matrix[k + a, k + a] = 0
            matrix[k + a, k + b] = 1
        return matrix
    raise ValueError(f"unknown gate name {name!r}")


def _solve_gf2(system: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    """One solution of ``system @ x = rhs`` over GF(2), or None."""

    rows, cols = system.shape
    augmented = np.hstack([system % 2, rhs[:, None] % 2]).astype(np.uint8)
    r = 0
    pivots = []
    for c in range(cols):
        hit = np.flatnonzero(augmented[r:, c])
        if hit.size == 0:
            continue
        sel = r + int(hit[0])
        if sel != r:
            augmented[[r, sel]] = augmented[[sel, r]]
        others = np.flatnonzero(augmented[:, c])
        others = others[others != r]
        if others.size:
            augmented[others] ^= augmented[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if augmented[i, -1] and not augmented[i, :-1].any():
            return None
    if any(augmented[i, -1] and not augmented[i, :-1].any() for i in range(rows)):
        return None
    solution = np.zeros(cols, dtype=np.uint8)
    for i, c in enumerate(pivots):
        solution[c] = augmented[i, -1]
    return solution


@dataclass(frozen=True)
class SynthesisResult:
    found: bool
    scope: str
    z_layer_1: np.ndarray | None  # a  in span(A_Z)
    x_layer: np.ndarray | None  # b  in span(A_X)
    z_layer_2: np.ndarray | None  # a' in span(A_Z)
    verified: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        def support(vector):
            return None if vector is None else np.flatnonzero(vector).astype(int).tolist()

        return {
            "found": self.found,
            "scope": self.scope,
            "implementation": (
                {
                    "Z_layer_1_support": support(self.z_layer_1),
                    "X_layer_support": support(self.x_layer),
                    "Z_layer_2_support": support(self.z_layer_2),
                }
                if self.found
                else None
            ),
            "verified": self.verified,
            "reason": self.reason,
        }


def verify_logical_gate(
    code: CSSCode, target: np.ndarray | str, *qubits: int
) -> SynthesisResult:
    """Decide whether ``target`` has a strict-transversal implementation.

    YES returns the three-layer witness (supports for ``sqrt(Z)``-type,
    ``sqrt(X)``-type, ``sqrt(Z)``-type layers), self-verified by
    recomputing the composed logical action.  NO is complete for the
    strict class by the three-layer structure theorem.
    """

    k = code.k
    if isinstance(target, str):
        target = logical_target(k, target, *qubits)
    target = np.asarray(target, dtype=np.uint8) % 2
    if target.shape != (2 * k, 2 * k):
        raise ValueError(f"target must be {2 * k} x {2 * k}")

    scope = "strict-transversal Clifford (complete; Albert arXiv:2608.05688 Eq. 21)"
    analysis = code.analyze_transversal()
    z_basis = analysis.a_z.basis
    x_basis = analysis.a_x.basis
    dz, dx = z_basis.shape[0], x_basis.shape[0]
    # shear images of the parameter bases
    z_shears = [
        ((code.logical_x * a) @ code.logical_x.T % 2).astype(np.uint8) for a in z_basis
    ]
    x_shears = [
        ((code.logical_z * b) @ code.logical_z.T % 2).astype(np.uint8) for b in x_basis
    ]

    alpha = target[:k, :k]
    beta = target[:k, k:]
    gamma = target[k:, :k]
    delta = target[k:, k:]
    eye = np.eye(k, dtype=np.uint8)

    # joint linear system over (c1 | b | c2): S1 = sum c1_i Sz_i etc.
    kk = k * k
    columns = dz + dx + dz
    rows_total = 4 * kk
    system = np.zeros((rows_total, columns), dtype=np.uint8)
    rhs = np.zeros(rows_total, dtype=np.uint8)
    # (1) T(b) = gamma
    for j, Tj in enumerate(x_shears):
        system[0:kk, dz + j] = Tj.reshape(-1)
    rhs[0:kk] = gamma.reshape(-1)
    # (2) S1 gamma = alpha + I
    for i, Si in enumerate(z_shears):
        system[kk : 2 * kk, i] = ((Si @ gamma) % 2).reshape(-1)
    rhs[kk : 2 * kk] = ((alpha ^ eye) % 2).reshape(-1)
    # (3) gamma S2 = delta + I
    for i, Si in enumerate(z_shears):
        system[2 * kk : 3 * kk, dz + dx + i] = ((gamma @ Si) % 2).reshape(-1)
    rhs[2 * kk : 3 * kk] = ((delta ^ eye) % 2).reshape(-1)
    # (4) S1 + alpha S2 = beta   (bilinear term linearized via (2))
    for i, Si in enumerate(z_shears):
        system[3 * kk :, i] = Si.reshape(-1)
        system[3 * kk :, dz + dx + i] = ((alpha @ Si) % 2).reshape(-1)
    rhs[3 * kk :] = beta.reshape(-1)

    solution = _solve_gf2(system, rhs)
    if solution is None:
        return SynthesisResult(
            found=False,
            scope=scope,
            z_layer_1=None,
            x_layer=None,
            z_layer_2=None,
            verified=True,
            reason=(
                "target is outside the exact logical image of the "
                "strict-transversal Clifford group (three-layer system "
                "unsolvable; complete by the structure theorem)"
            ),
        )

    c1 = solution[:dz]
    b_coeff = solution[dz : dz + dx]
    c2 = solution[dz + dx :]
    a1 = (c1 @ z_basis % 2).astype(np.uint8) if dz else np.zeros(code.n, dtype=np.uint8)
    b = (b_coeff @ x_basis % 2).astype(np.uint8) if dx else np.zeros(code.n, dtype=np.uint8)
    a2 = (c2 @ z_basis % 2).astype(np.uint8) if dz else np.zeros(code.n, dtype=np.uint8)

    # independent self-check: recompose the logical action
    def shear_z(a):
        S = ((code.logical_x * a) @ code.logical_x.T % 2).astype(np.uint8)
        return np.block([[eye, S], [np.zeros((k, k), np.uint8), eye]]).astype(np.uint8)

    def shear_x(bv):
        T = ((code.logical_z * bv) @ code.logical_z.T % 2).astype(np.uint8)
        return np.block([[eye, np.zeros((k, k), np.uint8)], [T, eye]]).astype(np.uint8)

    product = shear_z(a1) @ shear_x(b) @ shear_z(a2) % 2
    verified = bool(np.array_equal(product.astype(np.uint8), target))
    return SynthesisResult(
        found=True,
        scope=scope,
        z_layer_1=a1,
        x_layer=b,
        z_layer_2=a2,
        verified=verified,
        reason="three-layer witness found and recomposition verified"
        if verified
        else "witness found but recomposition FAILED — do not trust",
    )
