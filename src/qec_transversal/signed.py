"""Signed stabilizer representation and sign-exact gate verification.

Generalizes :mod:`.phase` from diagonal circuits to *arbitrary* symplectic
gates: a code is carried as ``(H, sigma)`` — symplectic rows plus explicit
generator signs — and any ``2n x 2n`` symplectic matrix (from the strict,
fold, partition, or monomial engines) is lifted to an exact Stim tableau,
conjugated through the signed generators, given an explicit Pauli
correction, and re-verified to fix every generator with sign ``+1``.

This removes the "modulo Pauli, modulo phase" fine print from any engine's
output: the returned object is an executable gate (tableau + Pauli fix)
with a machine-checked sign-exact certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover
    import stim
except ImportError:  # pragma: no cover
    stim = None


def _require_stim() -> None:
    if stim is None:
        raise ImportError("signed verification needs stim (pip install stim)")


class SignedStabilizer:
    """Stabilizer generators with explicit signs: the ``(H, sigma)`` object.

    ``rows`` is an ``(m, 2n)`` GF(2) matrix in ``[X | Z]`` convention and
    ``signs`` assigns each row a sign in ``{+1, -1}``.  The rows need *not*
    be linearly independent: dependent rows are fully supported, provided
    their signs are consistent with the group the rows generate — the sign
    attached to a dependent row must equal the sign of the Stim product of
    the rows it depends on (:meth:`validate` checks exactly this).  If some
    dependency multiplies to ``-I``, the input is not a stabilizer group
    (nothing is stabilized), and :func:`verify_sign_exact` refuses it with a
    ``ValueError`` up front instead of reporting a spurious gate failure.
    """

    def __init__(self, rows: np.ndarray, signs: list[int] | None = None):
        rows = np.asarray(rows, dtype=np.uint8) % 2
        if rows.ndim != 2 or rows.shape[1] % 2:
            raise ValueError("rows must be (m, 2n)")
        self.rows = rows
        self.n = rows.shape[1] // 2
        self.signs = list(signs) if signs is not None else [1] * rows.shape[0]
        if len(self.signs) != rows.shape[0]:
            raise ValueError("signs must have one entry per row")
        if any(s not in (1, -1) for s in self.signs):
            raise ValueError("signs must be +1 or -1")

    def validate(self) -> None:
        """Reject inputs that do not define a genuine stabilizer group.

        Raises ``ValueError`` if either check fails:

        1. every pair of rows commutes (symplectic inner product ``0``);
        2. for every GF(2) linear dependency ``v`` among the rows, the Stim
           product of the signed Paulis in ``supp(v)`` has sign ``+1`` — a
           ``-1`` sign means the generated group contains ``-I`` and
           stabilizes no state.

        Checking a *basis* of the left kernel suffices: commuting Hermitian
        Paulis square to ``+I``, so for dependencies ``v, w`` the products
        satisfy ``P_v P_w = P_{v xor w}`` and the sign is multiplicative
        along XOR of dependencies.  An imaginary sign cannot occur — a
        product of commuting Hermitian operators is Hermitian, hence its
        identity-support product has sign ``+-1`` — so it is asserted, not
        raised.
        """

        _require_stim()
        n = self.n
        x = self.rows[:, :n].astype(np.int64)
        z = self.rows[:, n:].astype(np.int64)
        gram = (x @ z.T + z @ x.T) % 2
        if gram.any():
            i, j = (int(v) for v in np.argwhere(gram)[0])
            raise ValueError(f"rows {i} and {j} anticommute: not a stabilizer group")
        for vector in _left_kernel_basis(self.rows):
            product = stim.PauliString(n)
            for j in np.flatnonzero(vector):
                product = product * self.pauli(int(j))
            sign = complex(product.sign)
            assert sign.imag == 0, (
                "imaginary sign for a product of commuting Hermitian Paulis"
            )
            if sign.real < 0:
                raise ValueError(
                    "generator signs inconsistent: -I in generated group"
                )

    def pauli(self, index: int) -> stim.PauliString:
        _require_stim()
        row = self.rows[index]
        ps = stim.PauliString(self.n)
        for i in range(self.n):
            x, z = int(row[i]), int(row[self.n + i])
            if x and z:
                ps[i] = "Y"
            elif x:
                ps[i] = "X"
            elif z:
                ps[i] = "Z"
        return ps * stim.PauliString(f"{'+' if self.signs[index] > 0 else '-'}I")


def tableau_from_symplectic(matrix: np.ndarray, n: int) -> stim.Tableau:
    """An exact Stim tableau realizing a symplectic matrix (signs zeroed).

    Row-vector convention: the image of ``X_i`` is row ``i`` of the matrix,
    the image of ``Z_i`` is row ``n + i``.
    """

    _require_stim()
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    if matrix.shape != (2 * n, 2 * n):
        raise ValueError("matrix must be 2n x 2n")
    x2x = matrix[:n, :n].astype(bool)
    x2z = matrix[:n, n:].astype(bool)
    z2x = matrix[n:, :n].astype(bool)
    z2z = matrix[n:, n:].astype(bool)
    signs = np.zeros(2 * n, dtype=bool)
    return stim.Tableau.from_numpy(
        x2x=x2x, x2z=x2z, z2x=z2x, z2z=z2z,
        x_signs=signs[:n], z_signs=signs[n:],
    )


def _left_kernel_basis(rows: np.ndarray) -> list[np.ndarray]:
    """Basis of ``{v : v @ rows = 0 mod 2}`` via a tracked row reduction."""

    m, cols = rows.shape
    augmented = np.hstack([rows % 2, np.eye(m, dtype=np.uint8)]).astype(np.uint8)
    r = 0
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
        r += 1
        if r == m:
            break
    return [augmented[i, cols:] for i in range(r, m)]


def _decompose_over(rows: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    m = rows.shape[0]
    augmented = np.hstack([rows.T % 2, (target % 2)[:, None]]).astype(np.uint8)
    r = 0
    pivots = []
    total_rows, _ = augmented.shape
    for c in range(m):
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
        if r == total_rows:
            break
    if any(augmented[i, -1] and not augmented[i, :-1].any() for i in range(total_rows)):
        return None
    coefficients = np.zeros(m, dtype=np.uint8)
    for i, c in enumerate(pivots):
        coefficients[c] = augmented[i, -1]
    return coefficients


@dataclass(frozen=True)
class SignExactResult:
    preserved: bool
    pauli_correction: str
    certificate: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preserved": self.preserved,
            "pauli_correction": self.pauli_correction,
            "certificate": self.certificate,
        }


def verify_sign_exact(code: SignedStabilizer, matrix: np.ndarray) -> SignExactResult:
    """Sign-exact verification of an arbitrary symplectic gate.

    Conjugates every signed generator through the exact tableau, solves the
    sign defects into an explicit Pauli correction, and re-verifies that
    the corrected gate maps every generator to a ``+1`` product of
    generators.  ``preserved = False`` means the symplectic matrix does not
    even fix the row space — a soundness guard, not a phase issue.

    The input is validated first (:meth:`SignedStabilizer.validate`), so a
    ``ValueError`` always flags bad *input*, never a gate failure.
    Dependent rows in ``code`` are fully supported: in a valid signed group
    (one without ``-I``) every element of the row space carries a unique
    sign, so the sign defect ``d(s)`` of a conjugated generator is a
    character on the stabilizer group — ``d(s t) = d(s) xor d(t)`` — which
    makes the defect vector orthogonal to every linear dependency among the
    rows and the Pauli-correction system automatically consistent.
    """

    _require_stim()
    code.validate()
    n = code.n
    tableau = tableau_from_symplectic(matrix, n)

    defects = np.zeros(code.rows.shape[0], dtype=np.uint8)
    for index in range(code.rows.shape[0]):
        image = tableau(code.pauli(index))
        x_bits = np.zeros(n, dtype=np.uint8)
        z_bits = np.zeros(n, dtype=np.uint8)
        x_bits[list(image.pauli_indices("XY"))] = 1
        z_bits[list(image.pauli_indices("ZY"))] = 1
        image_row = np.concatenate([x_bits, z_bits])
        coefficients = _decompose_over(code.rows, image_row)
        if coefficients is None:
            return SignExactResult(
                preserved=False,
                pauli_correction="",
                certificate={"row_space_preserved": False},
            )
        product = stim.PauliString(n)
        for j in np.flatnonzero(coefficients):
            product = product * code.pauli(j)
        ratio = complex(image.sign) / complex(product.sign)
        if ratio == 1:
            defects[index] = 0
        elif ratio == -1:
            defects[index] = 1
        else:  # pragma: no cover
            raise AssertionError("imaginary sign defect for a Clifford gate")

    # Pauli correction p with <p, s_i> = defect_i
    omega_rows = np.hstack([code.rows[:, n:], code.rows[:, :n]]).astype(np.uint8)
    augmented = np.hstack([omega_rows % 2, defects[:, None]]).astype(np.uint8)
    rows_, _ = augmented.shape
    r = 0
    pivots = []
    for c in range(2 * n):
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
        if r == rows_:
            break
    # Invariant: after ``code.validate()`` the sign defects form a character
    # on the stabilizer group, so ``defects`` annihilates every left-kernel
    # vector of ``omega_rows`` and the system ``<p, s_i> = defect_i`` is
    # always solvable.  An inconsistent row (zero coefficients, nonzero
    # right-hand side — the mirror of the ``_decompose_over`` check) is
    # unreachable unless a -I-generating input bypassed validation.
    assert not any(
        augmented[i, -1] and not augmented[i, :-1].any() for i in range(rows_)
    ), "sign-defect system inconsistent for a validated signed group"
    correction_vec = np.zeros(2 * n, dtype=np.uint8)
    for i, c in enumerate(pivots):
        correction_vec[c] = augmented[i, -1]
    correction = stim.PauliString(n)
    for i in range(n):
        x, z = int(correction_vec[i]), int(correction_vec[n + i])
        if x and z:
            correction[i] = "Y"
        elif x:
            correction[i] = "X"
        elif z:
            correction[i] = "Z"

    # re-verify with the correction applied
    all_plus = True
    for index in range(code.rows.shape[0]):
        image = tableau(code.pauli(index))
        anti = 1 if correction.commutes(code.pauli(index)) else -1
        corrected_sign = complex(image.sign) * anti
        x_bits = np.zeros(n, dtype=np.uint8)
        z_bits = np.zeros(n, dtype=np.uint8)
        x_bits[list(image.pauli_indices("XY"))] = 1
        z_bits[list(image.pauli_indices("ZY"))] = 1
        coefficients = _decompose_over(
            code.rows, np.concatenate([x_bits, z_bits])
        )
        product = stim.PauliString(n)
        for j in np.flatnonzero(coefficients):
            product = product * code.pauli(j)
        if corrected_sign != complex(product.sign):
            all_plus = False
    return SignExactResult(
        preserved=True,
        pauli_correction=str(correction),
        certificate={
            "row_space_preserved": True,
            "stabilizer_signs_corrected_to_plus": bool(all_plus),
        },
    )
