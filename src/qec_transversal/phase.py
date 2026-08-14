"""Sign-exact (circuit-level) verification of strict-transversal generators.

The symplectic framework of this package works modulo Pauli operators and
global phases.  This module closes the gap for the strict diagonal
generators: each parameter vector is turned into an actual circuit, the
*signed* stabilizer generators are conjugated through it with Stim's exact
tableau arithmetic, an explicit Pauli correction is solved for, and the
corrected gate is re-verified to preserve every stabilizer generator with
sign ``+1`` exactly.  The logical action is likewise extracted sign-exactly
(distinguishing, e.g., a logical ``S`` from ``S^dagger``).

Requires ``stim`` (``pip install stim``); everything here is independent of
the GF(2) engines, so it doubles as an end-to-end cross-check of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .css import CSSCode

try:  # pragma: no cover
    import stim
except ImportError:  # pragma: no cover
    stim = None


def _require_stim() -> None:
    if stim is None:
        raise ImportError("phase verification needs stim (pip install stim)")


def _pauli_string(x: np.ndarray, z: np.ndarray, n: int) -> stim.PauliString:
    ps = stim.PauliString(n)
    for i in range(n):
        if x[i] and z[i]:
            ps[i] = "Y"
        elif x[i]:
            ps[i] = "X"
        elif z[i]:
            ps[i] = "Z"
    return ps


def _row_to_pauli(row: np.ndarray, n: int) -> stim.PauliString:
    return _pauli_string(row[:n], row[n:], n)


def _decompose(row: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    """Coefficients expressing ``row`` over independent ``basis_rows`` (F2)."""

    # solve basis^T c = row^T by elimination over the stacked system
    m = basis_rows.shape[0]
    aug = np.hstack([basis_rows.T % 2, row[:, None] % 2]).astype(np.uint8)
    rows_, _ = aug.shape
    r = 0
    pivots = []
    for c in range(m):
        hit = np.flatnonzero(aug[r:, c])
        if hit.size == 0:
            continue
        sel = r + int(hit[0])
        if sel != r:
            aug[[r, sel]] = aug[[sel, r]]
        others = np.flatnonzero(aug[:, c])
        others = others[others != r]
        if others.size:
            aug[others] ^= aug[r]
        pivots.append(c)
        r += 1
        if r == rows_:
            break
    if any(aug[i, -1] and not aug[i, :-1].any() for i in range(rows_)):
        raise AssertionError("row not in span of basis")
    coefficients = np.zeros(m, dtype=np.uint8)
    for i, c in enumerate(pivots):
        coefficients[c] = aug[i, -1]
    return coefficients


@dataclass(frozen=True)
class PhaseVerifiedGenerator:
    family: str
    parameter: np.ndarray
    circuit: str
    pauli_correction: str
    logical_diagonal_phases: tuple[int, ...]  # exponent of i in Xbar_j image sign
    certificate: dict[str, bool]


class PhaseVerification:
    """Sign-exact verification of every strict diagonal generator."""

    def __init__(self, code: CSSCode):
        _require_stim()
        self.code = code
        analysis = code.analyze_transversal()
        self.generators = tuple(
            self._verify(g.family, g.parameter) for g in analysis.generators
        )

    def _stabilizer_rows(self) -> np.ndarray:
        return self.code.stabilizer

    def _verify(self, family: str, parameter: np.ndarray) -> PhaseVerifiedGenerator:
        code = self.code
        n = code.n
        gate = "S" if family == "Z" else "SQRT_X"
        support = [int(q) for q in np.flatnonzero(parameter)]
        circuit = stim.Circuit()
        if support:
            circuit.append(gate, support)
        circuit.append("I", range(n))  # pin the tableau width to n qubits
        tableau = circuit.to_tableau()

        stab_rows = self._stabilizer_rows()
        # conjugate each signed (+1) stabilizer generator; find sign defects
        defects = np.zeros(stab_rows.shape[0], dtype=np.uint8)
        for index, row in enumerate(stab_rows):
            image = tableau(_row_to_pauli(row, n))
            x_bits = np.zeros(n, dtype=np.uint8)
            z_bits = np.zeros(n, dtype=np.uint8)
            x_bits[list(image.pauli_indices("XY"))] = 1
            z_bits[list(image.pauli_indices("ZY"))] = 1
            image_row = np.concatenate([x_bits, z_bits])
            coefficients = _decompose(image_row, stab_rows)
            product = stim.PauliString(n)
            for j in np.flatnonzero(coefficients):
                product = product * _row_to_pauli(stab_rows[j], n)
            ratio = image.sign / product.sign
            if ratio == 1:
                defects[index] = 0
            elif ratio == -1:
                defects[index] = 1
            else:  # pragma: no cover - diagonal Cliffords give real signs
                raise AssertionError("imaginary sign defect")

        # Pauli correction: p with <p, s_i>_symplectic = defect_i for all i.
        # Solve via the full symplectic Gram matrix against a basis of F_2^2n.
        # Use destabilizer-style solve: rows of stab_rows are independent, so
        # the linear map v -> (<v, s_i>)_i is onto; solve by least effort.
        omega_rows = np.hstack([stab_rows[:, n:], stab_rows[:, :n]]).astype(np.uint8)
        # want v with omega_rows @ v^T = defects
        aug = np.hstack([omega_rows % 2, defects[:, None]]).astype(np.uint8)
        rows_, _ = aug.shape
        r = 0
        pivots = []
        for c in range(2 * n):
            hit = np.flatnonzero(aug[r:, c])
            if hit.size == 0:
                continue
            sel = r + int(hit[0])
            if sel != r:
                aug[[r, sel]] = aug[[sel, r]]
            others = np.flatnonzero(aug[:, c])
            others = others[others != r]
            if others.size:
                aug[others] ^= aug[r]
            pivots.append(c)
            r += 1
            if r == rows_:
                break
        correction_vec = np.zeros(2 * n, dtype=np.uint8)
        for i, c in enumerate(pivots):
            correction_vec[c] = aug[i, -1]
        correction = _row_to_pauli(correction_vec, n)

        # re-verify: corrected gate preserves every generator with sign +1
        all_plus = True
        for index, row in enumerate(stab_rows):
            image = tableau(_row_to_pauli(row, n))
            base = _row_to_pauli(row, n)
            # sign after Pauli correction: correction anticommutes defect away
            anti = 1 if correction.commutes(base) else -1
            corrected_sign = image.sign * anti
            coefficients = _decompose(
                np.concatenate([
                    np.isin(np.arange(n), list(image.pauli_indices("XY"))).astype(np.uint8),
                    np.isin(np.arange(n), list(image.pauli_indices("ZY"))).astype(np.uint8),
                ]),
                stab_rows,
            )
            product = stim.PauliString(n)
            for j in np.flatnonzero(coefficients):
                product = product * _row_to_pauli(stab_rows[j], n)
            if corrected_sign != product.sign:
                all_plus = False

        # sign-exact logical action on Xbar_j (diagonal gates: Xbar -> phase * Xbar Zbar^m)
        phases = []
        for j in range(code.k):
            row = np.concatenate([code.logical_x[j], np.zeros(n, dtype=np.uint8)]) \
                if family == "Z" else np.concatenate([np.zeros(n, dtype=np.uint8), code.logical_z[j]])
            image = tableau(_row_to_pauli(row, n))
            phase = image.sign
            # exponent of i: stim signs are in {1, -1, 1j, -1j}
            exponent = {1: 0, 1j: 1, -1: 2, -1j: 3}[complex(phase)]
            phases.append(exponent)

        certificate = {
            "stabilizer_signs_corrected_to_plus": bool(all_plus),
            "pauli_correction_found": True,
        }
        return PhaseVerifiedGenerator(
            family=family,
            parameter=parameter.copy(),
            circuit=str(circuit).strip(),
            pauli_correction=str(correction),
            logical_diagonal_phases=tuple(phases),
            certificate=certificate,
        )

    @property
    def certified(self) -> bool:
        return all(all(g.certificate.values()) for g in self.generators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generators": [
                {
                    "family": g.family,
                    "circuit": g.circuit,
                    "pauli_correction": g.pauli_correction,
                    "logical_phases_i_exponent": list(g.logical_diagonal_phases),
                    "certificate": g.certificate,
                }
                for g in self.generators
            ],
            "certified": self.certified,
        }


def verify_phases(code: CSSCode) -> PhaseVerification:
    """Sign-exact verification of the strict diagonal generators (needs stim)."""

    return PhaseVerification(code)
