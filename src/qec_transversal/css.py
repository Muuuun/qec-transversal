"""CSS code model and strict-transversal Clifford analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .gf2 import (
    BinaryMatrix,
    as_binary_matrix,
    gf2_inverse,
    is_in_rowspace,
    is_symplectic,
    nullspace,
    quotient_complement,
    rank,
    row_basis,
    supports,
    symplectic_product,
)
from .group import generated_group_order, symplectic_group_order

Family = Literal["Z", "X"]


def _infer_width(matrix: object) -> int | None:
    array = np.asarray(matrix)
    if array.ndim == 2:
        return int(array.shape[1])
    if array.ndim == 1 and array.size:
        return int(array.shape[0])
    return None


def _parameter_space(source: BinaryMatrix, target: BinaryMatrix) -> "ParameterSpace":
    """Compute ``{a : a * source_code is contained in target_code}``.

    For each low-weight source check, only the restriction of ``target^perp``
    to its support is row-reduced.  Consequently that check contributes at
    most its weight in independent constraints.
    """

    n = source.shape[1]
    target_perp = nullspace(target)
    constraints: list[BinaryMatrix] = []
    for check in source:
        support = np.flatnonzero(check)
        if support.size == 0:
            continue
        projected = target_perp[:, support]
        local_basis = row_basis(projected, ncols=len(support))
        for local_row in local_basis:
            full_row = np.zeros(n, dtype=np.uint8)
            full_row[support] = local_row
            constraints.append(full_row)
    if constraints:
        constraint_matrix = row_basis(np.asarray(constraints, dtype=np.uint8))
    else:
        constraint_matrix = np.zeros((0, n), dtype=np.uint8)
    return ParameterSpace(basis=nullspace(constraint_matrix), constraints=constraint_matrix)


@dataclass(frozen=True)
class ParameterSpace:
    basis: BinaryMatrix
    constraints: BinaryMatrix

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])


@dataclass(frozen=True)
class TransversalGenerator:
    family: Family
    parameter: BinaryMatrix
    logical_symplectic: BinaryMatrix
    physical_symplectic: BinaryMatrix
    certificate: dict[str, bool]

    @property
    def support(self) -> list[int]:
        return np.flatnonzero(self.parameter).astype(int).tolist()

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )

    def to_dict(self, *, include_physical: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "family": self.family,
            "gate": "sqrt_Z" if self.family == "Z" else "sqrt_X",
            "parameter": self.parameter.astype(int).tolist(),
            "support": self.support,
            "logical_symplectic": self.logical_symplectic.astype(int).tolist(),
            "logical_identity": self.is_logical_identity,
            "certificate": self.certificate,
        }
        if include_physical:
            result["physical_symplectic"] = self.physical_symplectic.astype(int).tolist()
        return result


@dataclass(frozen=True)
class TransversalAnalysis:
    code: "CSSCode"
    a_z: ParameterSpace
    a_x: ParameterSpace
    generators: tuple[TransversalGenerator, ...]

    @property
    def certified(self) -> bool:
        parameter_checks = (
            not (self.a_z.constraints @ self.a_z.basis.T & 1).any()
            and not (self.a_x.constraints @ self.a_x.basis.T & 1).any()
        )
        generator_checks = all(all(generator.certificate.values()) for generator in self.generators)
        logical_pairing = np.array_equal(
            (self.code.logical_x @ self.code.logical_z.T) & 1,
            np.eye(self.code.k, dtype=np.uint8),
        )
        return bool(parameter_checks and generator_checks and logical_pairing)

    def to_dict(
        self,
        *,
        group_cap: int = 100_000,
        include_constraints: bool = False,
        include_physical: bool = False,
    ) -> dict[str, Any]:
        nontrivial = [
            generator.logical_symplectic
            for generator in self.generators
            if not generator.is_logical_identity
        ]
        group = generated_group_order(nontrivial, cap=group_cap)
        target_order = symplectic_group_order(self.code.k)
        reached_target_bound = not group.exact and group.lower_bound == target_order
        effective_exact = group.exact or reached_target_bound
        effective_order = target_order if reached_target_bound else group.order
        is_full: bool | None
        if effective_exact:
            is_full = effective_order == target_order
        else:
            is_full = None

        parameter_spaces: dict[str, Any] = {
            "A_Z": {
                "dimension": self.a_z.dimension,
                "basis": self.a_z.basis.astype(int).tolist(),
                "supports": supports(self.a_z.basis),
            },
            "A_X": {
                "dimension": self.a_x.dimension,
                "basis": self.a_x.basis.astype(int).tolist(),
                "supports": supports(self.a_x.basis),
            },
        }
        if include_constraints:
            parameter_spaces["A_Z"]["constraints"] = self.a_z.constraints.astype(int).tolist()
            parameter_spaces["A_X"]["constraints"] = self.a_x.constraints.astype(int).tolist()

        css_orthogonality = not ((self.code.h_x @ self.code.h_z.T) & 1).any()
        logical_pairing = np.array_equal(
            (self.code.logical_x @ self.code.logical_z.T) & 1,
            np.eye(self.code.k, dtype=np.uint8),
        )
        parameter_nullspaces = (
            not (self.a_z.constraints @ self.a_z.basis.T & 1).any()
            and not (self.a_x.constraints @ self.a_x.basis.T & 1).any()
        )
        generators_verified = all(
            all(generator.certificate.values()) for generator in self.generators
        )

        return {
            "schema_version": "0.1",
            "scope": "CSS strict-transversal Clifford group modulo Pauli phases",
            "conventions": {
                "vectors": "row",
                "physical_coordinates": "X_0..X_(n-1) | Z_0..Z_(n-1)",
                "logical_coordinates": "Xbar_0..Xbar_(k-1) | Zbar_0..Zbar_(k-1)",
            },
            "code": {
                "n": self.code.n,
                "k": self.code.k,
                "rank_X": self.code.rank_x,
                "rank_Z": self.code.rank_z,
            },
            "logical_basis": {
                "X": self.code.logical_x.astype(int).tolist(),
                "Z": self.code.logical_z.astype(int).tolist(),
            },
            "parameter_spaces": parameter_spaces,
            "generators": [
                generator.to_dict(include_physical=include_physical)
                for generator in self.generators
            ],
            "logical_group": {
                "nonidentity_generator_count": len(nontrivial),
                "target_symplectic_order": target_order,
                "exact": effective_exact,
                "order": effective_order,
                "lower_bound": group.lower_bound,
                "is_full_logical_clifford": is_full,
                "method": (
                    "explicit closure"
                    if group.exact
                    else "symplectic ambient-order bound"
                    if reached_target_bound
                    else "explicit closure (capped)"
                ),
                "cap": group_cap,
            },
            "certificate": {
                "css_orthogonality": bool(css_orthogonality),
                "logical_pairing_canonical": bool(logical_pairing),
                "parameter_nullspaces_verified": bool(parameter_nullspaces),
                "all_generators_verified": bool(generators_verified),
                "certified": self.certified,
            },
        }


class CSSCode:
    """A binary CSS stabilizer code specified by X- and Z-check rows."""

    def __init__(self, h_x: object, h_z: object, *, n: int | None = None):
        inferred_x = _infer_width(h_x)
        inferred_z = _infer_width(h_z)
        widths = {width for width in (inferred_x, inferred_z, n) if width is not None}
        if not widths:
            raise ValueError("cannot infer n from two empty check matrices")
        if len(widths) != 1:
            raise ValueError(f"inconsistent physical-qubit counts: {sorted(widths)}")
        self.n = widths.pop()
        if self.n <= 0:
            raise ValueError("n must be positive")

        self.h_x = as_binary_matrix(h_x, ncols=self.n)
        self.h_z = as_binary_matrix(h_z, ncols=self.n)
        if ((self.h_x @ self.h_z.T) & 1).any():
            raise ValueError("invalid CSS checks: H_X H_Z^T is nonzero over GF(2)")

        self.c_x = row_basis(self.h_x, ncols=self.n)
        self.c_z = row_basis(self.h_z, ncols=self.n)
        self.rank_x = rank(self.c_x)
        self.rank_z = rank(self.c_z)
        self.k = self.n - self.rank_x - self.rank_z
        if self.k < 0:
            raise ValueError("check ranks exceed n")

        normalizer_x = nullspace(self.h_z)
        normalizer_z = nullspace(self.h_x)
        self.logical_x = quotient_complement(normalizer_x, self.c_x)
        logical_z_unpaired = quotient_complement(normalizer_z, self.c_z)
        if self.logical_x.shape[0] != self.k or logical_z_unpaired.shape[0] != self.k:
            raise ValueError("failed to construct the expected number of logical operators")

        pairing = (self.logical_x @ logical_z_unpaired.T) & 1
        self.logical_z = (gf2_inverse(pairing).T @ logical_z_unpaired) & 1
        if not np.array_equal(
            (self.logical_x @ self.logical_z.T) & 1,
            np.eye(self.k, dtype=np.uint8),
        ):
            raise AssertionError("internal error: logical basis is not symplectically paired")

        zeros_x = np.zeros((self.c_x.shape[0], self.n), dtype=np.uint8)
        zeros_z = np.zeros((self.c_z.shape[0], self.n), dtype=np.uint8)
        self.stabilizer = np.vstack(
            [
                np.hstack([self.c_x, zeros_x]),
                np.hstack([zeros_z, self.c_z]),
            ]
        )
        self.logical = np.vstack(
            [
                np.hstack([self.logical_x, np.zeros_like(self.logical_x)]),
                np.hstack([np.zeros_like(self.logical_z), self.logical_z]),
            ]
        )

    def _physical_gate(self, family: Family, parameter: BinaryMatrix) -> BinaryMatrix:
        diagonal = np.diag(parameter.astype(np.uint8))
        identity = np.eye(self.n, dtype=np.uint8)
        zero = np.zeros((self.n, self.n), dtype=np.uint8)
        if family == "Z":
            return np.block([[identity, diagonal], [zero, identity]]).astype(np.uint8)
        return np.block([[identity, zero], [diagonal, identity]]).astype(np.uint8)

    def _logical_image(self, physical: BinaryMatrix) -> BinaryMatrix:
        if self.k == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        images = (self.logical @ physical) & 1
        x_coefficients = symplectic_product(images, self.logical[self.k :], qubits=self.n)
        z_coefficients = symplectic_product(images, self.logical[: self.k], qubits=self.n)
        logical_image = np.hstack([x_coefficients, z_coefficients]).astype(np.uint8)
        residue = (images ^ ((logical_image @ self.logical) & 1)) & 1
        if any(not is_in_rowspace(row, self.stabilizer) for row in residue):
            raise AssertionError("internal error: logical image has a non-stabilizer residue")
        return logical_image

    def _preserves_stabilizer(self, physical: BinaryMatrix) -> bool:
        image = (self.stabilizer @ physical) & 1
        return all(is_in_rowspace(row, self.stabilizer) for row in image)

    def _generator(self, family: Family, parameter: BinaryMatrix) -> TransversalGenerator:
        physical = self._physical_gate(family, parameter)
        if family == "Z":
            shear = (self.logical_x @ np.diag(parameter) @ self.logical_x.T) & 1
            formula = np.block(
                [
                    [np.eye(self.k, dtype=np.uint8), shear],
                    [np.zeros((self.k, self.k), dtype=np.uint8), np.eye(self.k, dtype=np.uint8)],
                ]
            ).astype(np.uint8)
        else:
            shear = (self.logical_z @ np.diag(parameter) @ self.logical_z.T) & 1
            formula = np.block(
                [
                    [np.eye(self.k, dtype=np.uint8), np.zeros((self.k, self.k), dtype=np.uint8)],
                    [shear, np.eye(self.k, dtype=np.uint8)],
                ]
            ).astype(np.uint8)
        direct = self._logical_image(physical)
        certificate = {
            "physical_symplectic": is_symplectic(physical, qubits=self.n),
            "stabilizer_preserved": self._preserves_stabilizer(physical),
            "logical_formula_matches_quotient_projection": np.array_equal(formula, direct),
            "logical_symplectic": is_symplectic(formula, qubits=self.k),
        }
        return TransversalGenerator(
            family=family,
            parameter=parameter.copy(),
            logical_symplectic=formula,
            physical_symplectic=physical,
            certificate=certificate,
        )

    def analyze_transversal(self) -> TransversalAnalysis:
        """Find all strict-transversal Clifford parameter-space generators."""

        a_z = _parameter_space(self.h_x, self.h_z)
        a_x = _parameter_space(self.h_z, self.h_x)
        generators = tuple(
            [self._generator("Z", parameter) for parameter in a_z.basis]
            + [self._generator("X", parameter) for parameter in a_x.basis]
        )
        return TransversalAnalysis(code=self, a_z=a_z, a_x=a_x, generators=generators)
