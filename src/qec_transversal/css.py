"""CSS code model and strict-transversal Clifford analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .gf2 import (
    BinaryMatrix,
    as_binary_matrix,
    gf2_inverse,
    gf2_matmul,
    is_symplectic,
    nullspace,
    quotient_complement,
    rank,
    reduce_rows,
    row_basis,
    rowspace_residues,
    rref,
    supports,
    symplectic_product,
)
from .group import generated_group_order, schreier_sims_order, symplectic_group_order

Family = Literal["Z", "X"]

#: Dense certificates are only materialized below this qubit count; above it,
#: the shear structure ``[[I, diag(a)], [0, I]]`` is symplectic identically.
_DENSE_CERTIFICATE_LIMIT = 512


def shear_matrix(family: Family, parameter: BinaryMatrix) -> BinaryMatrix:
    """The dense ``2n x 2n`` symplectic matrix of a transversal shear layer."""

    n = int(parameter.shape[0])
    diagonal = np.diag(parameter.astype(np.uint8))
    identity = np.eye(n, dtype=np.uint8)
    zero = np.zeros((n, n), dtype=np.uint8)
    if family == "Z":
        return np.block([[identity, diagonal], [zero, identity]]).astype(np.uint8)
    return np.block([[identity, zero], [diagonal, identity]]).astype(np.uint8)


def shear_images(
    family: Family, parameter: BinaryMatrix, rows: BinaryMatrix, *, qubits: int
) -> BinaryMatrix:
    """Apply a transversal shear to symplectic row vectors without forming
    the dense ``2n x 2n`` matrix.

    ``U_Z(a)`` maps ``(x | z)`` to ``(x | z + x * a)`` coordinatewise, and
    ``U_X(b)`` maps it to ``(x + z * b | z)``.
    """

    x_part = rows[:, :qubits]
    z_part = rows[:, qubits:]
    if family == "Z":
        return np.hstack([x_part, z_part ^ (x_part & parameter)])
    return np.hstack([x_part ^ (z_part & parameter), z_part])


def _infer_width(matrix: object) -> int | None:
    array = np.asarray(matrix)
    if array.ndim == 2:
        return int(array.shape[1])
    if array.ndim == 1 and array.size:
        return int(array.shape[0])
    return None


def _parameter_space(source: BinaryMatrix, target: BinaryMatrix) -> ParameterSpace:
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
    certificate: dict[str, bool]

    @property
    def physical_symplectic(self) -> BinaryMatrix:
        """The dense ``2n x 2n`` physical matrix, built on demand."""

        return shear_matrix(self.family, self.parameter)

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
    code: CSSCode
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
        group_node_budget: int = 2_000_000,
        include_constraints: bool = False,
        include_physical: bool = False,
    ) -> dict[str, Any]:
        nontrivial = [
            generator.logical_symplectic
            for generator in self.generators
            if not generator.is_logical_identity
        ]
        target_order = symplectic_group_order(self.code.k)

        # Primary engine: exact order through a stabilizer chain, which
        # handles orders far beyond enumeration.  Small results are
        # cross-checked against independent breadth-first closure.
        chain_order = schreier_sims_order(nontrivial, node_budget=group_node_budget)
        cross_checked = False
        if chain_order is not None and chain_order <= min(group_cap, 20_000):
            closure = generated_group_order(nontrivial, cap=min(group_cap, 20_000))
            if closure.exact and closure.order != chain_order:
                raise AssertionError(
                    "internal error: stabilizer chain and explicit closure disagree"
                )
            cross_checked = closure.exact
        if chain_order is not None:
            effective_exact = True
            effective_order: int | None = chain_order
            lower_bound = chain_order
            method = "schreier-sims stabilizer chain"
        else:
            group = generated_group_order(nontrivial, cap=group_cap)
            effective_exact = group.exact
            effective_order = group.order
            lower_bound = group.lower_bound
            method = "explicit closure" if group.exact else "explicit closure (capped)"
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

        # Structural facts from Albert, arXiv:2608.05688.  The action map
        # a -> Sbar(a) is linear, so the logically-trivial subspace dimension
        # is dim A - rank of the stacked shear blocks; the all-ones vector
        # lies in A_Z exactly when C_X is contained in C_Z (and transversal
        # Hadamard exists for connected codes exactly when the code is
        # self-dual).
        k = self.code.k
        z_shears = [
            generator.logical_symplectic[:k, k:].reshape(-1)
            for generator in self.generators
            if generator.family == "Z"
        ]
        x_shears = [
            generator.logical_symplectic[k:, :k].reshape(-1)
            for generator in self.generators
            if generator.family == "X"
        ]
        rank_z_action = rank(np.asarray(z_shears, dtype=np.uint8)) if z_shears else 0
        rank_x_action = rank(np.asarray(x_shears, dtype=np.uint8)) if x_shears else 0
        stacked_constraints = np.vstack([self.a_z.constraints, self.a_x.constraints])
        intersection_dimension = self.code.n - rank(stacked_constraints)
        c_x_subset_c_z = not rowspace_residues(self.code.h_x, self.code.c_z).any()
        c_z_subset_c_x = not rowspace_residues(self.code.h_z, self.code.c_x).any()
        structure = {
            "self_dual": bool(c_x_subset_c_z and c_z_subset_c_x),
            "c_x_subset_c_z": bool(c_x_subset_c_z),
            "c_z_subset_c_x": bool(c_z_subset_c_x),
            "all_ones_in_A_Z": bool(not (self.a_z.constraints.sum(axis=1) & 1).any()),
            "all_ones_in_A_X": bool(not (self.a_x.constraints.sum(axis=1) & 1).any()),
            "dim_A_Z_intersect_A_X": int(intersection_dimension),
            "logically_trivial_dimension_A_Z": int(self.a_z.dimension - rank_z_action),
            "logically_trivial_dimension_A_X": int(self.a_x.dimension - rank_x_action),
            "logically_nontrivial_rank_A_Z": int(rank_z_action),
            "logically_nontrivial_rank_A_X": int(rank_x_action),
        }

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
            "structure": structure,
            "generators": [
                generator.to_dict(include_physical=include_physical)
                for generator in self.generators
            ],
            "logical_group": {
                "nonidentity_generator_count": len(nontrivial),
                "target_symplectic_order": target_order,
                "exact": effective_exact,
                "order": effective_order,
                "lower_bound": lower_bound,
                "is_full_logical_clifford": is_full,
                "method": method,
                "cross_checked_by_enumeration": cross_checked,
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
        self._stabilizer_rref = rref(self.stabilizer)

    def _logical_image(self, family: Family, parameter: BinaryMatrix) -> BinaryMatrix:
        if self.k == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        images = shear_images(family, parameter, self.logical, qubits=self.n)
        x_coefficients = symplectic_product(images, self.logical[self.k :], qubits=self.n)
        z_coefficients = symplectic_product(images, self.logical[: self.k], qubits=self.n)
        logical_image = np.hstack([x_coefficients, z_coefficients]).astype(np.uint8)
        residue = (images ^ gf2_matmul(logical_image, self.logical)) & 1
        if reduce_rows(residue, *self._stabilizer_rref).any():
            raise AssertionError("internal error: logical image has a non-stabilizer residue")
        return logical_image

    def _preserves_stabilizer(self, family: Family, parameter: BinaryMatrix) -> bool:
        image = shear_images(family, parameter, self.stabilizer, qubits=self.n)
        return not reduce_rows(image, *self._stabilizer_rref).any()

    def _generator(self, family: Family, parameter: BinaryMatrix) -> TransversalGenerator:
        if family == "Z":
            shear = gf2_matmul(self.logical_x & parameter, self.logical_x.T)
            formula = np.block(
                [
                    [np.eye(self.k, dtype=np.uint8), shear],
                    [np.zeros((self.k, self.k), dtype=np.uint8), np.eye(self.k, dtype=np.uint8)],
                ]
            ).astype(np.uint8)
        else:
            shear = gf2_matmul(self.logical_z & parameter, self.logical_z.T)
            formula = np.block(
                [
                    [np.eye(self.k, dtype=np.uint8), np.zeros((self.k, self.k), dtype=np.uint8)],
                    [shear, np.eye(self.k, dtype=np.uint8)],
                ]
            ).astype(np.uint8)
        direct = self._logical_image(family, parameter)
        if self.n <= _DENSE_CERTIFICATE_LIMIT:
            physical_symplectic = is_symplectic(shear_matrix(family, parameter), qubits=self.n)
        else:
            # [[I, diag(a)], [0, I]] preserves the form iff diag(a) is
            # symmetric, which holds identically for a diagonal block, so the
            # dense materialization is elided at large n.
            physical_symplectic = True
        certificate = {
            "physical_symplectic": physical_symplectic,
            "stabilizer_preserved": self._preserves_stabilizer(family, parameter),
            "logical_formula_matches_quotient_projection": np.array_equal(formula, direct),
            "logical_symplectic": is_symplectic(formula, qubits=self.k),
        }
        return TransversalGenerator(
            family=family,
            parameter=parameter.copy(),
            logical_symplectic=formula,
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
