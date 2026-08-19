"""Fold-transversal (fixed-matching) Clifford analysis.

Fix an involution ``tau`` on the physical qubits.  A *Z-type matching layer*
is a depth-one diagonal circuit

    U = prod_{(i,j) in pairs(tau)} CZ_ij^{c_ij} * prod_i sqrt(Z)_i^{a_i},

whose symplectic action is the shear ``[[I, Sigma], [0, I]]`` with ``Sigma``
the symmetric matrix carrying ``a`` on the diagonal and ``c`` on the matched
off-diagonal positions.  Stabilizer preservation is linear in ``(a, c)``, so
the complete family is a GF(2) kernel — the fixed-matching analogue of the
strict parameter code ``A_Z`` (Albert, arXiv:2608.05688, the ``S_M^Z``
family).  The X-type family is the transpose construction.

When ``tau`` is additionally a ZX-duality (it maps ``C_X`` onto ``C_Z``),
the fold-Hadamard gate — transversal H followed by the permutation — is a
further generator, and is certified here directly from its symplectic
action.

Scope: this module classifies the *diagonal* layers on the given matching
completely, plus the fold-Hadamard.  It does not enumerate arbitrary
two-qubit Cliffords on the matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .css import CSSCode, Family
from .gf2 import (
    BinaryMatrix,
    gf2_matmul,
    is_symplectic,
    nullspace,
    reduce_rows,
    row_basis,
    rowspace_residues,
    symplectic_product,
)
from .group import generated_group_order, schreier_sims_order, symplectic_group_order

#: Below this k the closure fallback runs at the caller's full cap; above it
#: the fallback is kept token-sized so a big-k sweep can never stall there.
_GROUP_K_LIMIT = 14

#: Schreier-Sims node budget for the k > _GROUP_K_LIMIT attempt.  The chain
#: is tried at EVERY k — its cost is bounded by this transversal-point budget,
#: not by the dimension, and moderate groups stay exact where the old
#: k-gate silently degraded them to a cap-200 closure (the Kasai
#: [[56,18,4]] depth-one group, order 1,524,096 at k = 18, certifies in
#: seconds; under the gate it reported "lower bound 201").
_BIG_K_NODE_BUDGET = 1_000_000


def involution_pairs(tau: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    """Split an involution into matched pairs (i < j) and fixed points."""

    tau = np.asarray(tau, dtype=int)
    n = tau.shape[0]
    if sorted(tau.tolist()) != list(range(n)):
        raise ValueError("tau must be a permutation")
    if not np.array_equal(tau[tau], np.arange(n)):
        raise ValueError("tau must be an involution")
    pairs = [(i, int(tau[i])) for i in range(n) if tau[i] > i]
    fixed = [i for i in range(n) if tau[i] == i]
    return pairs, fixed


def sigma_matrix(parameter: BinaryMatrix, pairs: list[tuple[int, int]], n: int) -> BinaryMatrix:
    """The symmetric shear block for a parameter vector ``(a | c)``."""

    sigma = np.diag(parameter[:n].astype(np.uint8))
    for index, (i, j) in enumerate(pairs):
        sigma[i, j] = sigma[j, i] = parameter[n + index]
    return sigma


def _sigma_apply(
    parameter: BinaryMatrix, pairs: list[tuple[int, int]], rows: BinaryMatrix
) -> BinaryMatrix:
    """``rows @ Sigma`` computed sparsely (Sigma is symmetric)."""

    n = rows.shape[1]
    out = rows & parameter[:n]
    for index, (i, j) in enumerate(pairs):
        if parameter[n + index]:
            out[:, i] ^= rows[:, j]
            out[:, j] ^= rows[:, i]
    return out


def _matching_kernel(
    source: BinaryMatrix,
    target: BinaryMatrix,
    pairs: list[tuple[int, int]],
    tau: np.ndarray,
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Kernel of the fixed-matching constraints.

    Parameters ``p = (a | c)`` with ``a`` over qubits and ``c`` over pairs;
    for every source check ``v`` the vector ``Sigma(p) v^T`` must lie in the
    row span of ``target``.  Returns ``(basis, constraints)``.
    """

    n = source.shape[1]
    width = n + len(pairs)
    pair_index = {}
    for index, (i, j) in enumerate(pairs):
        pair_index[i] = (index, j)
        pair_index[j] = (index, i)
    target_perp = nullspace(target)
    constraints: list[BinaryMatrix] = []
    for check in source:
        support = np.flatnonzero(check)
        if support.size == 0:
            continue
        region = np.unique(np.concatenate([support, tau[support]]))
        projected = target_perp[:, region]
        local_basis = row_basis(projected, ncols=len(region))
        for local_row in local_basis:
            row = np.zeros(width, dtype=np.uint8)
            for local_col, qubit in enumerate(region):
                if not local_row[local_col]:
                    continue
                # (Sigma v)_qubit = a_qubit v_qubit + c_pair v_partner
                if check[qubit]:
                    row[qubit] ^= 1
                if qubit in pair_index:
                    index, partner = pair_index[qubit]
                    if check[partner]:
                        row[n + index] ^= 1
            if row.any():
                constraints.append(row)
    if constraints:
        constraint_matrix = row_basis(np.asarray(constraints, dtype=np.uint8))
    else:
        constraint_matrix = np.zeros((0, width), dtype=np.uint8)
    return nullspace(constraint_matrix), constraint_matrix


@dataclass(frozen=True)
class MatchingGenerator:
    family: Family
    parameter: BinaryMatrix  # (a | c) over n + len(pairs)
    logical_symplectic: BinaryMatrix
    certificate: dict[str, bool]

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )


@dataclass(frozen=True)
class FoldHadamard:
    logical_symplectic: BinaryMatrix
    certificate: dict[str, bool]

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )


class MatchingAnalysis:
    """Complete diagonal fold-layer analysis for one involution ``tau``."""

    def __init__(self, code: CSSCode, tau: object):
        self.code = code
        self.tau = np.asarray(tau, dtype=int)
        if self.tau.shape != (code.n,):
            raise ValueError("tau must assign an image to every qubit")
        self.pairs, self.fixed = involution_pairs(self.tau)
        self.perm = np.eye(code.n, dtype=np.uint8)[self.tau]

        # tau is a ZX-duality when column-permuting C_X gives exactly C_Z.
        image = row_basis(gf2_matmul(code.c_x, self.perm), ncols=code.n)
        self.is_zx_duality = (
            image.shape[0] == code.c_z.shape[0]
            and not rowspace_residues(image, code.c_z).any()
        )

        self.z_basis, self.z_constraints = _matching_kernel(
            code.h_x, code.h_z, self.pairs, self.tau
        )
        self.x_basis, self.x_constraints = _matching_kernel(
            code.h_z, code.h_x, self.pairs, self.tau
        )
        self.generators = tuple(
            [self._generator("Z", parameter) for parameter in self.z_basis]
            + [self._generator("X", parameter) for parameter in self.x_basis]
        )
        self.fold_hadamard = self._fold_hadamard() if self.is_zx_duality else None

    # -- diagonal layers ---------------------------------------------------

    def _shear_rows(self, family: Family, parameter: BinaryMatrix, rows: BinaryMatrix):
        n = self.code.n
        x_part, z_part = rows[:, :n].copy(), rows[:, n:].copy()
        if family == "Z":
            return np.hstack([x_part, z_part ^ _sigma_apply(parameter, self.pairs, x_part)])
        return np.hstack([x_part ^ _sigma_apply(parameter, self.pairs, z_part), z_part])

    def _logical_image_rows(self, images: BinaryMatrix) -> BinaryMatrix:
        code = self.code
        x_coefficients = symplectic_product(images, code.logical[code.k :], qubits=code.n)
        z_coefficients = symplectic_product(images, code.logical[: code.k], qubits=code.n)
        logical_image = np.hstack([x_coefficients, z_coefficients]).astype(np.uint8)
        residue = (images ^ gf2_matmul(logical_image, code.logical)) & 1
        if reduce_rows(residue, *code._stabilizer_rref).any():
            raise AssertionError("internal error: image has a non-stabilizer residue")
        return logical_image

    def _generator(self, family: Family, parameter: BinaryMatrix) -> MatchingGenerator:
        code = self.code
        basis = code.logical_x if family == "Z" else code.logical_z
        shear = gf2_matmul(_sigma_apply(parameter, self.pairs, basis), basis.T)
        identity = np.eye(code.k, dtype=np.uint8)
        zero = np.zeros((code.k, code.k), dtype=np.uint8)
        if family == "Z":
            formula = np.block([[identity, shear], [zero, identity]]).astype(np.uint8)
        else:
            formula = np.block([[identity, zero], [shear, identity]]).astype(np.uint8)

        stab_images = self._shear_rows(family, parameter, code.stabilizer)
        preserved = not reduce_rows(stab_images, *code._stabilizer_rref).any()
        if code.k:
            direct_images = self._shear_rows(family, parameter, code.logical)
            direct = self._logical_image_rows(direct_images)
        else:
            direct = formula
        certificate = {
            "stabilizer_preserved": bool(preserved),
            "logical_formula_matches_quotient_projection": bool(np.array_equal(formula, direct)),
            "logical_symplectic": is_symplectic(formula, qubits=code.k),
        }
        return MatchingGenerator(
            family=family,
            parameter=parameter.copy(),
            logical_symplectic=formula,
            certificate=certificate,
        )

    # -- fold Hadamard -----------------------------------------------------

    def _fold_hadamard(self) -> FoldHadamard:
        code = self.code
        n = code.n

        def apply(rows: BinaryMatrix) -> BinaryMatrix:
            x_part, z_part = rows[:, :n], rows[:, n:]
            return np.hstack(
                [gf2_matmul(z_part, self.perm), gf2_matmul(x_part, self.perm)]
            )

        stab_images = apply(code.stabilizer)
        preserved = not reduce_rows(stab_images, *code._stabilizer_rref).any()
        if code.k:
            logical = self._logical_image_rows(apply(code.logical))
        else:
            logical = np.zeros((0, 0), dtype=np.uint8)
        certificate = {
            "zx_duality_certified": bool(self.is_zx_duality),
            "stabilizer_preserved": bool(preserved),
            "logical_symplectic": is_symplectic(logical, qubits=code.k),
        }
        return FoldHadamard(logical_symplectic=logical, certificate=certificate)

    # -- summary -----------------------------------------------------------

    @property
    def certified(self) -> bool:
        checks = all(
            all(generator.certificate.values()) for generator in self.generators
        )
        if self.fold_hadamard is not None:
            checks = checks and all(self.fold_hadamard.certificate.values())
        kernels = (
            not (self.z_constraints @ self.z_basis.T & 1).any()
            and not (self.x_constraints @ self.x_basis.T & 1).any()
        )
        return bool(checks and kernels)

    def to_dict(self, *, group_cap: int = 100_000) -> dict[str, Any]:
        code = self.code
        nontrivial = [
            generator.logical_symplectic
            for generator in self.generators
            if not generator.is_logical_identity
        ]
        fold_h_nontrivial = bool(
            self.fold_hadamard is not None and not self.fold_hadamard.is_logical_identity
        )
        if fold_h_nontrivial:
            nontrivial.append(self.fold_hadamard.logical_symplectic)

        group = logical_group_summary(nontrivial, code.k, group_cap=group_cap)

        z_trivial = sum(
            1 for g in self.generators if g.family == "Z" and g.is_logical_identity
        )
        x_trivial = sum(
            1 for g in self.generators if g.family == "X" and g.is_logical_identity
        )
        return {
            "pairs": len(self.pairs),
            "fixed_points": len(self.fixed),
            "is_zx_duality": bool(self.is_zx_duality),
            "dim_S_MZ": int(self.z_basis.shape[0]),
            "dim_S_MX": int(self.x_basis.shape[0]),
            "logically_trivial_Z": int(z_trivial),
            "logically_trivial_X": int(x_trivial),
            "nontrivial_generator_count": len(nontrivial),
            "fold_hadamard_nontrivial": fold_h_nontrivial,
            "logical_group": group,
            # embedding |Sp(2k,2)| verbatim for k in the hundreds would bloat
            # serialized reports with 10^5-digit integers
            "target_symplectic_order": (
                symplectic_group_order(code.k) if code.k <= 32 else None
            ),
            "target_symplectic_order_digits": len(str(symplectic_group_order(code.k))),
            "certified": self.certified,
        }


def logical_group_summary(
    generators: list[BinaryMatrix], k: int, *, group_cap: int = 100_000
) -> dict[str, Any]:
    """Order of the group generated by logical symplectic generators.

    Schreier-Sims for moderate ``k``; above that a tightly capped closure is
    still attempted because small groups (the common case) close quickly
    even for very large matrices.
    """

    if k == 0 or not generators:
        return {"computed": True, "exact": True, "order": 1, "lower_bound": 1}
    # The chain is tried at every k: its cost is governed by the node budget,
    # not the dimension, and it is the only route to an exact order once the
    # group outgrows any affordable closure cap.
    node_budget = 2_000_000 if k <= _GROUP_K_LIMIT else _BIG_K_NODE_BUDGET
    chain = schreier_sims_order(generators, node_budget=node_budget)
    if chain is not None:
        return {"computed": True, "exact": True, "order": chain, "lower_bound": chain}
    if k <= _GROUP_K_LIMIT:
        cap = group_cap
    else:
        # chain over budget at very large k: keep the closure attempt cheap
        # enough that a pathological generating set cannot stall a sweep
        cap = min(group_cap, 200)
    closure = generated_group_order(generators, cap=cap)
    return {
        "computed": True,
        "exact": closure.exact,
        "order": closure.order,
        "lower_bound": closure.lower_bound,
    }


def analyze_matching(code: CSSCode, tau: object) -> MatchingAnalysis:
    """Complete diagonal fold-layer analysis of ``code`` on involution ``tau``."""

    return MatchingAnalysis(code, tau)
