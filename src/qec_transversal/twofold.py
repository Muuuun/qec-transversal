"""The two-fold-transversal group: depth-one two-local layers over matchings.

Albert (arXiv:2608.05688) defines ``N_2fold = < union over matchings M of
N_M >`` and proves (Thm. D.1) that each fixed-matching group is generated
by the diagonal families ``S_M^Z``, ``S_M^X`` together with the Levi
(CNOT-network) factor ``L_M``.  This module assembles those generators —
the diagonal families from :mod:`.matching`, the Levi units from a
dedicated algebra kernel — and grows the logical group over sampled
matchings until it reaches the full symplectic target (a positive
certificate of fullness) or stops growing (an honest lower bound; sampling
never proves emptiness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .css import CSSCode
from .gf2 import gf2_inverse, nullspace, row_basis, rowspace_residues
from .group import schreier_sims_order, symplectic_group_order
from .matching import analyze_matching

_LEVI_DIM_CAP = 12


def levi_logical_generators(code: CSSCode, tau: np.ndarray) -> list[np.ndarray]:
    """Logical actions of the Levi factor ``L_M`` on matching ``tau``.

    A Levi element acts as ``x -> x K`` on X labels and ``z -> z K^{-T}`` on
    Z labels with ``K`` supported on the matching's cells; validity is the
    pair of *linear* conditions ``C_X K \\subseteq C_X`` and
    ``C_Z K^T \\subseteq C_Z`` (for invertible ``K`` the second is
    equivalent to ``C_Z K^{-T} \\subseteq C_Z``).  The valid ``K`` form a
    matrix algebra computed by one kernel; its invertible elements are
    enumerated within a dimension cap.
    """

    n = code.n
    tau = np.asarray(tau, dtype=int)
    # unknown entries: diagonal everywhere + the two off-diagonal slots of
    # every matched pair
    slots: list[tuple[int, int]] = [(i, i) for i in range(n)]
    for i in range(n):
        j = int(tau[i])
        if j != i:
            slots.append((i, j))
    slot_index = {slot: s for s, slot in enumerate(slots)}
    width = len(slots)

    def constraint_rows(space: np.ndarray, transpose: bool) -> list[np.ndarray]:
        rows = []
        perp = nullspace(space)
        for v in space:
            support = np.flatnonzero(v)
            if support.size == 0:
                continue
            region = np.unique(np.concatenate([support, tau[support]]))
            local = row_basis(perp[:, region], ncols=len(region))
            for w_local in local:
                w = np.zeros(n, dtype=np.uint8)
                w[region] = w_local
                row = np.zeros(width, dtype=np.uint8)
                for (a, b), s in slot_index.items():
                    # (v K)_b includes v_a K[a,b]; dual w picks w_b
                    if transpose:
                        # v K^T: (v K^T)_a includes v_b K[a,b]
                        if v[b] and w[a]:
                            row[s] ^= 1
                    else:
                        if v[a] and w[b]:
                            row[s] ^= 1
                if row.any():
                    rows.append(row)
        return rows

    rows = constraint_rows(code.c_x, transpose=False) + constraint_rows(
        code.c_z, transpose=True
    )
    if rows:
        constraint_matrix = row_basis(np.asarray(rows, dtype=np.uint8))
    else:
        constraint_matrix = np.zeros((0, width), dtype=np.uint8)
    algebra = nullspace(constraint_matrix)
    if algebra.shape[0] > _LEVI_DIM_CAP:
        # honest cap: enumerate only the K = I + single-basis-element
        # perturbations (still certified individually below)
        candidates = [algebra[i] for i in range(algebra.shape[0])]
        identity_entries = np.zeros(width, dtype=np.uint8)
        for i in range(n):
            identity_entries[slot_index[(i, i)]] = 1
        masks = []
        for c in candidates:
            masks.append((identity_entries ^ c) if not np.array_equal(c, identity_entries) else c)
        return _levi_from_entries(code, tau, slots, slot_index, masks)

    all_entries = []
    for mask in range(1, 1 << algebra.shape[0]):
        entries = np.zeros(width, dtype=np.uint8)
        for bit in range(algebra.shape[0]):
            if (mask >> bit) & 1:
                entries ^= algebra[bit]
        all_entries.append(entries)
    return _levi_from_entries(code, tau, slots, slot_index, all_entries)


def _levi_from_entries(code, tau, slots, slot_index, entry_list):
    n = code.n
    logicals: list[np.ndarray] = []
    for entries in entry_list:
        K = np.zeros((n, n), dtype=np.uint8)
        for (a, b), s in slot_index.items():
            K[a, b] = entries[s]
        # invertibility cell by cell
        ok = True
        for i in range(n):
            j = int(tau[i])
            if j == i:
                if not K[i, i]:
                    ok = False
                    break
            elif j > i:
                det = (K[i, i] & K[j, j]) ^ (K[i, j] & K[j, i])
                if not det:
                    ok = False
                    break
        if not ok:
            continue
        K_inv_T = gf2_inverse(K).T
        big = np.zeros((2 * n, 2 * n), dtype=np.uint8)
        big[:n, :n] = K
        big[n:, n:] = K_inv_T
        # certify stabilizer preservation, then extract the logical action
        image = (code.stabilizer.astype(np.int64) @ big.astype(np.int64) % 2).astype(
            np.uint8
        )
        if rowspace_residues(image, code.stabilizer).any():
            continue
        if code.k:
            from .gf2 import gf2_matmul, symplectic_product

            images = gf2_matmul(code.logical, big)
            x_c = symplectic_product(images, code.logical[code.k :], qubits=n)
            z_c = symplectic_product(images, code.logical[: code.k], qubits=n)
            logical = np.hstack([x_c, z_c]).astype(np.uint8)
            if not np.array_equal(logical, np.eye(2 * code.k, dtype=np.uint8)):
                logicals.append(logical)
    return logicals


def _random_involution(rng: np.random.Generator, n: int) -> np.ndarray:
    order = rng.permutation(n)
    tau = np.arange(n)
    for index in range(0, n - 1, 2):
        i, j = int(order[index]), int(order[index + 1])
        tau[i], tau[j] = j, i
    return tau


@dataclass(frozen=True)
class TwoFoldResult:
    logical_order: int | None
    lower_bound: int
    is_full: bool
    matchings_used: int
    saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_order": self.logical_order,
            "lower_bound": self.lower_bound,
            "is_full": self.is_full,
            "matchings_used": self.matchings_used,
            "saturated": self.saturated,
        }


def automorphism_involutions(
    aut_generators: list[np.ndarray], n: int, *, count: int, seed: int
) -> list[np.ndarray]:
    """Involutions sampled from the automorphism group (random words)."""

    rng = np.random.default_rng(seed)
    found: list[np.ndarray] = []
    seen: set[bytes] = set()
    identity = np.arange(n)
    for generator in aut_generators:  # generators themselves first
        candidate = np.asarray(generator, dtype=int)
        if not np.array_equal(candidate, identity) and np.array_equal(
            candidate[candidate], identity
        ):
            key = candidate.tobytes()
            if key not in seen:
                seen.add(key)
                found.append(candidate)
    tries = 0
    while len(found) < count and tries < 60 * count and aut_generators:
        tries += 1
        word = identity.copy()
        for _ in range(int(rng.integers(1, 6))):
            g = aut_generators[int(rng.integers(0, len(aut_generators)))]
            word = np.asarray(g, dtype=int)[word]
        if np.array_equal(word, identity):
            continue
        if not np.array_equal(word[word], identity):
            # square down: word^2 has halved order; try its odd-power core
            word = word[word]
            if np.array_equal(word, identity) or not np.array_equal(
                word[word], identity
            ):
                continue
        key = word.tobytes()
        if key not in seen:
            seen.add(key)
            found.append(word)
    return found[:count]


def two_fold_group(
    code: CSSCode,
    *,
    rounds: int = 40,
    plateau: int = 8,
    seed: int = 7,
    include_levi: bool = True,
    extra_taus: list[np.ndarray] | None = None,
    aut_generators: list[np.ndarray] | None = None,
) -> TwoFoldResult:
    """Grow the logical image of ``N_2fold`` over sampled matchings.

    Fullness (order equals ``|Sp(2k,2)|``) is a positive certificate; a
    plateau is only a lower bound — sampling cannot prove absence.
    """

    rng = np.random.default_rng(seed)
    target = symplectic_group_order(code.k)
    generators: list[np.ndarray] = []
    seen: set[bytes] = set()
    order = 1
    quiet = 0
    used = 0
    queue: list[np.ndarray] = list(extra_taus or [])
    if aut_generators:
        queue.extend(
            automorphism_involutions(aut_generators, code.n, count=rounds // 2, seed=seed)
        )
    for round_index in range(rounds):
        tau = queue.pop(0) if queue else _random_involution(rng, code.n)
        analysis = analyze_matching(code, tau)
        new = [
            g.logical_symplectic
            for g in analysis.generators
            if not g.is_logical_identity
        ]
        if analysis.fold_hadamard is not None and not analysis.fold_hadamard.is_logical_identity:
            new.append(analysis.fold_hadamard.logical_symplectic)
        if include_levi:
            new.extend(levi_logical_generators(code, tau))
        used += 1
        fresh = False
        for g in new:
            key = g.tobytes()
            if key not in seen:
                seen.add(key)
                generators.append(g)
                fresh = True
        if not fresh:
            quiet += 1
            if quiet >= plateau:
                break
            continue
        new_order = schreier_sims_order(generators)
        if new_order is None:
            return TwoFoldResult(None, order, False, used, False)
        if new_order == order:
            quiet += 1
            if quiet >= plateau:
                break
        else:
            order = new_order
            quiet = 0
        if order == target:
            return TwoFoldResult(order, order, True, used, True)
    return TwoFoldResult(order, order, order == target, used, False)
