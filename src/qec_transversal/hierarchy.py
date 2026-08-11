"""Transversal diagonal gates in the third Clifford hierarchy level.

A strict-transversal diagonal layer at level three is ``U(t) = prod_i
T_i^{t_i}`` with ``t in Z_8^n`` (``T = diag(1, exp(i pi/4))``; ``t_i = 2``
is S, ``t_i = 4`` is Z, so the family subsumes the diagonal Cliffords and
Pauli dressing).  Acting on a CSS codeword ``|g>`` (a superposition over the
coset ``g + C_X``), the layer multiplies the branch ``u`` by ``omega^{t.u}``
with ``omega = exp(i pi/4)``.  ``U(t)`` preserves the code space exactly
when the branch phase is constant on every coset of ``C_X`` inside
``ker H_Z``, and then acts as the *logical diagonal gate*
``phi(g) = t.g mod 8``.

Using ``t.(u xor v) = t.u + t.v - 2 t.(u & v)`` the coset-constancy
condition closes on basis vectors into finitely many linear congruences:

- ``t . v = 0 (mod 8)``          for each X-check basis vector ``v``,
- ``t . (g & v) = 0 (mod 4)``    for ``g`` in a basis of ``ker H_Z``,
- ``t . (g & g' & v) = 0 (mod 2)`` and ``t . (g & v & v') = 0 (mod 2)``.

The solution set is a subgroup of ``Z_8^n`` computed exactly by module
elimination over ``Z_8``; the logical action ``phi`` is linear in ``t``, so
generators decide nontriviality and hierarchy level.  The X-type family
(powers of ``sqrt(X)``-hierarchy gates) is the same computation with X and
Z exchanged.  This certifies the *strict single-qubit diagonal* class at
level three completely; two-local diagonal layers (CS/CCZ across qubits)
are outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .css import CSSCode
from .gf2 import gf2_matmul, row_basis, rref

Z8 = 8


def _valuation(value: int, exponent: int = 3) -> int:
    """2-adic valuation of ``value`` within Z_{2^exponent} (v(0) = exponent)."""

    modulus = 1 << exponent
    if value % modulus == 0:
        return exponent
    value %= modulus
    v = 0
    while value % 2 == 0:
        value //= 2
        v += 1
    return v


def module_kernel(
    constraints: list[tuple[np.ndarray, int]], n: int, *, exponent: int = 3
) -> np.ndarray:
    """Generators of ``{t in Z_{2^exponent}^n : factor * (u . t) = 0}`` for
    all ``(u, factor)`` constraints.

    Standard module elimination over the local ring ``Z_8``: each constraint
    refines the generating set; a pivot generator with minimal 2-adic value
    clears the others and is then scaled by the exact power of two that
    keeps it in the kernel.
    """

    modulus = 1 << exponent
    generators = np.eye(n, dtype=np.int64)
    for pattern, factor in constraints:
        values = (factor * (generators @ pattern.astype(np.int64))) % modulus
        support = np.flatnonzero(values)
        if support.size == 0:
            continue
        valuations = np.array([_valuation(int(v), exponent) for v in values[support]])
        pivot_pos = int(support[np.argmin(valuations)])
        pivot_val = int(values[pivot_pos])
        v_star = _valuation(pivot_val, exponent)
        unit = pivot_val >> v_star
        unit_inv = pow(unit, -1, modulus)
        for index in support:
            if index == pivot_pos:
                continue
            coefficient = ((int(values[index]) >> v_star) * unit_inv) % modulus
            generators[index] = (generators[index] - coefficient * generators[pivot_pos]) % modulus
        generators[pivot_pos] = (generators[pivot_pos] * (1 << (exponent - v_star))) % modulus
    keep = [row for row in generators if (row % modulus).any()]
    if not keep:
        return np.zeros((0, n), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64) % modulus


def _basis_and_normalizer(code: CSSCode, family: str):
    if family == "Z":
        checks = code.c_x
        normalizer = np.vstack([code.c_x, code.logical_x])
    else:
        checks = code.c_z
        normalizer = np.vstack([code.c_z, code.logical_z])
    return checks.astype(np.int64), normalizer.astype(np.int64)


def _collect_constraints(
    checks: np.ndarray, normalizer: np.ndarray, n: int, *, exponent: int = 3
):
    """The closed constraint ladder for level ``L = exponent``.

    For each X-check basis vector ``v``: the check itself at modulus
    ``2^L``, then depth-``t`` products of a local F_2 basis of the
    restrictions ``{g & v}`` at modulus ``2^{L-t}`` for ``t = 1..L-1``.
    Basis reduction at every tier is justified by multilinearity: the
    correction terms of expressing ``g`` over the basis land exactly in the
    next tier down, which is also included (verified by brute force against
    the raw coset-phase definition in the tests).
    """

    from itertools import combinations

    constraints: list[tuple[np.ndarray, int]] = []
    seen: dict[int, set[bytes]] = {t: set() for t in range(1, exponent)}
    final_rows: list[np.ndarray] = []
    running: list[np.ndarray] = []  # global F2 RREF of the mod-2 tier

    def add_final(pattern: np.ndarray) -> None:
        candidate = (pattern % 2).astype(np.uint8)
        if not candidate.any():
            return
        stacked = np.vstack(running + [candidate]) if running else candidate[None, :]
        reduced, _ = rref(stacked)
        if len(reduced) > len(running):
            running.clear()
            running.extend(list(reduced))
            final_rows.append(pattern)

    for v in checks:
        constraints.append((v.copy(), 1))  # t.v = 0 mod 2^L
        support = np.flatnonzero(v)
        width = support.size
        if width == 0:
            continue
        restricted = np.array(
            [row for row in (normalizer & v) if row.any()], dtype=np.uint8
        )
        if restricted.size == 0:
            continue
        basis_local = row_basis(restricted[:, support], ncols=width)
        patterns = []
        for local_row in basis_local:
            pattern = np.zeros(n, dtype=np.int64)
            pattern[support] = local_row
            patterns.append(pattern)
        for depth in range(1, exponent):
            factor = 1 << depth
            if depth == exponent - 1:
                # final tier (mod 2): F2 span reduction is complete
                for combo in combinations(range(len(patterns)), depth):
                    product = patterns[combo[0]].copy()
                    for index in combo[1:]:
                        product = product & patterns[index]
                    add_final(product)
            else:
                for combo in combinations(range(len(patterns)), depth):
                    product = patterns[combo[0]].copy()
                    for index in combo[1:]:
                        product = product & patterns[index]
                    if not product.any():
                        continue
                    key = product.astype(np.uint8).tobytes()
                    if key not in seen[depth]:
                        seen[depth].add(key)
                        constraints.append((product, factor))
    for pattern in final_rows:
        constraints.append((pattern, 1 << (exponent - 1)))
    return constraints


@dataclass(frozen=True)
class DiagonalGenerator:
    parameter: np.ndarray  # t in Z_8^n
    singles: np.ndarray  # phi on logical basis vectors, mod 8
    pair_coefficients: np.ndarray | None  # -2 t.(g_i & g_j) mod 8, or None (k too large)
    level: int | None  # 0 = trivial, 1 = Pauli, 2 = Clifford, 3 = T-level
    certificate: dict[str, bool]

    @property
    def is_logical_identity(self) -> bool:
        return self.level == 0


_PAIR_K_LIMIT = 1024
_TRIPLE_K_LIMIT = 1024


class HierarchyAnalysis:
    """Complete strict-diagonal level-3 analysis of one family (Z or X)."""

    def __init__(self, code: CSSCode, family: str = "Z", *, level: int = 3):
        if family not in ("Z", "X"):
            raise ValueError("family must be 'Z' or 'X'")
        if not 2 <= level <= 6:
            raise ValueError("level must be between 2 and 6")
        self.code = code
        self.family = family
        self.level = level
        checks, normalizer = _basis_and_normalizer(code, family)
        self.checks = checks
        self.normalizer = normalizer
        constraints = _collect_constraints(checks, normalizer, code.n, exponent=level)
        self.kernel = module_kernel(constraints, code.n, exponent=level)
        self._constraints = constraints
        logicals = (code.logical_x if family == "Z" else code.logical_z).astype(np.int64)
        self.generators = tuple(
            self._describe(parameter, logicals) for parameter in self.kernel
        )

    def _describe(self, parameter: np.ndarray, logicals: np.ndarray) -> DiagonalGenerator:
        k = self.code.k
        modulus = 1 << self.level
        singles = (logicals @ parameter) % modulus
        pair = None
        # (value, degree) pairs; a degree-d monomial with coefficient c sits
        # at hierarchy level d + 2 - v_2(c): T=(1,1)->3, S=(2,1)->2,
        # Z=(4,1)->1, CS=(2,2)->3, CZ=(4,2)->2, CCZ=(4,3)->3.
        monomials = [(int(v), 1) for v in singles]
        if k <= _PAIR_K_LIMIT and k > 1:
            weighted = (logicals * parameter) @ logicals.T
            pair = (-2 * weighted) % modulus
            np.fill_diagonal(pair, 0)
            upper = pair[np.triu_indices(k, 1)]
            monomials.extend((int(v), 2) for v in upper[upper != 0])
        _EXACT_K = 24
        triples_checked = k < 3 or k <= _TRIPLE_K_LIMIT
        if self.level >= 4 and k > _EXACT_K:
            triples_checked = False  # higher-level grading needs exact overlaps
        if k >= 3 and triples_checked and self.level >= 3:
            if k <= _EXACT_K:
                # exact degree-3 and (level >= 4) degree-4 coefficients
                for i in range(k):
                    for j in range(i + 1, k):
                        for l in range(j + 1, k):
                            overlap = int((logicals[i] & logicals[j] & logicals[l]) @ parameter)
                            value = (4 * overlap) % modulus
                            if value:
                                monomials.append((value, 3))
                if self.level >= 4 and k >= 4:
                    for i in range(k):
                        for j in range(i + 1, k):
                            for l in range(j + 1, k):
                                for m in range(l + 1, k):
                                    overlap = int(
                                        (logicals[i] & logicals[j] & logicals[l] & logicals[m])
                                        @ parameter
                                    )
                                    value = (8 * overlap) % modulus
                                    if value:
                                        monomials.append((value, 4))
            else:
                # large k at level 3: only odd overlaps matter mod 8, so the
                # masked parity screen is exact
                odd_columns = np.flatnonzero(parameter % 2)
                if odd_columns.size:
                    restricted = np.ascontiguousarray(
                        logicals[:, odd_columns].astype(np.uint8)
                    )
                    for i in range(k):
                        masked = restricted & restricted[i][None, :]
                        parity = gf2_matmul(masked, restricted.T)
                        sub = parity[i + 1 :, i + 1 :][np.triu_indices(k - i - 1, 1)]
                        if sub.any():
                            monomials.append((4, 3))
                            break
        nonzero = [(v % modulus, d) for v, d in monomials if v % modulus]
        if not nonzero:
            level = 0 if (k <= _PAIR_K_LIMIT and triples_checked) else None
        else:
            level = max(
                d + self.level - 1 - _valuation(v, self.level) for v, d in nonzero
            )
        verified = all(
            (factor * int(pattern @ parameter)) % (1 << self.level) == 0
            for pattern, factor in self._constraints
        )
        certificate = {
            "kernel_membership_verified": bool(verified),
            "level_complete": bool(k <= _PAIR_K_LIMIT and triples_checked),
        }
        return DiagonalGenerator(
            parameter=parameter.copy(),
            singles=singles,
            pair_coefficients=pair,
            level=level,
            certificate=certificate,
        )

    @property
    def has_t_level_gate(self) -> bool:
        """A logical gate genuinely at the analysis level."""

        return any(g.level == self.level for g in self.generators)

    @property
    def certified(self) -> bool:
        return all(g.certificate["kernel_membership_verified"] for g in self.generators)

    def to_dict(self) -> dict[str, Any]:
        levels = [g.level for g in self.generators]
        return {
            "family": self.family,
            "level": self.level,
            "kernel_generators": int(self.kernel.shape[0]),
            "constraint_count": len(self._constraints),
            "nontrivial_generators": int(sum(1 for g in self.generators if g.level not in (0, None))),
            "has_t_level_gate": bool(self.has_t_level_gate),
            "has_clifford_level_gate": bool(any(lv == 2 for lv in levels)),
            "max_level": max((lv for lv in levels if lv is not None), default=0),
            "levels_complete": all(
                g.certificate["level_complete"] for g in self.generators
            ) if self.generators else True,
            "certified": self.certified,
        }


def analyze_hierarchy(
    code: CSSCode, family: str = "Z", *, level: int = 3
) -> HierarchyAnalysis:
    """Complete strict single-qubit diagonal analysis at the given level."""

    return HierarchyAnalysis(code, family, level=level)
