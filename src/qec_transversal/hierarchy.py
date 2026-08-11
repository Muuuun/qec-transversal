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
from .gf2 import row_basis, rref

Z8 = 8


def _valuation(value: int) -> int:
    """2-adic valuation of ``value`` within Z_8 (v(0) = 3)."""

    if value % 8 == 0:
        return 3
    value %= 8
    v = 0
    while value % 2 == 0:
        value //= 2
        v += 1
    return v


def module_kernel(constraints: list[tuple[np.ndarray, int]], n: int) -> np.ndarray:
    """Generators of ``{t in Z_8^n : factor * (u . t) = 0 mod 8}`` for all
    ``(u, factor)`` constraints.

    Standard module elimination over the local ring ``Z_8``: each constraint
    refines the generating set; a pivot generator with minimal 2-adic value
    clears the others and is then scaled by the exact power of two that
    keeps it in the kernel.
    """

    generators = np.eye(n, dtype=np.int64)
    for pattern, factor in constraints:
        values = (factor * (generators @ pattern.astype(np.int64))) % Z8
        support = np.flatnonzero(values)
        if support.size == 0:
            continue
        valuations = np.array([_valuation(int(v)) for v in values[support]])
        pivot_pos = int(support[np.argmin(valuations)])
        pivot_val = int(values[pivot_pos])
        v_star = _valuation(pivot_val)
        unit = pivot_val >> v_star
        unit_inv = pow(unit, -1, Z8)
        for index in support:
            if index == pivot_pos:
                continue
            coefficient = ((int(values[index]) >> v_star) * unit_inv) % Z8
            generators[index] = (generators[index] - coefficient * generators[pivot_pos]) % Z8
        generators[pivot_pos] = (generators[pivot_pos] * (1 << (3 - v_star))) % Z8
    keep = [row for row in generators if (row % Z8).any()]
    if not keep:
        return np.zeros((0, n), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64) % Z8


def _basis_and_normalizer(code: CSSCode, family: str):
    if family == "Z":
        checks = code.c_x
        normalizer = np.vstack([code.c_x, code.logical_x])
    else:
        checks = code.c_z
        normalizer = np.vstack([code.c_z, code.logical_z])
    return checks.astype(np.int64), normalizer.astype(np.int64)


def _collect_constraints(checks: np.ndarray, normalizer: np.ndarray, n: int):
    """The closed constraint system (A, D, E) with mod-2 rows span-reduced.

    All D and E patterns for a check ``v`` live inside ``supp(v)``, so each
    check contributes at most ``|supp(v)|`` independent mod-2 rows; unique
    restrictions plus a local-rank early exit keep the pair loop cheap even
    when the normalizer is large.
    """

    constraints: list[tuple[np.ndarray, int]] = []
    seen_mod4: set[bytes] = set()
    mod2_rows: list[np.ndarray] = []
    running: list[np.ndarray] = []  # global F2 RREF rows of the mod-2 span

    def add_mod2(pattern: np.ndarray) -> bool:
        candidate = (pattern % 2).astype(np.uint8)
        if not candidate.any():
            return False
        stacked = np.vstack(running + [candidate]) if running else candidate[None, :]
        reduced, _ = rref(stacked)
        if len(reduced) > len(running):
            running.clear()
            running.extend(list(reduced))
            mod2_rows.append(pattern)
            return True
        return False

    for v in checks:
        constraints.append((v.copy(), 1))  # t.v = 0 mod 8
        support = np.flatnonzero(v)
        width = support.size
        if width == 0:
            continue
        # Coordinatewise AND is bilinear over F_2, so a local F_2 basis of
        # the restrictions {g & v} suffices: D on the basis together with E
        # implies D for every g, and basis pairs span all E patterns.
        restricted = np.array(
            [row for row in (normalizer & v) if row.any()], dtype=np.uint8
        )
        if restricted.size == 0:
            continue
        basis_local = row_basis(restricted[:, support], ncols=width)
        unique: dict[bytes, np.ndarray] = {}
        for local_row in basis_local:
            pattern = np.zeros(n, dtype=np.int64)
            pattern[support] = local_row
            unique.setdefault(pattern.astype(np.uint8).tobytes(), pattern)
        for key, pattern in unique.items():
            if key not in seen_mod4:
                seen_mod4.add(key)
                constraints.append((pattern, 2))  # t.(g&v) = 0 mod 4
        # E-rows: pairwise products of the unique restrictions.  Everything
        # lives in the |supp(v)|-dimensional local space, so patterns are
        # first reduced against a tiny local bitmask RREF and only locally
        # new ones (at most |supp(v)| per check) reach the global span.
        restrictions = list(unique.values())
        masks = [int("".join(str(int(r[q])) for q in support[::-1]), 2) for r in restrictions]
        local: list[int] = []  # local F2 basis as bitmasks, kept reduced

        def locally_new(mask: int) -> bool:
            reduced = mask
            changed = True
            while changed:
                changed = False
                for row in local:
                    pivot = row & -row
                    if reduced & pivot:
                        reduced ^= row
                        changed = True
            if reduced == 0:
                return False
            local.append(reduced)
            return True

        done = False
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                mask = masks[a] & masks[b]
                if mask and locally_new(mask):
                    pattern = restrictions[a] & restrictions[b]
                    add_mod2(pattern)
                    if len(local) >= width:
                        done = True
                        break
            if done:
                break
    for pattern in mod2_rows:
        constraints.append((pattern, 4))  # t.pattern = 0 mod 2
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


_PAIR_K_LIMIT = 900
_TRIPLE_K_LIMIT = 128


class HierarchyAnalysis:
    """Complete strict-diagonal level-3 analysis of one family (Z or X)."""

    def __init__(self, code: CSSCode, family: str = "Z"):
        if family not in ("Z", "X"):
            raise ValueError("family must be 'Z' or 'X'")
        self.code = code
        self.family = family
        checks, normalizer = _basis_and_normalizer(code, family)
        self.checks = checks
        self.normalizer = normalizer
        constraints = _collect_constraints(checks, normalizer, code.n)
        self.kernel = module_kernel(constraints, code.n)
        self._constraints = constraints
        logicals = (code.logical_x if family == "Z" else code.logical_z).astype(np.int64)
        self.generators = tuple(
            self._describe(parameter, logicals) for parameter in self.kernel
        )

    def _describe(self, parameter: np.ndarray, logicals: np.ndarray) -> DiagonalGenerator:
        k = self.code.k
        singles = (logicals @ parameter) % Z8
        pair = None
        # (value, degree) pairs; a degree-d monomial with coefficient c sits
        # at hierarchy level d + 2 - v_2(c): T=(1,1)->3, S=(2,1)->2,
        # Z=(4,1)->1, CS=(2,2)->3, CZ=(4,2)->2, CCZ=(4,3)->3.
        monomials = [(int(v), 1) for v in singles]
        if k <= _PAIR_K_LIMIT and k > 1:
            # W_ij = sum_q l_iq l_jq t_q, so the CS/CZ coefficient is -2 W.
            weighted = (logicals * parameter) @ logicals.T
            pair = (-2 * weighted) % Z8
            np.fill_diagonal(pair, 0)
            upper = pair[np.triu_indices(k, 1)]
            monomials.extend((int(v), 2) for v in upper[upper != 0])
        triples_checked = k < 3 or k <= _TRIPLE_K_LIMIT
        if triples_checked and k >= 3:
            # CCZ coefficient is 4 * (triple overlap mod 2): parity suffices.
            odd = (logicals * parameter) % 2
            for i in range(k):
                masked = logicals & (odd[i][None, :])
                parity = (masked @ logicals.T) % 2
                sub = parity[i + 1 :, i + 1 :][np.triu_indices(k - i - 1, 1)]
                if sub.any():
                    monomials.append((4, 3))
                    break
        nonzero = [(v % Z8, d) for v, d in monomials if v % Z8]
        if not nonzero:
            level = 0 if (k <= _PAIR_K_LIMIT and triples_checked) else None
        else:
            level = max(d + 2 - _valuation(v) for v, d in nonzero)
        verified = all(
            (factor * int(pattern @ parameter)) % Z8 == 0
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
        """A logical gate genuinely at the third level (odd logical phase)."""

        return any(g.level == 3 for g in self.generators)

    @property
    def certified(self) -> bool:
        return all(g.certificate["kernel_membership_verified"] for g in self.generators)

    def to_dict(self) -> dict[str, Any]:
        levels = [g.level for g in self.generators]
        return {
            "family": self.family,
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


def analyze_hierarchy(code: CSSCode, family: str = "Z") -> HierarchyAnalysis:
    """Complete strict single-qubit diagonal level-3 analysis."""

    return HierarchyAnalysis(code, family)
