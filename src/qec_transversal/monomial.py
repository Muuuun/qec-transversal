"""Permutation-times-local-Clifford gates: monomial GF(4) automorphisms.

Under the CRSS correspondence (Calderbank-Rains-Shor-Sloane, IEEE TIT 44,
1369 (1998)) a stabilizer code is an additive code over GF(4), and the
group of qubit permutations combined with per-qubit Cliffords mod Pauli is
the code's automorphism group under ``S_3 wr S_n`` — coordinate
permutations times the six axis permutations of {X, Y, Z} per coordinate.

This module computes that group exactly for the given row set through the
CRSS three-column binary encoding (their p. 11 "artifice"): each qubit
becomes three columns ``(x_i, z_i, x_i + z_i)`` — the axes X, Z, Y — and a
Pauli row is incident to exactly the two axes it anticommutes with.
Automorphisms of the resulting colored incidence graph (computed exactly by
BLISS) are precisely the monomial maps preserving the row set; every
generator is then certified against the full stabilizer row space, and the
logical action is extracted symplectically.

Row-set scope: when the stabilizer group is small (rank <= ``_FULL_GROUP_RANK``)
all nonzero stabilizer elements are used, making the result basis-independent
— the true automorphism group of the code.  For larger codes the given
generator rows are used, the same scope as the Tanner-graph literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .automorphisms import _require_igraph, permutation_group_order
from .gf2 import reduce_rows, symplectic_product
from .matching import logical_group_summary
from .stabilizer import StabilizerCode, analyze_local_clifford

try:  # pragma: no cover
    import igraph
except ImportError:  # pragma: no cover
    igraph = None

_FULL_GROUP_RANK = 14

#: Axis coordinate vectors: X = (1,0), Z = (0,1), Y = (1,1).
_AXES = ((1, 0), (0, 1), (1, 1))


def _row_set(code: StabilizerCode, natural_rows: np.ndarray | None) -> np.ndarray:
    if code.rank <= _FULL_GROUP_RANK:
        rows = []
        for mask in range(1, 1 << code.rank):
            v = np.zeros(2 * code.n, dtype=np.uint8)
            for bit in range(code.rank):
                if (mask >> bit) & 1:
                    v ^= code.h[bit]
            if v.any():
                rows.append(v)
        return np.asarray(rows, dtype=np.uint8)
    if natural_rows is not None:
        # symmetry-preserving generator set (RREF bases break natural
        # symmetries such as lattice translations)
        return np.asarray(natural_rows, dtype=np.uint8) & 1
    return code.h.copy()


@dataclass(frozen=True)
class MonomialGenerator:
    qubit_permutation: np.ndarray
    blocks: tuple[tuple[int, int, int, int], ...]  # per-qubit 2x2 (a,b,c,d)
    logical_symplectic: np.ndarray
    certificate: dict[str, bool]

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )


class MonomialAnalysis:
    """Exact monomial (permutation x local-Clifford) automorphism group."""

    def __init__(self, code: StabilizerCode, *, natural_rows: np.ndarray | None = None):
        _require_igraph()
        self.code = code
        n = code.n
        rows = _row_set(code, natural_rows)
        self.row_set_complete = code.rank <= _FULL_GROUP_RANK

        # colored incidence graph: hubs (0), axis columns (1), rows (2)
        hub = list(range(n))
        col = lambda i, a: n + 3 * i + a  # noqa: E731
        row0 = n + 3 * n
        graph = igraph.Graph(row0 + rows.shape[0])
        edges = []
        for i in range(n):
            for a in range(3):
                edges.append((hub[i], col(i, a)))
        for r, s in enumerate(rows):
            x, z = s[:n], s[n:]
            for i in np.flatnonzero(x | z):
                xi, zi = int(x[i]), int(z[i])
                # a Pauli on axis (xi, zi) is incident to the two OTHER axes
                for a, axis in enumerate(_AXES):
                    if axis != (xi, zi):
                        edges.append((col(i, a), row0 + r))
        graph.add_edges(edges)
        colors = [0] * n + [1] * (3 * n) + [2] * rows.shape[0]

        raw = graph.automorphism_group(color=colors)
        self.column_generators: list[np.ndarray] = []
        generators: list[MonomialGenerator] = []
        seen: set[bytes] = set()
        for g in raw:
            perm_full = np.asarray(g, dtype=int)
            qubit_perm = perm_full[:n].copy()
            # per-qubit axis map: axis a of qubit i -> axis of qubit perm[i]
            blocks = []
            for i in range(n):
                image_axes = []
                for a in range(3):
                    target = int(perm_full[col(i, a)])
                    image_axes.append((target - n) % 3)
                m_x = _AXES[image_axes[0]]
                m_z = _AXES[image_axes[1]]
                blocks.append((m_x[0], m_x[1], m_z[0], m_z[1]))
            key = qubit_perm.tobytes() + bytes(b for block in blocks for b in block)
            if key in seen:
                continue
            seen.add(key)
            column_action = perm_full[n : n + 3 * n] - n
            self.column_generators.append(column_action)
            generators.append(self._describe(qubit_perm, blocks))
        self.generators = tuple(g for g in generators if g is not None)
        self.group_order = (
            permutation_group_order(self.column_generators)
            if self.column_generators
            else 1
        )

    def _describe(self, perm: np.ndarray, blocks) -> MonomialGenerator | None:
        code = self.code
        n = code.n
        matrix = np.zeros((2 * n, 2 * n), dtype=np.uint8)
        for i in range(n):
            a, b, c, d = blocks[i]
            j = int(perm[i])
            matrix[i, j] = a
            matrix[i, n + j] = b
            matrix[n + i, j] = c
            matrix[n + i, n + j] = d
        image = (code.h.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(np.uint8)
        preserved = not reduce_rows(image, *code._h_rref).any()
        if code.k:
            images = (code.logical.astype(np.int64) @ matrix.astype(np.int64) % 2).astype(
                np.uint8
            )
            x_coeff = symplectic_product(images, code.logical[code.k :], qubits=n)
            z_coeff = symplectic_product(images, code.logical[: code.k], qubits=n)
            logical = np.hstack([x_coeff, z_coeff]).astype(np.uint8)
            residue = (
                images
                ^ (logical.astype(np.int64) @ code.logical.astype(np.int64) % 2).astype(
                    np.uint8
                )
            ) & 1
            residue_ok = not reduce_rows(residue, *code._h_rref).any()
        else:
            logical = np.zeros((0, 0), dtype=np.uint8)
            residue_ok = True
        certificate = {
            "stabilizer_preserved": bool(preserved),
            "logical_residue_in_stabilizer": bool(residue_ok),
        }
        return MonomialGenerator(
            qubit_permutation=perm,
            blocks=tuple(blocks),
            logical_symplectic=logical,
            certificate=certificate,
        )

    @property
    def certified(self) -> bool:
        return all(all(g.certificate.values()) for g in self.generators)

    @property
    def permutation_image_order(self) -> int:
        """Order of the image of the monomial group in ``S_n`` — the qubit
        permutations with the per-qubit local-Clifford factors forgotten."""

        if not self.generators:
            return 1
        return permutation_group_order([g.qubit_permutation for g in self.generators])

    def to_dict(self) -> dict[str, Any]:
        nontrivial = [
            g.logical_symplectic for g in self.generators if not g.is_logical_identity
        ]
        return {
            "row_set_complete": bool(self.row_set_complete),
            "generator_count": len(self.generators),
            "monomial_group_order": int(self.group_order),
            "permutation_image_order": int(self.permutation_image_order),
            "nontrivial_logical_generators": len(nontrivial),
            "logical_group": logical_group_summary(nontrivial, self.code.k),
            "certified": self.certified,
        }


def analyze_monomial(
    code: StabilizerCode, *, natural_rows: np.ndarray | None = None
) -> MonomialAnalysis:
    """Exact permutation x local-Clifford automorphism group (needs igraph).

    ``natural_rows`` supplies a symmetry-preserving generator set (dependent
    rows welcome) used when the stabilizer group is too large to enumerate.
    """

    return MonomialAnalysis(code, natural_rows=natural_rows)


def strict_cross_check(
    code: StabilizerCode, *, natural_rows: np.ndarray | None = None
) -> dict[str, Any]:
    """Cross-check the monomial group against the strict-transversal group.

    Kernel identity.  Project the monomial group ``G <= S_3 wr S_n`` onto
    its qubit-permutation part ``P <= S_n``.  The kernel of that projection
    — the monomial maps whose permutation is trivial — consists of exactly
    the width-one (strict transversal) local Cliffords preserving the row
    set, so the first isomorphism theorem gives ``|kernel| = |G| / |P|``.

    In FULL-GROUP scope (``row_set_complete``: the row set is every nonzero
    stabilizer element) preserving the row set is the same condition as
    preserving the stabilizer group, hence the kernel *is* the group
    computed by :func:`~.stabilizer.analyze_local_clifford` and

        ``|strict| = |monomial| / |image of monomial in S_n|``.

    In generator-row scope the row-SET condition is strictly stronger than
    stabilizer-GROUP preservation, so only the one-sided bound
    ``|kernel| <= |strict|`` holds — on Steane's RREF basis the kernel has
    order 2 against a strict group of order 6.

    ``mode`` is therefore ``"equality"`` only when the row set is complete
    AND the strict analysis is certified; otherwise the check degrades to
    ``"lower_bound_only"``.  This is an independent consistency probe of
    two separately certified engines — it feeds into neither.
    """

    mono = MonomialAnalysis(code, natural_rows=natural_rows)
    strict = analyze_local_clifford(code)
    image_order = mono.permutation_image_order
    assert mono.group_order % image_order == 0, (
        "first-isomorphism-theorem invariant violated: the permutation-image "
        "order must divide the monomial group order"
    )
    kernel_order = mono.group_order // image_order
    strict_order = strict.group_order
    equality_scope = mono.row_set_complete and strict.certified
    applicable = strict_order is not None
    if not applicable:
        consistent = None
    elif equality_scope:
        consistent = bool(kernel_order == strict_order)
    else:
        consistent = bool(kernel_order <= strict_order)
    return {
        "applicable": bool(applicable),
        "monomial_order": int(mono.group_order),
        "permutation_image_order": int(image_order),
        "kernel_order": int(kernel_order),
        "strict_order": int(strict_order) if strict_order is not None else None,
        "consistent": consistent,
        "mode": "equality" if equality_scope else "lower_bound_only",
    }
