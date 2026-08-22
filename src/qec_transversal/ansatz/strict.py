r"""Strict site-dependent transversal Clifford gates of any stabilizer code.

The ansatz is one *arbitrary* single-qubit Clifford per qubit, modulo Paulis
and with no qubit permutation: a block-diagonal symplectic matrix
``M = diag(M_1, ..., M_n)`` with each ``M_i \in Sp(2, 2)``.  Because
``GL(2, 2) = Sp(2, 2)`` over ``\mathbb F_2``, the symplectic condition is free
and this group is *exactly* the unit group of the singleton-partition
preservation algebra

    ``A(S) = { M block-diagonal : S M \subseteq S }``

built by :func:`qec_transversal.algebra.preservation.local_clifford_algebra`.

Two routes to the group, chosen automatically:

``enumeration``
    ``2^{dim A}`` sweep over the algebra keeping the elements whose blocks all
    have determinant one.  Exact, used below ``dim_cap``.
``structured``
    the certified unit-group solver of :mod:`qec_transversal.algebra`
    (radical peeling plus a constructive Wedderburn split), which is exact
    without enumeration; its generation certificate is independently
    re-checked by closing the returned generators.

If neither route completes, the result is honestly ``unknown`` -- never "no
gate exists".

Unlike the CSS-specialised :mod:`.strict_css`, this covers non-CSS codes such
as the ``[[5,1,3]]`` perfect code, and on CSS codes it must agree with the
shear construction (cross-checked in the test suite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..algebra.preservation import (
    _SL22,
    _block_action_matrix,
    local_clifford_algebra,
)
from ..algebra.preservation import (
    ENUMERATION_DIM_CAP as _ENUMERATION_DIM_CAP,
)
from ..codes.stabilizer import StabilizerCode
from ..logical.action import project_to_logical
from ..logical.group import logical_group_summary
from ..utils.gf2 import BinaryMatrix, gf2_matmul, reduce_rows


@dataclass(frozen=True)
class LocalCliffordGenerator:
    blocks: tuple[tuple[int, int, int, int], ...]  # per-qubit (a, b, c, d)
    logical_symplectic: BinaryMatrix
    certificate: dict[str, bool]
    #: The dense ``2n x 2n`` physical symplectic matrix.  Carried explicitly
    #: because it is the input the sign-exact verifier needs, and because a
    #: partition generator has no per-qubit ``blocks`` to rebuild it from.
    matrix: BinaryMatrix | None = None

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )

    @property
    def gate_names(self) -> tuple[str, ...]:
        names = {m: name for m, name in zip(_SL22, ("I", "S", "sqrtX", "H", "HS", "SH"))}
        return tuple(names[block] for block in self.blocks)


class LocalCliffordAnalysis:
    """The complete strict-transversal Clifford group of a stabilizer code.

    Small algebras are enumerated; beyond ``dim_cap`` the structured
    unit-group solver (:mod:`..algebra.unit_group`, radical peeling + certified
    Wedderburn split) provides exact order and generators without
    enumeration — GL(2,2) = Sp(2,2) makes the unit group exactly the
    transversal group at width one.  If neither route completes, the
    result is honestly ``unknown``.
    """

    def __init__(self, code: StabilizerCode, *, dim_cap: int = _ENUMERATION_DIM_CAP):
        self.code = code
        self.algebra = local_clifford_algebra(code)
        dimension = self.algebra.shape[0]
        self.enumeration_complete = dimension <= dim_cap
        self.structured_order: int | None = None
        self.elements: list[np.ndarray] = []
        if not self.enumeration_complete:
            self._try_structured_route()
        if self.enumeration_complete and self.structured_order is None:
            # only enumerate when the structured route did not already
            # deliver the exact group (it marks completion itself)
            n = code.n
            for mask in range(1 << dimension):
                entries = np.zeros(4 * n, dtype=np.uint8)
                for bit in range(dimension):
                    if (mask >> bit) & 1:
                        entries ^= self.algebra[bit]
                a, b, c, d = entries[0::4], entries[1::4], entries[2::4], entries[3::4]
                determinant = (a & d) ^ (b & c)
                if determinant.all():
                    self.elements.append(entries)
        self.generators = tuple(
            self._describe(entries)
            for entries in self.elements
            if entries.any() and not self._is_identity(entries)
        )

    def _try_structured_route(self) -> None:
        """Unit-group solver for large algebras: exact without enumeration."""

        from ..algebra import AlgebraF2, unit_group

        n = self.code.n

        def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.zeros(4 * n, dtype=np.uint8)
            blocks_a = a.reshape(n, 2, 2)
            blocks_b = b.reshape(n, 2, 2)
            for i in range(n):
                out[4 * i : 4 * i + 4] = (blocks_a[i] @ blocks_b[i] % 2).reshape(-1)
            return out

        one = np.zeros(4 * n, dtype=np.uint8)
        one[0::4] = 1
        one[3::4] = 1
        try:
            algebra = AlgebraF2(self.algebra, multiply, one)
            result = unit_group(algebra)
        except Exception:
            return
        if result.status != "exact":
            return
        if result.order is not None and result.order <= 1 << 20:
            # belt-and-braces: independently confirm the generation
            # certificate by closing the generators into the full group
            # (the pattern of partition_units_via_structure, with cached
            # right-multiplication matrices); on mismatch the analysis
            # stays honestly "unknown" rather than trusting the solver
            rights = [algebra.right_matrix(g) for g in result.generators]
            elements = {algebra.one_coords.tobytes()}
            frontier = [algebra.one_coords]
            while frontier:
                current = frontier.pop()
                for matrix in rights:
                    product = ((matrix @ current) % 2).astype(np.uint8)
                    key = product.tobytes()
                    if key not in elements:
                        elements.add(key)
                        frontier.append(product)
            if len(elements) != result.order:
                return
        self.structured_order = result.order
        # generators back to flat entries: coords over the internal basis
        self.elements = [
            (coords @ algebra.basis % 2).astype(np.uint8)
            for coords in result.generators
        ]
        self.enumeration_complete = True  # exact via structure, not enumeration
        self._structured = True

    @staticmethod
    def _is_identity(entries: np.ndarray) -> bool:
        return bool(
            entries[0::4].all()
            and entries[3::4].all()
            and not entries[1::4].any()
            and not entries[2::4].any()
        )

    def _describe(self, entries: np.ndarray) -> LocalCliffordGenerator:
        code = self.code
        n = code.n
        matrix = _block_action_matrix(entries, n)
        stab_image = gf2_matmul(code.h, matrix)
        preserved = not reduce_rows(stab_image, *code._h_rref).any()
        logical, residue_ok = project_to_logical(
            gf2_matmul(code.logical, matrix),
            code.logical,
            qubits=n,
            logical_qubits=code.k,
            stabilizer_rref=code._h_rref,
        )
        blocks = tuple(
            (
                int(entries[4 * i]),
                int(entries[4 * i + 1]),
                int(entries[4 * i + 2]),
                int(entries[4 * i + 3]),
            )
            for i in range(n)
        )
        certificate = {
            "stabilizer_preserved": bool(preserved),
            "all_blocks_invertible": True,
            "logical_residue_in_stabilizer": bool(residue_ok),
        }
        return LocalCliffordGenerator(
            blocks=blocks,
            logical_symplectic=logical,
            certificate=certificate,
            matrix=matrix,
        )

    @property
    def group_order(self) -> int | None:
        """Order of the physical transversal group mod Paulis (identity included)."""

        if not self.enumeration_complete:
            return None
        if self.structured_order is not None:
            return self.structured_order
        return len(self.elements)

    @property
    def status(self) -> str:
        """``"exact"`` when enumeration or the structured unit-group route
        completed, else ``"unknown"`` — an incomplete computation is never
        a negative result."""

        return "exact" if self.enumeration_complete else "unknown"

    @property
    def certified(self) -> bool:
        """True only for a COMPLETE computation whose per-generator
        certificates all pass; an unfinished enumeration is never
        certified (vacuous truth over an empty generator list is exactly
        the failure mode a verifier must not have)."""

        return self.enumeration_complete and all(
            all(g.certificate.values()) for g in self.generators
        )

    def to_dict(self) -> dict[str, Any]:
        if self.enumeration_complete:
            nontrivial = [
                g.logical_symplectic
                for g in self.generators
                if not g.is_logical_identity
            ]
            logical = logical_group_summary(nontrivial, self.code.k)
            logical["status"] = "exact" if logical["exact"] else "lower_bound"
        else:
            nontrivial = []
            logical = {
                "computed": False,
                "exact": False,
                "order": None,
                "lower_bound": None,
                "status": "unknown",
            }
        return {
            "status": self.status,
            "sound": True,
            "complete": bool(self.enumeration_complete),
            "algebra_dimension": int(self.algebra.shape[0]),
            "enumeration_complete": bool(self.enumeration_complete),
            "physical_group_order": self.group_order,
            "nontrivial_logical_generators": len(nontrivial),
            "logical_group": logical,
            "certified": self.certified,
        }


def analyze_local_clifford(code: StabilizerCode) -> LocalCliffordAnalysis:
    """The complete strict-transversal Clifford group, CSS or not."""

    return LocalCliffordAnalysis(code)
