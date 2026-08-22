r"""Code-preserving Clifford gates on a *prescribed qubit partition*.

This is the general framework of the package.  Fix a partition
``P = {C_1, ..., C_r}`` of the physical qubits; the ansatz is one arbitrary
Clifford supported on each cell, all applied in a single layer.  The
code-preserving elements are exactly

.. math::

    A_P(S)^\times \cap \prod_{C \in P} \mathrm{Sp}(2|C|, 2),

the symplectic units of the preservation algebra
(:mod:`qec_transversal.algebra.preservation`).  Singleton cells recover the
strict transversal group of :mod:`.strict`; pairs recover the fixed-matching
two-local group; a single cell containing every qubit recovers the whole
code-preserving Clifford group.

Three routes, all exact where they apply:

:class:`PartitionCliffordAnalysis`
    ``2^{dim A}`` enumeration filtered by the per-cell symplectic-form
    condition.  Cheap and exact for small algebras.
:func:`partition_units_via_structure` with ``method="enumeration"``
    the certified unit group ``A^x`` from :mod:`qec_transversal.algebra`
    (no ``2^{dim}`` sweep), whose *group* -- ``|A^x|`` elements, typically far
    fewer than ``2^{dim A}`` -- is then enumerated and cut by the per-block
    form condition.  Blockwise-symplectic elements are closed under products,
    so the cut really is a subgroup.
:func:`partition_units_via_structure` with ``method="phi"``
    the same certified ``A^x``, but the symplectic cut taken as an *index*
    rather than a sweep: ``|G| = |A^x| / |orbit of 1|`` under the congruence
    action ``a . u = sigma(u) a u``, generators by Schreier's lemma
    (:mod:`qec_transversal.algebra.unitary_group`).  Cost is linear in
    ``|A^x| / |G|`` instead of ``|A^x|``, which is what makes cells wider
    than three qubits decidable at all.

Every exit reports its status honestly; a route that cannot certify returns
``unknown`` rather than a truncated count dressed up as an answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..algebra.preservation import (
    ENUMERATION_DIM_CAP as _ENUMERATION_DIM_CAP,
)
from ..algebra.preservation import (
    _cell_blocks,
    _local_symplectic_form,
    _partition_action_matrix,
    _partition_multiply,
    partition_algebra,
    symplectic_involution,
)
from ..algebra.unitary_group import DEFAULT_ORBIT_CAP as _DEFAULT_ORBIT_CAP
from ..algebra.unitary_group import unitary_group
from ..codes.stabilizer import StabilizerCode
from ..logical.action import project_to_logical
from ..logical.group import logical_group_summary, schreier_sims_order
from ..utils.gf2 import gf2_matmul, reduce_rows
from ..utils.polynomials import _solve_coords
from ..utils.symplectic import symplectic_group_order
from .strict import LocalCliffordGenerator


class PartitionCliffordAnalysis:
    """The complete depth-one Clifford group on a fixed qubit partition.

    For singleton cells this is the strict-transversal group; for pair cells
    it is Albert's fixed-matching two-local group ``N_M`` (arXiv:2608.05688,
    App. D), obtained here for arbitrary stabilizer codes as the symplectic
    unit set of the cell-block algebra.  Width-1 blocks are automatically
    symplectic when invertible (``GL(2,2) = Sp(2,2)``); wider blocks need
    the explicit symplectic-form condition, which is imposed per element.
    """

    def __init__(
        self,
        code: StabilizerCode,
        cells: list[tuple[int, ...]],
        *,
        dim_cap: int = _ENUMERATION_DIM_CAP,
    ):
        self.code = code
        self.cells = [tuple(cell) for cell in cells]
        self.algebra, self._layout = partition_algebra(code, self.cells)
        dimension = self.algebra.shape[0]
        self.enumeration_complete = dimension <= dim_cap
        self.elements: list[np.ndarray] = []
        if self.enumeration_complete:
            forms = {width: _local_symplectic_form(width) for _, width in self._layout}
            for mask in range(1 << dimension):
                entries = np.zeros(self.algebra.shape[1], dtype=np.uint8)
                for bit in range(dimension):
                    if (mask >> bit) & 1:
                        entries ^= self.algebra[bit]
                ok = True
                for (start, width), block in zip(
                    self._layout, _cell_blocks(entries, self._layout)
                ):
                    form = forms[width]
                    if not np.array_equal((block @ form @ block.T) & 1, form):
                        ok = False
                        break
                if ok:
                    self.elements.append(entries)
        self.generators: tuple[LocalCliffordGenerator, ...] = tuple(
            self._describe(entries)
            for entries in self.elements
            if not self._is_identity(entries)
        )

    def _is_identity(self, entries: np.ndarray) -> bool:
        for (start, width), block in zip(self._layout, _cell_blocks(entries, self._layout)):
            if not np.array_equal(block, np.eye(2 * width, dtype=np.uint8)):
                return False
        return True

    def _describe(self, entries: np.ndarray):
        code = self.code
        n = code.n
        matrix = _partition_action_matrix(entries, self.cells, self._layout, n)
        stab_image = gf2_matmul(code.h, matrix)
        preserved = not reduce_rows(stab_image, *code._h_rref).any()
        logical, residue_ok = project_to_logical(
            gf2_matmul(code.logical, matrix),
            code.logical,
            qubits=n,
            logical_qubits=code.k,
            stabilizer_rref=code._h_rref,
        )
        certificate = {
            "stabilizer_preserved": bool(preserved),
            "all_blocks_invertible": True,
            "logical_residue_in_stabilizer": bool(residue_ok),
        }
        return LocalCliffordGenerator(
            blocks=(),
            logical_symplectic=logical,
            certificate=certificate,
            matrix=matrix,
        )

    @property
    def group_order(self) -> int | None:
        # The orbit-stabilizer route (:func:`partition_units_via_phi_orbit`)
        # knows |G| exactly while holding only a generating set, so it sets
        # this override rather than a list of every element.
        override = getattr(self, "_group_order_override", None)
        if override is not None:
            return int(override)
        if not self.enumeration_complete:
            return None
        return len(self.elements)

    @property
    def status(self) -> str:
        return "exact" if self.enumeration_complete else "unknown"

    @property
    def certified(self) -> bool:
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
            "cells": [list(cell) for cell in self.cells],
            "algebra_dimension": int(self.algebra.shape[0]),
            "enumeration_complete": bool(self.enumeration_complete),
            "physical_group_order": self.group_order,
            "nontrivial_logical_generators": len(nontrivial),
            "logical_group": logical,
            "certified": self.certified,
        }


def analyze_partition_clifford(
    code: StabilizerCode,
    cells: list[tuple[int, ...]],
    *,
    dim_cap: int = _ENUMERATION_DIM_CAP,
) -> PartitionCliffordAnalysis:
    """Complete depth-one Clifford group on a fixed qubit partition."""

    return PartitionCliffordAnalysis(code, cells, dim_cap=dim_cap)


def partition_units_via_structure(
    code: StabilizerCode,
    cells: list[tuple[int, ...]],
    *,
    group_enumeration_cap: int = 2_000_000,
    method: str = "auto",
    orbit_cap: int = _DEFAULT_ORBIT_CAP,
) -> dict:
    """The two-local group as ``A^x  intersect  prod Sp(2|C|, 2)``.

    The unit group ``A^x`` of the partition algebra is constructed as a
    certified overgroup through the structured solver (no ``2^dim``
    enumeration).  The symplectic cut is then taken one of two ways:

    ``"enumeration"``
        sweep the *group* — ``|A^x|`` elements, typically far fewer than
        ``2^dim`` — and filter by the per-block form condition (a subgroup,
        since blockwise-symplectic elements are closed under products).
    ``"phi"``
        no sweep at all: ``|G| = |A^x| / |orbit of 1|`` under the congruence
        action ``a . u = sigma(u) a u``, with generators from Schreier's
        lemma (:func:`..algebra.unitary_group.unitary_group`).  This is what
        makes wide cells decidable: the sweep is linear in ``|A^x|``, the
        orbit is linear in ``|A^x| / |G|``, so a *large* gate group now costs
        *less*, not more.

    ``"auto"`` (the default) keeps the sweep while it is affordable, so every
    previously reported number is reproduced element for element, and switches
    to the orbit route above ``group_enumeration_cap``.  Status is honest at
    every exit.
    """

    if method not in ("auto", "enumeration", "phi"):
        raise ValueError("method must be 'auto', 'enumeration', or 'phi'")

    from ..algebra import AlgebraF2, unit_group
    from ..logical.group import logical_group_summary

    algebra_basis, layout = partition_algebra(code, cells)
    width = algebra_basis.shape[1] if algebra_basis.size else sum(
        (2 * len(c)) ** 2 for c in cells
    )
    multiply = _partition_multiply(cells, layout, width)
    one = np.zeros(width, dtype=np.uint8)
    for (start, cell_width) in layout:
        size = 2 * cell_width
        one[start : start + size * size] = np.eye(size, dtype=np.uint8).reshape(-1)

    try:
        algebra = AlgebraF2(algebra_basis, multiply, one)
        units = unit_group(algebra)
    except Exception:
        units = None
    if units is None or units.status != "exact":
        # fall back to exact enumeration when feasible (still certified);
        # the deterministic Cohen-Ivanyos-Wales radical now covers the
        # structured route, but a failed downstream certification (e.g.
        # in the Wedderburn stage) can still land here honestly
        if algebra_basis.shape[0] <= _ENUMERATION_DIM_CAP:
            fallback = PartitionCliffordAnalysis(code, cells).to_dict()
            fallback["unit_group_order"] = None
            fallback["symplectic_group_order"] = fallback["physical_group_order"]
            fallback["detail"] = (
                "structured unit-group route not certified; exact by "
                "enumeration instead"
            )
            return fallback
        return {
            "status": "unknown",
            "unit_group_order": None,
            "symplectic_group_order": None,
            "logical_group": {"computed": False, "status": "unknown"},
            "detail": "unit-group structure not certified and dim exceeds cap",
        }
    if units.order > group_enumeration_cap and method == "enumeration":
        return {
            "status": "unknown",
            "unit_group_order": units.order,
            "symplectic_group_order": None,
            "logical_group": {"computed": False, "status": "unknown"},
            "detail": (
                f"|A^x| = {units.order} certified, but exceeds the group "
                "enumeration cap for the symplectic cut"
            ),
        }
    if method == "phi" or (method == "auto" and units.order > group_enumeration_cap):
        summary = _phi_orbit_summary(
            code,
            cells,
            algebra,
            algebra_basis,
            layout,
            units,
            orbit_cap=orbit_cap,
        )
        if summary["status"] == "exact" or method == "phi":
            return summary
        # "auto" and the orbit route could not certify: fall through only if
        # the sweep is affordable, otherwise report the orbit route's reason.
        if units.order > group_enumeration_cap:
            return summary

    # enumerate the GROUP (not the algebra) by closure over unit generators
    forms = {cw: _local_symplectic_form(cw) for _, cw in layout}
    elements = {algebra.one_coords.tobytes(): algebra.one_coords}
    frontier = [algebra.one_coords]
    while frontier:
        current = frontier.pop()
        for coords in units.generators:
            gen_coords = _solve_coords_cached(algebra, coords)
            product = algebra.coords_multiply(current, gen_coords)
            key = product.tobytes()
            if key not in elements:
                elements[key] = product
                frontier.append(product)
    if len(elements) != units.order:
        return {
            "status": "unknown",
            "unit_group_order": units.order,
            "symplectic_group_order": None,
            "logical_group": {"computed": False, "status": "unknown"},
            "detail": "group closure disagreed with certified order",
        }

    symplectic_entries = []
    for coords in elements.values():
        entries = (coords @ algebra.basis) % 2
        ok = True
        for (start, cell_width) in layout:
            size = 2 * cell_width
            block = entries[start : start + size * size].reshape(size, size)
            form = forms[cell_width]
            if not np.array_equal((block @ form @ block.T) % 2, form):
                ok = False
                break
        if ok:
            symplectic_entries.append(entries.astype(np.uint8))

    analysis = PartitionCliffordAnalysis.__new__(PartitionCliffordAnalysis)
    analysis.code = code
    analysis.cells = [tuple(c) for c in cells]
    analysis.algebra = algebra_basis
    analysis._layout = layout
    analysis.enumeration_complete = True
    analysis.structured = True
    analysis.elements = symplectic_entries
    analysis.generators = tuple(
        analysis._describe(entries)
        for entries in symplectic_entries
        if not analysis._is_identity(entries)
    )
    nontrivial = [
        g.logical_symplectic for g in analysis.generators if not g.is_logical_identity
    ]
    logical = logical_group_summary(nontrivial, code.k)
    logical["status"] = "exact" if logical["exact"] else "lower_bound"
    return {
        "status": "exact",
        "unit_group_order": units.order,
        "symplectic_group_order": len(symplectic_entries),
        "logical_group": logical,
        "certified": all(all(g.certificate.values()) for g in analysis.generators),
        "detail": units.detail,
        # The populated analysis object, so callers (notably
        # :func:`qec_transversal.api.partition_clifford_group`) can hand back
        # the certified generators and not merely the counts.  Not JSON-safe;
        # every other key is.
        "analysis": analysis,
    }


def _solve_coords_cached(algebra, coords_or_entries):
    """Unit-group generators are algebra coordinates already."""

    return np.asarray(coords_or_entries, dtype=np.uint8)


def _involution_matrix(algebra, layout) -> np.ndarray | None:
    """Coordinate matrix of ``sigma`` on ``A``, or ``None`` if ``A`` is not stable.

    ``partition_algebra(refine=True)`` returns ``A'(S) = A(S) cap A(N)``, which
    is ``sigma``-stable by construction, so ``None`` here means the caller asked
    for the unrefined algebra (or something is wrong) -- either way the orbit
    route must decline rather than guess.
    """

    columns = []
    for index in range(algebra.dim):
        image = symplectic_involution(algebra.basis[index], layout)
        coords = _solve_coords(algebra.basis, image)
        if coords is None:
            return None
        columns.append(np.asarray(coords, dtype=np.uint8))
    return np.array(columns, dtype=np.uint8).T


def _phi_orbit_summary(
    code: StabilizerCode,
    cells: list[tuple[int, ...]],
    algebra,
    algebra_basis: np.ndarray,
    layout: list[tuple[int, int]],
    units,
    *,
    orbit_cap: int,
) -> dict:
    """The symplectic cut as an index computation.  See :mod:`..algebra.unitary_group`."""

    sigma = _involution_matrix(algebra, layout)
    if sigma is None:
        return {
            "status": "unknown",
            "unit_group_order": units.order,
            "symplectic_group_order": None,
            "logical_group": {"computed": False, "status": "unknown"},
            "detail": "the partition algebra is not stable under sigma",
        }

    n = code.n
    cell_tuples = [tuple(cell) for cell in cells]

    def entries_of(coords: np.ndarray) -> np.ndarray:
        # coordinates are over ``algebra.basis`` (row-reduced), not over the
        # raw ``algebra_basis`` handed to the constructor
        return ((np.asarray(coords, dtype=np.uint8) @ algebra.basis) % 2).astype(np.uint8)

    probe_state = {"count": 0, "seen": 0}

    def order_probe(collected: list[np.ndarray]) -> int | None:
        # Schreier-Sims on the faithful physical action.  Only worth running
        # when the generating set actually grew, and only a bounded number of
        # times: it is an accelerator, never part of the certificate.
        if len(collected) == probe_state["seen"] or probe_state["count"] >= 24:
            return None
        probe_state["seen"] = len(collected)
        probe_state["count"] += 1
        matrices = [
            _partition_action_matrix(entries_of(coords), cell_tuples, layout, n)
            for coords in collected
        ]
        return schreier_sims_order(matrices, node_budget=200_000)

    result = unitary_group(
        algebra,
        sigma,
        unit_order=units.order,
        unit_generators=units.generators,
        orbit_cap=orbit_cap,
        order_probe=order_probe,
    )
    if result.status != "exact":
        return {
            "status": "unknown",
            "unit_group_order": units.order,
            "symplectic_group_order": None,
            "orbit": result.to_dict(),
            "logical_group": {"computed": False, "status": "unknown"},
            "detail": result.detail,
        }

    generator_entries = [entries_of(coords) for coords in result.generators]
    analysis = PartitionCliffordAnalysis.__new__(PartitionCliffordAnalysis)
    analysis.code = code
    analysis.cells = cell_tuples
    analysis.algebra = algebra_basis
    analysis._layout = layout
    analysis.enumeration_complete = True
    analysis.structured = True
    analysis.generation = "phi-orbit"
    analysis._group_order_override = result.order
    analysis.elements = generator_entries
    analysis.generators = tuple(
        analysis._describe(entries)
        for entries in generator_entries
        if not analysis._is_identity(entries)
    )
    nontrivial = [
        g.logical_symplectic for g in analysis.generators if not g.is_logical_identity
    ]
    logical = logical_group_summary(nontrivial, code.k)
    logical["status"] = "exact" if logical["exact"] else "lower_bound"
    reached = logical.get("order") or logical.get("lower_bound")
    if not result.generators_complete and reached != symplectic_group_order(code.k):
        # |G| stays exact -- it came from the index -- but the logical image is
        # only what these generators reach, so it must be reported as a bound.
        # The one exception is reaching |Sp(2k,2)| itself: the image cannot
        # exceed it, so a lower bound that touches it is the exact answer.
        logical["exact"] = False
        logical["status"] = "lower_bound"
        logical["lower_bound"] = reached
        logical["order"] = None
    return {
        "status": "exact",
        "unit_group_order": units.order,
        "symplectic_group_order": result.order,
        "orbit": result.to_dict(),
        "logical_group": logical,
        "certified": all(all(g.certificate.values()) for g in analysis.generators),
        "generator_records": (
            "certified generating set (Schreier)"
            if result.generators_complete
            else "certified subgroup generators"
        ),
        "detail": result.detail,
        "analysis": analysis,
    }
