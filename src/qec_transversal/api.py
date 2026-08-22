r"""The public API: one call per physical gate ansatz, one result shape.

Every entry point here answers the same question for a different ansatz --
*which physical transformations of this shape preserve the code, what do they
do logically, and is the answer complete?* -- and returns the same
:class:`GateSearchResult`.

    physical gate ansatz
        -> code-preservation constraints
        -> exact algebraic / group solver
        -> logical action
        -> completeness certificate

The completeness field is the point of the package and is never rounded in the
optimistic direction.  A computation that ran out of budget reports
:attr:`Completeness.UNKNOWN`; a search that sampled rather than exhausted its
ansatz reports :attr:`Completeness.INCOMPLETE_LOWER_BOUND`.  Neither is ever
converted into "no such gate exists".

Read ``docs/mathematics.md`` for the framework and ``README.md`` for the scope
of the word *complete*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .codes.css import CSSCode
from .codes.stabilizer import StabilizerCode

__all__ = [
    "Completeness",
    "GateSearchResult",
    "SignCertificate",
    "certify_signs",
    "css_strict_transversal_clifford",
    "diagonal_transversal_gates",
    "matching_clifford_group",
    "monomial_clifford_group",
    "one_block_clifford_group",
    "partition_clifford_group",
    "permutation_automorphism_group",
    "strict_transversal_clifford",
    "transversal_clifford_across_blocks",
]


class Completeness(str, Enum):
    """How much the returned generating set is known to cover.

    The scope is always *the stated ansatz*, never "all fault-tolerant logical
    gates" -- see the README section "What complete means".
    """

    #: The returned set is provably the whole solution set of the ansatz.
    COMPLETE = "COMPLETE"
    #: Everything returned is certified, but the search was capped, sampled, or
    #: scoped to a subgroup; the true answer can only be larger.
    INCOMPLETE_LOWER_BOUND = "INCOMPLETE_LOWER_BOUND"
    #: A verification or budget failed.  Nothing may be concluded, in either
    #: direction, from this result.
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class GateSearchResult:
    """The uniform result of every gate search in this package."""

    #: Which solver ran, e.g. ``"css-shear-kernel"`` or ``"preservation-algebra-units"``.
    method: str
    #: The physical gate class searched, in words.
    ansatz: str
    #: Coverage of the returned set with respect to ``ansatz``.
    completeness: Completeness
    #: Backend-specific generator records, each carrying its own certificate.
    #:
    #: Caution on the name, which predates this API and is kept for
    #: compatibility: enumeration-based routes put *every non-identity group
    #: element* here (a generating set a fortiori, just not a small one),
    #: while structured routes put a certified minimal generating set.
    #: ``group_order`` is the authoritative size in both cases -- never
    #: ``len(generators)``.  ``metadata["generator_records"]`` says which of
    #: the two you got.
    generators: tuple[Any, ...] = ()
    #: Logical symplectic images in ``Sp(2k, 2)``, identities removed.
    logical_generators: tuple[np.ndarray, ...] = ()
    #: Order of the physical group modulo Paulis, or ``None`` when not computed.
    group_order: int | None = None
    #: Order of the logical image, or ``None``.
    logical_group_order: int | None = None
    #: Whether ``logical_group_order`` is exact (else it is a lower bound).
    logical_group_order_is_exact: bool = False
    #: Machine-checkable facts established during the computation.
    certificate: dict[str, Any] = field(default_factory=dict)
    #: Everything else worth reporting: caps hit, dimensions, scope notes.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The underlying analysis object, for callers that need backend detail.
    analysis: Any = None

    @property
    def complete(self) -> bool:
        """``True`` only for :attr:`Completeness.COMPLETE`."""

        return self.completeness is Completeness.COMPLETE

    @property
    def logical_group_lower_bound(self) -> int | None:
        """A number the logical group order is guaranteed to be at least."""

        return self.logical_group_order

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (logical matrices as 0/1 nested lists)."""

        return {
            "method": self.method,
            "ansatz": self.ansatz,
            "completeness": self.completeness.value,
            "complete": self.complete,
            "generator_count": len(self.generators),
            "logical_generators": [
                matrix.astype(int).tolist() for matrix in self.logical_generators
            ],
            "group_order": self.group_order,
            "logical_group_order": self.logical_group_order,
            "logical_group_order_is_exact": self.logical_group_order_is_exact,
            "certificate": self.certificate,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SignCertificate:
    """Outcome of lifting a result's generators to exact signed circuits.

    The symplectic picture works modulo Pauli operators and global phase.  That
    is sound for group orders -- the sign defect of a Clifford on the
    stabilizer group is a linear character, so a Pauli correction always exists
    and never changes the symplectic action -- but it is not a circuit-level
    claim.  This closes the gap: each generator becomes an exact Stim tableau,
    is conjugated through the signed generators, is given an explicit Pauli
    correction, and is re-verified to fix every stabilizer generator with sign
    ``+1``.
    """

    #: Generators lifted and verified.
    checked: int
    #: Generators whose record carried no dense physical matrix to lift.
    skipped: int
    #: Every checked generator fixed the stabilizer row space.
    all_preserved: bool
    #: Every checked generator, after its Pauli correction, fixes every
    #: stabilizer generator with sign ``+1``.
    all_signs_plus: bool
    #: One Pauli correction string per checked generator, in order.
    corrections: tuple[str, ...]
    detail: str

    @property
    def certified(self) -> bool:
        """``True`` only if something was checked and everything passed."""

        return bool(self.checked and self.all_preserved and self.all_signs_plus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "skipped": self.skipped,
            "all_preserved": self.all_preserved,
            "all_signs_plus": self.all_signs_plus,
            "certified": self.certified,
            "corrections": list(self.corrections),
            "detail": self.detail,
        }


def _physical_symplectic(record: Any, qubits: int) -> np.ndarray | None:
    """The dense ``2n x 2n`` physical matrix of a generator record, or None.

    Shape is the discriminator on purpose: :class:`OneBlockGenerator` also has
    a ``matrix`` attribute, but it holds the ``2k x 2k`` *logical* action, and
    mistaking one for the other would silently verify the wrong object.
    """

    matrix = getattr(record, "physical_symplectic", None)
    if matrix is None:
        matrix = getattr(record, "matrix", None)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.uint8)
    if matrix.shape != (2 * qubits, 2 * qubits):
        return None
    return matrix


def certify_signs(
    code: CSSCode | StabilizerCode,
    result: GateSearchResult,
    *,
    signs: Sequence[int] | None = None,
    limit: int | None = None,
) -> SignCertificate:
    """Verify a result's generators sign-exactly, as executable circuits.

    Works for any backend whose records carry a dense physical matrix -- that
    is every gate ansatz in the package.  ``one_block_clifford_group`` is the
    exception by construction: it collects *logical* actions rather than
    physical layers, so its records are skipped and the certificate says so.

    Needs the optional ``stim`` dependency.
    """

    from .certificates.signed import SignedStabilizer, verify_sign_exact

    stabilizer = _as_stabilizer_code(code)
    signed = SignedStabilizer(stabilizer.h, list(signs) if signs is not None else None)
    records = result.generators if limit is None else result.generators[:limit]

    checked = skipped = 0
    all_preserved = all_plus = True
    corrections: list[str] = []
    for record in records:
        matrix = _physical_symplectic(record, stabilizer.n)
        if matrix is None:
            skipped += 1
            continue
        outcome = verify_sign_exact(signed, matrix)
        checked += 1
        corrections.append(outcome.pauli_correction)
        if not outcome.preserved:
            all_preserved = False
        if not outcome.certificate.get("stabilizer_signs_corrected_to_plus", False):
            all_plus = False
    detail = (
        f"{checked} generator(s) lifted to exact tableaux and sign-verified"
        + (f"; {skipped} skipped (no physical matrix on the record)" if skipped else "")
    )
    return SignCertificate(
        checked=checked,
        skipped=skipped,
        all_preserved=all_preserved,
        all_signs_plus=all_plus,
        corrections=tuple(corrections),
        detail=detail,
    )


def _nontrivial(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    return tuple(
        matrix
        for matrix in matrices
        if matrix.size and not np.array_equal(matrix, np.eye(matrix.shape[0], dtype=np.uint8))
    )


def _as_stabilizer_code(code: CSSCode | StabilizerCode) -> StabilizerCode:
    if isinstance(code, StabilizerCode):
        return code
    if isinstance(code, CSSCode):
        return code.to_stabilizer_code()
    raise TypeError("expected a CSSCode or a StabilizerCode")


# ---------------------------------------------------------------------------
# Strict (site-dependent single-qubit) transversal Clifford gates
# ---------------------------------------------------------------------------


def strict_transversal_clifford(
    code: CSSCode | StabilizerCode, *, method: str = "auto", dim_cap: int = 24
) -> GateSearchResult:
    """Depth-one layers of one arbitrary single-qubit Clifford per qubit.

    ``method="auto"`` (default) picks the specialised CSS shear solver for a
    :class:`~.codes.css.CSSCode` and the general preservation-algebra solver
    for a :class:`~.codes.stabilizer.StabilizerCode`.  Force one with
    ``method="css"`` or ``method="general"``; both are exact and, on CSS
    codes, must agree -- ``tests/test_cross_validation.py`` checks that they
    do.
    """

    if method not in ("auto", "css", "general"):
        raise ValueError("method must be 'auto', 'css', or 'general'")
    if method == "auto":
        method = "css" if isinstance(code, CSSCode) else "general"
    if method == "css":
        if not isinstance(code, CSSCode):
            raise TypeError("the CSS solver needs a CSSCode")
        return css_strict_transversal_clifford(code)
    return _strict_general(_as_stabilizer_code(code), dim_cap=dim_cap)


def _strict_general(code: StabilizerCode, *, dim_cap: int) -> GateSearchResult:
    from .ansatz.strict import LocalCliffordAnalysis

    analysis = LocalCliffordAnalysis(code, dim_cap=dim_cap)
    summary = analysis.to_dict()
    logical = summary["logical_group"]
    exact = analysis.status == "exact"
    return GateSearchResult(
        method="preservation-algebra units (singleton partition)",
        ansatz="one arbitrary single-qubit Clifford per qubit, modulo Pauli",
        completeness=Completeness.COMPLETE if exact else Completeness.UNKNOWN,
        generators=analysis.generators,
        logical_generators=_nontrivial(
            [g.logical_symplectic for g in analysis.generators]
        ),
        group_order=analysis.group_order,
        logical_group_order=logical.get("order") or logical.get("lower_bound"),
        logical_group_order_is_exact=bool(logical.get("exact")),
        certificate={
            "certified": analysis.certified,
            "route": "structured" if analysis.structured_order is not None else "enumeration",
        },
        metadata={
            "n": code.n,
            "k": code.k,
            "algebra_dimension": summary["algebra_dimension"],
            "enumeration_dim_cap": dim_cap,
            "generator_records": (
                "certified generating set"
                if analysis.structured_order is not None
                else "all non-identity group elements"
            ),
            "note": (
                "GL(2,2) = Sp(2,2), so the strict group is exactly the unit "
                "group of the singleton-partition preservation algebra"
            ),
        },
        analysis=analysis,
    )


def css_strict_transversal_clifford(code: CSSCode) -> GateSearchResult:
    """The CSS specialisation: the two diagonal shear parameter spaces.

    Complete for the strict ansatz -- and cheap, because both parameter spaces
    are ordinary GF(2) kernels with no enumeration at all.
    """

    analysis = code.analyze_transversal()
    summary = analysis.to_dict()
    logical = summary["logical_group"]
    return GateSearchResult(
        method="css-shear-kernel",
        ansatz="one arbitrary single-qubit Clifford per qubit, modulo Pauli",
        completeness=Completeness.COMPLETE if analysis.certified else Completeness.UNKNOWN,
        generators=analysis.generators,
        logical_generators=_nontrivial(
            [g.logical_symplectic for g in analysis.generators]
        ),
        group_order=None,
        logical_group_order=logical["order"] if logical["exact"] else logical["lower_bound"],
        logical_group_order_is_exact=bool(logical["exact"]),
        certificate=summary["certificate"],
        metadata={
            "n": code.n,
            "k": code.k,
            "dim_A_Z": analysis.a_z.dimension,
            "dim_A_X": analysis.a_x.dimension,
            "structure": summary["structure"],
            "generator_records": "one per parameter-space basis vector",
            "physical_group_order_note": (
                "not computed by this route; use method='general' for the "
                "exact physical group order modulo Pauli"
            ),
        },
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Prescribed-partition Clifford gates: the general framework
# ---------------------------------------------------------------------------


def partition_clifford_group(
    code: CSSCode | StabilizerCode,
    partition: Sequence[Sequence[int]],
    *,
    method: str = "auto",
    dim_cap: int = 24,
) -> GateSearchResult:
    r"""Depth-one layers of one arbitrary Clifford per partition cell.

    The solution set is ``A_P(S)^x \cap prod_C Sp(2|C|, 2)`` -- the symplectic
    units of the preservation algebra.  ``method="enumeration"`` sweeps
    ``2^{dim A}``; ``method="structure"`` computes the certified unit group and
    enumerates *it* instead; ``method="phi"`` takes the symplectic cut as an
    index, ``|G| = |A^x| / |orbit of 1|`` under ``a . u = sigma(u) a u``, which
    is the only route that survives cells wider than three qubits;
    ``"auto"`` uses the structured route, switching to the index when the unit
    group is too large to sweep, and falls back to enumeration otherwise.
    Singleton cells recover :func:`strict_transversal_clifford`.
    """

    if method not in ("auto", "enumeration", "structure", "phi"):
        raise ValueError(
            "method must be 'auto', 'enumeration', 'structure', or 'phi'"
        )
    stabilizer = _as_stabilizer_code(code)
    cells = [tuple(int(q) for q in cell) for cell in partition]
    ansatz = "one arbitrary Clifford per prescribed partition cell, modulo Pauli"

    if method in ("auto", "structure", "phi"):
        from .ansatz.partition import partition_units_via_structure

        summary = partition_units_via_structure(
            stabilizer, cells, method="phi" if method == "phi" else "auto"
        )
        if summary.get("status") == "exact":
            logical = summary.get("logical_group", {})
            backend = summary.get("analysis")
            records = tuple(getattr(backend, "generators", ()) or ())
            return GateSearchResult(
                method="preservation-algebra units (structured)",
                ansatz=ansatz,
                completeness=Completeness.COMPLETE,
                generators=records,
                logical_generators=_nontrivial(
                    [g.logical_symplectic for g in records]
                ),
                group_order=summary.get("symplectic_group_order"),
                logical_group_order=logical.get("order") or logical.get("lower_bound"),
                logical_group_order_is_exact=bool(logical.get("exact")),
                certificate={"certified": bool(summary.get("certified", False))},
                metadata={
                    "n": stabilizer.n,
                    "k": stabilizer.k,
                    "cells": [list(cell) for cell in cells],
                    "unit_group_order": summary.get("unit_group_order"),
                    "orbit": summary.get("orbit"),
                    "generator_records": summary.get(
                        "generator_records", "all non-identity group elements"
                    ),
                    "detail": summary.get("detail", ""),
                },
                analysis=backend if backend is not None else summary,
            )
        if method in ("structure", "phi"):
            return GateSearchResult(
                method="preservation-algebra units (structured)",
                ansatz=ansatz,
                completeness=Completeness.UNKNOWN,
                certificate={"certified": False},
                metadata={
                    "n": stabilizer.n,
                    "k": stabilizer.k,
                    "cells": [list(cell) for cell in cells],
                    "detail": summary.get("detail", ""),
                },
                analysis=summary,
            )

    from .ansatz.partition import PartitionCliffordAnalysis

    analysis = PartitionCliffordAnalysis(stabilizer, cells, dim_cap=dim_cap)
    summary = analysis.to_dict()
    logical = summary["logical_group"]
    exact = analysis.status == "exact"
    return GateSearchResult(
        method="preservation-algebra units (enumeration)",
        ansatz=ansatz,
        completeness=Completeness.COMPLETE if exact else Completeness.UNKNOWN,
        generators=analysis.generators,
        logical_generators=_nontrivial(
            [g.logical_symplectic for g in analysis.generators]
        ),
        group_order=analysis.group_order,
        logical_group_order=logical.get("order") or logical.get("lower_bound"),
        logical_group_order_is_exact=bool(logical.get("exact")),
        certificate={"certified": analysis.certified},
        metadata={
            "n": stabilizer.n,
            "k": stabilizer.k,
            "cells": [list(cell) for cell in cells],
            "algebra_dimension": summary["algebra_dimension"],
            "enumeration_dim_cap": dim_cap,
            "generator_records": "all non-identity group elements",
        },
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Specialised backends
# ---------------------------------------------------------------------------


def transversal_clifford_across_blocks(
    code: CSSCode | StabilizerCode,
    *,
    blocks: int = 2,
    method: str = "auto",
    dim_cap: int = 24,
) -> GateSearchResult:
    r"""Depth-one gates acting across ``blocks`` copies of a code.

    The classical notion of "transversal": one gate per *corresponding-qubit
    tuple* across ``l`` blocks, e.g. transversal CNOT between two blocks.  It
    needs no new mathematics -- it is the prescribed-partition problem for the
    joint code ``S^{\otimes l}`` with cells ``{i, n+i, ..., (l-1)n+i}`` -- so
    the same exact solver and the same completeness semantics apply.

    Worked examples, all ``COMPLETE``: two Steane blocks realise the whole
    ``Sp(4,2)`` logical Clifford group on their two logical qubits (order 720),
    while two ``[[5,1,3]]`` blocks reach only order 18 of the same 720 -- a
    *certified* negative, not a failed search.

    ``blocks=1`` reduces to :func:`strict_transversal_clifford`.
    """

    from .codes.stabilizer import corresponding_qubit_cells, tensor_power

    if blocks < 1:
        raise ValueError("blocks must be positive")
    single = _as_stabilizer_code(code)
    joint = tensor_power(single, blocks)
    cells = corresponding_qubit_cells(joint.n, blocks)
    result = partition_clifford_group(joint, cells, method=method, dim_cap=dim_cap)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "blocks": blocks,
            "qubits_per_block": single.n,
            "logical_qubits_per_block": single.k,
            "joint_n": joint.n,
            "joint_k": joint.k,
        }
    )
    return GateSearchResult(
        method=result.method,
        ansatz=(
            f"depth-one gate on each corresponding-qubit tuple across "
            f"{blocks} code blocks, modulo Pauli"
        ),
        completeness=result.completeness,
        generators=result.generators,
        logical_generators=result.logical_generators,
        group_order=result.group_order,
        logical_group_order=result.logical_group_order,
        logical_group_order_is_exact=result.logical_group_order_is_exact,
        certificate=result.certificate,
        metadata=metadata,
        analysis=result.analysis,
    )


def matching_clifford_group(code: CSSCode, tau: object) -> GateSearchResult:
    """CSS fold-transversal layers on one fixed matching ``tau``.

    Complete for *diagonal* layers on that matching (the ``S_M^Z``/``S_M^X``
    families) plus the fold Hadamard when ``tau`` certifies as a ZX-duality.
    The Levi (CNOT-network) factor of the fixed-matching group is **not**
    included here; :func:`qec_transversal.ansatz.twofold.two_fold_group` adds
    it.
    """

    from .ansatz.matching import analyze_matching

    analysis = analyze_matching(code, tau)
    summary = analysis.to_dict()
    logical = summary["logical_group"]
    generators = [g.logical_symplectic for g in analysis.generators]
    if analysis.fold_hadamard is not None:
        generators.append(analysis.fold_hadamard.logical_symplectic)
    return GateSearchResult(
        method="fixed-matching diagonal kernels",
        ansatz=(
            "depth-one diagonal layer on a fixed matching (S/CZ and sqrt(X)/XX), "
            "plus the fold Hadamard; Levi/CNOT factor excluded"
        ),
        completeness=Completeness.COMPLETE if analysis.certified else Completeness.UNKNOWN,
        generators=analysis.generators,
        logical_generators=_nontrivial(generators),
        logical_group_order=logical["order"] if logical["exact"] else logical["lower_bound"],
        logical_group_order_is_exact=bool(logical["exact"]),
        certificate={"certified": analysis.certified, "zx_duality": analysis.is_zx_duality},
        metadata={
            "n": code.n,
            "k": code.k,
            "pairs": summary["pairs"],
            "fixed_points": summary["fixed_points"],
            "dim_S_MZ": summary["dim_S_MZ"],
            "dim_S_MX": summary["dim_S_MX"],
        },
        analysis=analysis,
    )


def diagonal_transversal_gates(
    code: CSSCode | StabilizerCode, *, level: int = 3, family: str = "Z"
) -> GateSearchResult:
    """Strict single-qubit diagonal gates at Clifford-hierarchy level ``level``.

    The solution set ``{t in Z_{2^L}^n : U(t) preserves the code}`` is a
    submodule of ``Z_{2^L}^n``, computed exactly by elimination over the local
    ring -- no search.

    For a :class:`~.codes.css.CSSCode` the coset-phase ladder of
    :mod:`.hierarchy.css` is **complete at every level**; the hierarchy-*level*
    labelling of each generator can still be incomplete at very large ``k``,
    which ``metadata["levels_complete"]`` reports.  ``family`` selects the
    Z-diagonal or X-diagonal family and applies to CSS codes only.

    For a general :class:`~.codes.stabilizer.StabilizerCode` the operator-level
    solver of :mod:`.hierarchy.general` is used.  It is always **sound**, and
    complete when the exact support-coset enumeration fits under its cap or the
    code has no Z-type stabilizers; otherwise the result is a certified
    subgroup and reports ``INCOMPLETE_LOWER_BOUND``.  This route needs the
    optional ``stim`` dependency.
    """

    if not isinstance(code, CSSCode):
        return _diagonal_general(_as_stabilizer_code(code), level=level, family=family)

    from .hierarchy.css import analyze_hierarchy

    analysis = analyze_hierarchy(code, family, level=level)
    summary = analysis.to_dict()
    return GateSearchResult(
        method="Z_{2^L} coset-phase module kernel",
        ansatz=f"depth-one diagonal layer of level-{level} single-qubit phase gates",
        completeness=Completeness.COMPLETE if analysis.certified else Completeness.UNKNOWN,
        generators=analysis.generators,
        logical_generators=(),
        certificate={
            "certified": analysis.certified,
            "kernel_completeness": "Smith-form certificate via analysis.kernel_certificate()",
        },
        metadata={
            "n": code.n,
            "k": code.k,
            "family": family,
            "level": level,
            "kernel_generators": summary["kernel_generators"],
            "max_level": summary["max_level"],
            "has_level_gate": summary["has_t_level_gate"],
            "levels_complete": summary["levels_complete"],
        },
        analysis=analysis,
    )


def _diagonal_general(
    code: StabilizerCode, *, level: int, family: str
) -> GateSearchResult:
    from .hierarchy.general import diagonal_kernel_general_exact

    if family != "Z":
        raise ValueError(
            "the general-stabilizer diagonal solver has no X/Z family split; "
            "conjugate the code with hierarchy.frames.frame_conjugated_code "
            "and pass family='Z'"
        )
    kernel, complete = diagonal_kernel_general_exact(code, level=level)
    modulus = 1 << level
    nontrivial = int(sum(1 for row in kernel if (row % (modulus // 2)).any()))
    return GateSearchResult(
        method="Z_{2^L} operator congruences (general stabilizer)",
        ansatz=f"depth-one diagonal layer of level-{level} single-qubit phase gates",
        completeness=(
            Completeness.COMPLETE if complete else Completeness.INCOMPLETE_LOWER_BOUND
        ),
        generators=tuple(np.asarray(row) for row in kernel),
        logical_generators=(),
        certificate={
            "sound": True,
            "complete": bool(complete),
            "note": (
                "sound always; complete when the exact support-coset "
                "enumeration fits under its cap or the code has no Z-type "
                "stabilizers"
            ),
        },
        metadata={
            "n": code.n,
            "k": code.k,
            "family": family,
            "level": level,
            "kernel_generators": int(kernel.shape[0]),
            "beyond_pauli_generators": nontrivial,
            "generator_records": "kernel basis vectors t in Z_{2^L}^n",
        },
        analysis=kernel,
    )


def monomial_clifford_group(
    code: CSSCode | StabilizerCode, *, natural_rows: np.ndarray | None = None
) -> GateSearchResult:
    """Qubit permutations combined with per-qubit Cliffords (the GF(4) route).

    Exact -- the true monomial automorphism group of the code -- when the
    stabilizer group is small enough to enumerate every nonzero element
    (``metadata["row_set_complete"]``).  Otherwise the computation is scoped to
    the *given generating rows*, a strictly stronger condition than preserving
    the stabilizer group, so the answer is a certified subgroup and the result
    reports :attr:`Completeness.INCOMPLETE_LOWER_BOUND`.

    Needs the optional ``python-igraph`` dependency.
    """

    from .ansatz.monomial import MonomialAnalysis

    stabilizer = _as_stabilizer_code(code)
    analysis = MonomialAnalysis(stabilizer, natural_rows=natural_rows)
    summary = analysis.to_dict()
    logical = summary["logical_group"]
    return GateSearchResult(
        method="CRSS GF(4) monomial automorphisms (BLISS)",
        ansatz="qubit permutation composed with one single-qubit Clifford per qubit",
        completeness=(
            Completeness.COMPLETE
            if analysis.row_set_complete and analysis.certified
            else Completeness.INCOMPLETE_LOWER_BOUND
        ),
        generators=analysis.generators,
        logical_generators=_nontrivial(
            [g.logical_symplectic for g in analysis.generators]
        ),
        group_order=int(analysis.group_order),
        logical_group_order=logical["order"] if logical["exact"] else logical["lower_bound"],
        logical_group_order_is_exact=bool(logical["exact"]),
        certificate={"certified": analysis.certified},
        metadata={
            "n": stabilizer.n,
            "k": stabilizer.k,
            "row_set_complete": bool(analysis.row_set_complete),
            "permutation_image_order": summary["permutation_image_order"],
        },
        analysis=analysis,
    )


def permutation_automorphism_group(
    code: CSSCode, *, method: str = "codewords"
) -> GateSearchResult:
    """SWAP-class gates: qubit permutations preserving the code.

    ``method="codewords"`` computes the *row-space* automorphism group
    ``{pi : C_X pi = C_X, C_Z pi = C_Z}`` through characteristic bounded-weight
    codeword sets; it is complete when the classes are enumerated exactly and
    span.  ``method="tanner"`` computes the automorphism group of the Tanner
    graph of the checks *as given*, which is a subgroup of the row-space group
    and is therefore always reported as a lower bound.

    Needs the optional ``python-igraph`` dependency.
    """

    if method not in ("codewords", "tanner"):
        raise ValueError("method must be 'codewords' or 'tanner'")
    if method == "tanner":
        from .ansatz.permutation import analyze_automorphisms

        analysis = analyze_automorphisms(code)
        summary = analysis.to_dict()
        logical = summary["logical_group"]
        return GateSearchResult(
            method="Tanner-graph automorphisms (BLISS)",
            ansatz="qubit permutations preserving both check row spaces",
            completeness=Completeness.INCOMPLETE_LOWER_BOUND,
            generators=analysis.generators,
            logical_generators=_nontrivial(
                [g.logical_symplectic for g in analysis.generators]
            ),
            group_order=int(analysis.group_order),
            logical_group_order=logical["order"] if logical["exact"] else logical["lower_bound"],
            logical_group_order_is_exact=bool(logical["exact"]),
            certificate={"certified": analysis.certified},
            metadata={
                "n": code.n,
                "k": code.k,
                "scope": (
                    "row-SET automorphisms of the checks as given; a symmetry "
                    "permuting the row SPACE without fixing these rows is "
                    "invisible to the Tanner graph"
                ),
                "duality_exists": summary["duality_exists"],
            },
            analysis=analysis,
        )

    from .ansatz.codeword_permutation import analyze_codeword_automorphisms
    from .logical.group import logical_group_summary

    analysis = analyze_codeword_automorphisms(code)
    logical_matrices = _nontrivial([g.logical_symplectic for g in analysis.generators])
    logical = logical_group_summary(list(logical_matrices), code.k)
    return GateSearchResult(
        method="characteristic-codeword automorphisms (BLISS)",
        ansatz="qubit permutations preserving both check row spaces",
        completeness=(
            Completeness.COMPLETE if analysis.exact else Completeness.INCOMPLETE_LOWER_BOUND
        ),
        generators=analysis.generators,
        logical_generators=logical_matrices,
        group_order=int(analysis.group_order),
        logical_group_order=logical["order"] if logical["exact"] else logical["lower_bound"],
        logical_group_order_is_exact=bool(logical["exact"]),
        certificate={"certified": analysis.certified},
        metadata={
            "n": code.n,
            "k": code.k,
            "weight_cap_X": analysis.weight_cap_x,
            "weight_cap_Z": analysis.weight_cap_z,
            "codewords_X": analysis.codewords_x,
            "codewords_Z": analysis.codewords_z,
            "notes": list(analysis.notes),
        },
        analysis=analysis,
    )


def one_block_clifford_group(
    code: CSSCode,
    *,
    name: str | None = None,
    involution_cap: int = 16,
    time_budget_s: float = 120.0,
    seed: int = 7,
) -> GateSearchResult:
    """The logical group generated by *all* depth-one layers of one code block.

    Strict shears, a fold layer over every certified matching, and permutation
    gates, taken together.  Completeness is one-sided by construction:
    involutions are *sampled*, so reaching ``|Sp(2k, 2)|`` is a certificate of
    fullness (:attr:`Completeness.COMPLETE`) while anything short of it is a
    lower bound on what the code admits, never a no-go.
    """

    from .logical.generated import analyze_one_block

    analysis = analyze_one_block(
        code,
        name=name,
        involution_cap=involution_cap,
        time_budget_s=time_budget_s,
        seed=seed,
    )
    summary = analysis.to_dict()
    if analysis.is_full:
        completeness = Completeness.COMPLETE
    elif analysis.logical_order_exact:
        completeness = Completeness.INCOMPLETE_LOWER_BOUND
    else:
        completeness = Completeness.UNKNOWN
    return GateSearchResult(
        method=f"one-block varying-partition ({analysis.certification})",
        ansatz=(
            "products of depth-one one-block layers: strict shears, fold layers "
            "over sampled matchings, and permutation gates"
        ),
        completeness=completeness,
        generators=analysis.generators,
        logical_generators=tuple(record.matrix for record in analysis.generators),
        logical_group_order=analysis.logical_order,
        logical_group_order_is_exact=bool(analysis.logical_order_exact),
        certificate={
            "is_full_symplectic": analysis.is_full,
            "certification_tier": analysis.certification,
        },
        metadata={
            "n": code.n,
            "k": code.k,
            "sp_target": summary["sp_target"],
            "involutions_used": summary["involutions_used"],
            "detail": analysis.detail,
            "one_sided": (
                "is_full is True or undecided, never False: involutions are "
                "sampled, so a short group means the sample did not suffice"
            ),
        },
        analysis=analysis,
    )
