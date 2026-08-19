"""Blind structural discovery of matchings and permutation symmetries.

The 2026-08-19 cross-check against the arXiv:2608.05688 census (repo
``memory/2026-08-19.md``) found the one-block engine reporting order 1 on
quasi-cyclic codes it had no registry name for, while certified depth-one
groups of order 60 to 5x10^11 existed on the same codes.  Every missing
matching had *negation* shape — ``r -> (c - r) mod b`` inside cyclic blocks,
optionally swapping whole blocks — which is inferable from the check
matrices alone.  This module does that inference:

1. **Shift certification.**  For divisors ``s | b | n``, the blockwise shift
   by ``s`` inside contiguous ``b``-blocks either preserves both rowspaces
   (a rank identity over GF(2)) or is discarded.  A certified ``(b, 1)``
   exposes 1D cyclic structure; a certified pair ``(b, b2), (b2, 1)``
   exposes 2D ``b1 x b2`` structure (bivariate-bicycle-style tori).

2. **Candidate generation.**  Certified structure yields blockwise
   involutions built from per-orbit residue maps — negation ``r -> c - r``
   (1D and 2D), cross-block shift pairings ``r <-> r + c``, and the
   identity on untouched blocks — combined with involutions of the block
   indices themselves (all of them for at most ``_FULL_BLOCK_ENUM``
   blocks, a canned set beyond).

3. **Screening.**  A candidate survives only if the permutation certifies
   as a ZX-duality (``C_X tau = C_Z``) or as an automorphism (preserves
   both rowspaces).  Membership in a rowspace is a nullspace orthogonality
   test, and that test is ADDITIVE over the block orbits, so per-orbit
   offsets are searched exactly by meet-in-the-middle over the orbit
   halves instead of enumerating the full offset product (the ``9^6``
   mixed-offset matchings of the census Kasai codes are found in
   milliseconds this way).  Every survivor is re-checked against the
   direct rank condition before it is forwarded.

Discovery never certifies gates itself: survivors are handed to
:class:`.matching.MatchingAnalysis` (matchings) and
:func:`.automorphisms.describe_permutation` (permutation gates), so a bug
here can cost coverage but never soundness.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from .css import CSSCode
from .gf2 import nullspace, rowspace_residues

#: enumerate every involution of the block indices up to this many blocks
#: (764 involutions at 8); past it only identity/reversal/adjacent swaps.
_FULL_BLOCK_ENUM = 8

#: bail out of a (block-involution, shape) family when either half of the
#: meet-in-the-middle offset search would enumerate more sums than this.
_HALF_ENUM_CAP = 60_000

#: survivors kept per (block-involution, shape) family: highly symmetric
#: codes accept thousands of assignments that generate the same layers, and
#: forwarding them all would just crowd the downstream involution cap.
_FAMILY_CAP = 6


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _shift_perm(n: int, block: int, stride: int) -> np.ndarray:
    """The permutation shifting every contiguous ``block``-block by ``stride``."""

    index = np.arange(n)
    return (index // block) * block + ((index % block) + stride) % block


def _preserves(rows: np.ndarray, basis: np.ndarray, tau: np.ndarray) -> bool:
    """Whether column-permuting ``rows`` by ``tau`` stays inside ``basis``."""

    return not rowspace_residues(rows[:, tau], basis).any()


def certified_shift_structure(code: CSSCode) -> dict[int, list[int]]:
    """``{block: [strides]}`` of certified blockwise cyclic shifts.

    ``(b, s)`` is certified when the shift by ``s`` inside contiguous
    ``b``-blocks preserves both ``C_X`` and ``C_Z`` (rowspace identities).
    A certified ``(b, 1)`` certifies every stride of that block for free
    (powers of a symmetry are symmetries), recorded as stride ``1`` alone.
    """

    n = code.n
    structure: dict[int, list[int]] = {}
    for block in _divisors(n):
        if block < 2:
            continue
        strides: list[int] = []
        unit = _shift_perm(n, block, 1)
        if _preserves(code.c_x, code.c_x, unit) and _preserves(code.c_z, code.c_z, unit):
            strides.append(1)
        else:
            for stride in _divisors(block):
                if stride in (1, block):
                    continue
                tau = _shift_perm(n, block, stride)
                if _preserves(code.c_x, code.c_x, tau) and _preserves(code.c_z, code.c_z, tau):
                    strides.append(stride)
        if strides:
            structure[block] = strides
    return structure


def _block_involutions(count: int) -> list[np.ndarray]:
    """Involutions of ``range(count)``: all of them up to ``_FULL_BLOCK_ENUM``
    blocks, else identity, reversal, and the adjacent-pair swap."""

    identity = np.arange(count)
    if count <= _FULL_BLOCK_ENUM:
        results: list[np.ndarray] = []

        def extend(perm: list[int], remaining: list[int]) -> None:
            if not remaining:
                results.append(np.array(perm_fill(perm), dtype=int))
                return
            first = remaining[0]
            extend(perm + [(first, first)], remaining[1:])
            for other in remaining[1:]:
                rest = [x for x in remaining[1:] if x != other]
                extend(perm + [(first, other), (other, first)], rest)

        def perm_fill(pairs: list[tuple[int, int]]) -> list[int]:
            image = list(range(count))
            for a, b in pairs:
                image[a] = b
            return image

        extend([], list(range(count)))
        return results
    canned = [identity, identity[::-1].copy()]
    if count % 2 == 0:
        swap = identity.copy()
        swap[0::2], swap[1::2] = identity[1::2], identity[0::2]
        canned.append(swap)
    return canned


def _orbit_maps(
    block: int, shape: tuple[int, int], paired: bool
) -> list[tuple[str, np.ndarray]]:
    """Residue maps available to one block orbit.

    A fixed block needs a self-inverse map: the identity or a negation
    ``(u, v) -> (c1 - u, c2 - v)``.  A 2-cycle of blocks applies the map one
    way and its inverse back, so shifts ``(u, v) -> (u + c1, v + c2)`` are
    additionally valid (a shift's inverse is the opposite shift, and both
    negations and shifts of a cyclic structure send certified rows to
    certified rows or not — the screen decides which).
    """

    outer, inner = shape
    residues = np.arange(block)
    u, v = residues // inner, residues % inner
    maps: list[tuple[str, np.ndarray]] = []
    for c_outer, c_inner in product(range(outer), range(inner)):
        maps.append(
            (f"neg({c_outer},{c_inner})", ((c_outer - u) % outer) * inner + (c_inner - v) % inner)
        )
    if paired:
        for c_outer, c_inner in product(range(outer), range(inner)):
            if (c_outer, c_inner) == (0, 0):
                continue  # zero shift duplicates the identity pairing below
            maps.append(
                (
                    f"shift({c_outer},{c_inner})",
                    ((u + c_outer) % outer) * inner + (v + c_inner) % inner,
                )
            )
        maps.append(("id", residues.copy()))
    else:
        maps.append(("id", residues.copy()))
    return maps


def _build_tau(
    n: int, block: int, orbits: list[tuple[int, int]], maps: list[np.ndarray]
) -> np.ndarray:
    """Assemble the qubit involution from per-orbit residue maps."""

    tau = np.arange(n)
    for (j, partner), residue_map in zip(orbits, maps):
        inverse = np.argsort(residue_map)
        tau[j * block : (j + 1) * block] = partner * block + residue_map
        if partner != j:
            tau[partner * block : (partner + 1) * block] = j * block + inverse
    return tau


def _pack_key(matrix: np.ndarray) -> bytes:
    return np.packbits(matrix.reshape(-1)).tobytes()


def _search_family(
    code: CSSCode,
    block: int,
    orbits: list[tuple[int, int]],
    options: list[list[tuple[str, np.ndarray]]],
    targets: list[tuple[np.ndarray, np.ndarray]],
    cap: int,
) -> list[tuple[str, list[np.ndarray]]]:
    """Exact meet-in-the-middle search over per-orbit residue maps.

    ``targets`` is a list of ``(rows, orthogonal)`` conditions requiring
    ``rows[:, tau] @ orthogonal.T = 0`` over GF(2).  Each condition is a sum
    of per-orbit terms, so assignments are found by hashing the partial sums
    of one half of the orbits against the other — the full offset product is
    never enumerated.  Returns up to ``cap`` ``(label, maps)`` assignments.
    """

    def orbit_term(orbit: tuple[int, int], residue_map: np.ndarray) -> np.ndarray:
        j, partner = orbit
        pieces = []
        inverse = np.argsort(residue_map)
        for rows, orthogonal in targets:
            source = rows[:, partner * block : (partner + 1) * block][:, residue_map]
            term = (
                source.astype(np.int64)
                @ orthogonal[:, j * block : (j + 1) * block].astype(np.int64).T
                % 2
            )
            if partner != j:
                back = rows[:, j * block : (j + 1) * block][:, inverse]
                term = (
                    term
                    + back.astype(np.int64)
                    @ orthogonal[:, partner * block : (partner + 1) * block].astype(np.int64).T
                ) % 2
            pieces.append(term.astype(np.uint8))
        return np.concatenate([piece.reshape(-1) for piece in pieces])

    half = max(1, len(orbits) // 2)
    front, back = list(range(half)), list(range(half, len(orbits)))
    front_size = int(np.prod([len(options[i]) for i in front], dtype=np.int64))
    back_size = int(np.prod([len(options[i]) for i in back], dtype=np.int64)) if back else 1
    if front_size > _HALF_ENUM_CAP or back_size > _HALF_ENUM_CAP:
        return []

    terms = [
        [(label, residue_map, orbit_term(orbits[i], residue_map)) for label, residue_map in opts]
        for i, opts in enumerate(options)
    ]

    def half_sums(indices: list[int]):
        combos: list[tuple[list[str], list[np.ndarray], np.ndarray]] = [
            ([], [], np.zeros(terms[0][0][2].shape, dtype=np.uint8))
        ]
        for i in indices:
            combos = [
                (labels + [label], maps + [residue_map], (total ^ term))
                for labels, maps, total in combos
                for label, residue_map, term in terms[i]
            ]
        return combos

    table: dict[bytes, list[tuple[list[str], list[np.ndarray]]]] = {}
    for labels, maps, total in half_sums(front):
        table.setdefault(_pack_key(total), []).append((labels, maps))
    results: list[tuple[str, list[np.ndarray]]] = []
    for labels, maps, total in half_sums(back):
        for front_labels, front_maps in table.get(_pack_key(total), []):
            results.append((",".join(front_labels + labels), front_maps + maps))
            if len(results) >= cap:
                return results
    return results


def discover_involutions(
    code: CSSCode,
    *,
    time_budget_s: float = 20.0,
    forward_cap: int = 48,
) -> list[tuple[str, np.ndarray]]:
    """Screened blockwise involutions of a code's certified cyclic structure.

    Returns ``(label, tau)`` pairs, ZX-dualities first (they carry the
    fold-Hadamard on top of the diagonal layers), then automorphism-type
    matchings, at most ``forward_cap`` in total, drawn round-robin across
    the structural families so no single symmetric family crowds out the
    rest.  Every survivor passed the nullspace screen AND the direct rank
    re-check; the downstream matching analysis re-certifies from scratch,
    so this list is a search frontier, not a trust boundary.
    """

    deadline = time.perf_counter() + time_budget_s
    structure = certified_shift_structure(code)
    n = code.n
    x_rank, z_rank = code.c_x.shape[0], code.c_z.shape[0]
    null_x, null_z = nullspace(code.c_x), nullspace(code.c_z)
    duality_families: list[list[tuple[str, np.ndarray]]] = []
    automorphism_families: list[list[tuple[str, np.ndarray]]] = []
    seen: set[bytes] = set()
    identity = np.arange(n)

    def screened(tau: np.ndarray) -> str | None:
        """Direct rank re-check; classification of a surviving candidate."""

        if np.array_equal(tau, identity) or not np.array_equal(tau[tau], identity):
            return None
        image = code.c_x[:, tau]
        if x_rank == z_rank and not rowspace_residues(image, code.c_z).any():
            return "duality"
        if not rowspace_residues(image, code.c_x).any() and not rowspace_residues(
            code.c_z[:, tau], code.c_z
        ).any():
            return "automorphism"
        return None

    for block in sorted(structure, reverse=True):
        strides = structure[block]
        shapes: list[tuple[int, int]] = []
        if 1 in strides:
            shapes.append((1, block))
        # 2D shapes b = outer x inner need the outer shift (stride = inner in
        # b-blocks) and the inner cycle (stride 1 in inner-blocks) certified.
        for inner in _divisors(block):
            if inner in (1, block):
                continue
            outer = block // inner
            if 1 in structure.get(inner, []) and (1 in strides or inner in strides):
                shapes.append((outer, inner))
        if not shapes:
            continue
        blocks = n // block
        for pi in _block_involutions(blocks):
            if time.perf_counter() > deadline:
                break
            orbits = [(j, int(pi[j])) for j in range(blocks) if int(pi[j]) >= j]
            for shape in shapes:
                options = [
                    _orbit_maps(block, shape, paired=(j != partner)) for j, partner in orbits
                ]
                for kind, targets in (
                    ("duality", [(code.c_x, null_z)] if x_rank == z_rank else []),
                    ("automorphism", [(code.c_x, null_x), (code.c_z, null_z)]),
                ):
                    if not targets:
                        continue
                    family: list[tuple[str, np.ndarray]] = []
                    for label, maps in _search_family(
                        code, block, orbits, options, targets, cap=4 * _FAMILY_CAP
                    ):
                        tau = _build_tau(n, block, orbits, maps)
                        key = tau.tobytes()
                        if key in seen or screened(tau) != kind:
                            continue
                        seen.add(key)
                        text = (
                            f"discovered-{kind} b={block} shape={shape[0]}x{shape[1]} "
                            f"blocks={tuple(int(x) for x in pi)} maps={label}"
                        )
                        family.append((text, tau))
                        if len(family) >= _FAMILY_CAP:
                            break
                    if family:
                        (duality_families if kind == "duality" else automorphism_families).append(
                            family
                        )

    def round_robin(families: list[list[tuple[str, np.ndarray]]]):
        for rank_index in range(max((len(f) for f in families), default=0)):
            for family in families:
                if rank_index < len(family):
                    yield family[rank_index]

    forwarded = list(round_robin(duality_families)) + list(round_robin(automorphism_families))
    return forwarded[:forward_cap]


def structural_permutations(code: CSSCode, *, cap: int = 8) -> list[np.ndarray]:
    """Certified shift permutations of the code's cyclic structure.

    One generator per certified ``(block, stride)`` — enough to generate the
    whole shift group downstream.  Each is re-certified by
    :func:`.automorphisms.describe_permutation` before any gate is trusted.
    """

    perms: list[np.ndarray] = []
    for block, strides in sorted(certified_shift_structure(code).items(), reverse=True):
        for stride in strides:
            perms.append(_shift_perm(code.n, block, stride))
            if len(perms) >= cap:
                return perms
    return perms


__all__ = [
    "certified_shift_structure",
    "discover_involutions",
    "structural_permutations",
]
