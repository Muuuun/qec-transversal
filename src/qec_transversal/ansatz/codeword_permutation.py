"""Row-space permutation automorphisms via characteristic codeword sets.

The Tanner-graph route (:mod:`.permutation`) is row-SET scoped: BLISS sees
the check matrix's rows, so a symmetry that permutes the row SPACE without
fixing the given generating rows is invisible — the failure measured on the
census Kasai [[54,8,4]] code, whose >= 108-element permutation group the
Tanner graph reports as trivial (2026-08-19 cross-check, repo
``memory/2026-08-19.md``).  This module implements the classical fix — the
invariant-set reduction that underlies Leon's 1982 algorithm and its
descendants (Feulner's codecan in SageMath, MAGMA's ``AutomorphismGroup``,
GUAVA's ``desauto``): replace the arbitrary generating rows by a
CHARACTERISTIC set — all codewords of weight at most ``w``, for the smallest
``w`` at which that set spans the code — and hand its coordinate/codeword
incidence graph to BLISS.

Exactness argument.  Weight is permutation-invariant, so every CSS
permutation automorphism preserves each bounded-weight class of ``C_X`` and
of ``C_Z`` and therefore extends to a color-preserving automorphism of the
incidence graph.  Conversely a graph automorphism permutes each class within
itself; if the classes span, its coordinate part maps a spanning subset of
``C_X`` into ``C_X`` (likewise ``C_Z``) and hence preserves both row spaces.
So with a complete enumeration and spanning classes, the coordinate
projection of the graph group EQUALS ``{pi : C_X pi = C_X and
C_Z pi = C_Z}``.  When the size cap stops the classes short of spanning,
projections are merely candidates; every generator is re-certified by
:func:`.permutation.describe_permutation`, and the surviving group is
reported as a certified subgroup (``exact = False``), never as the full
group.

Cost.  The exponential half is codeword enumeration.  Small ranks use
``2^rank`` packed recursive doubling.  Large ranks use the
Brouwer-Zimmermann-style route that also powers Leon's algorithm and MAGMA:
build ``h`` systematic bases on DISJOINT information sets; a codeword of
weight ``w`` restricts to weight ``<= floor(w / h)`` on at least one of them
(the disjoint sets partition at most ``w`` support bits), so enumerating
messages of weight ``<= t`` on every basis yields EVERY codeword of weight
``<= h*t + h - 1`` — completeness at combinatorial rather than exponential
cost (gross, rank 66: ~10^5 combinations instead of 2^66).  The graph half
is BLISS, fast in practice.  The payoff is MAGMA/GAP-class row-space
capability (the backends every other quantum tool shells out to) on an
igraph dependency the repo already carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from ..codes.css import CSSCode
from ..utils.gf2 import row_basis, rref
from ..utils.graph import igraph, require_igraph
from ..utils.permutations import permutation_group_order
from .permutation import AutomorphismGenerator, describe_permutation

_FULL_ENUM_RANK = 24

#: complete weight classes are added until they span or the next class would
#: push the selected-codeword count past this (graph size, BLISS time).
_SIZE_CAP = 50_000

#: bounded-weight route: total message combinations enumerated across all
#: information sets and depths before declaring subgroup scope.
_WORK_CAP = 2_000_000

_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def _pack_basis(basis: np.ndarray, n: int) -> np.ndarray:
    """Basis rows as little-endian packed uint64 lanes, one row per entry."""

    lanes = (n + 63) // 64
    padded = np.zeros((basis.shape[0], lanes * 64), dtype=np.uint8)
    padded[:, :n] = basis
    return np.packbits(padded, axis=1, bitorder="little").view(np.uint64)


def _enumerate_space(packed_basis: np.ndarray) -> np.ndarray:
    """All ``2^rank`` codewords of the packed row space, by recursive doubling."""

    words = np.zeros((1, packed_basis.shape[1]), dtype=np.uint64)
    for row in packed_basis:
        words = np.concatenate([words, words ^ row[None, :]], axis=0)
    return words


def _word_weights(words: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    weights = np.empty(words.shape[0], dtype=np.uint16)
    stride = words.shape[1] * 8
    for start in range(0, words.shape[0], chunk):
        block = words[start : start + chunk].view(np.uint8).reshape(-1, stride)
        weights[start : start + block.shape[0]] = _POPCOUNT[block].sum(axis=1, dtype=np.uint16)
    return weights


def _unpack(words: np.ndarray, n: int) -> np.ndarray:
    stride = words.shape[1] * 8
    return np.unpackbits(
        words.view(np.uint8).reshape(-1, stride), axis=1, bitorder="little"
    )[:, :n]


def _characteristic_classes(
    basis: np.ndarray, n: int, size_cap: int
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """``(codeword bits, weights, spans, weight_cap)`` of the smallest
    spanning union of complete weight classes (or the largest capped one)."""

    rank = basis.shape[0]
    if rank == 0:
        return np.zeros((0, n), np.uint8), np.zeros(0), True, 0
    words = _enumerate_space(_pack_basis(basis, n))
    weights = _word_weights(words)
    selected: list[int] = []
    spans = False
    weight_cap = 0
    for weight in np.unique(weights):
        if weight == 0:
            continue
        members = np.flatnonzero(weights == weight)
        if len(selected) + len(members) > size_cap:
            break
        selected.extend(int(index) for index in members)
        weight_cap = int(weight)
        bits = _unpack(words[np.array(selected)], n)
        if row_basis(bits, ncols=n).shape[0] == rank:
            spans = True
            break
    bits = _unpack(words[np.array(selected)], n) if selected else np.zeros((0, n), np.uint8)
    return bits, weights[np.array(selected, dtype=int)] if selected else np.zeros(0), spans, weight_cap


def _systematic_bases(basis: np.ndarray, n: int) -> list[np.ndarray]:
    """Systematic bases of the row space on pairwise DISJOINT information
    sets, greedily extracted until the leftover columns lose full rank."""

    rank = basis.shape[0]
    bases: list[np.ndarray] = []
    remaining = list(range(n))
    while len(remaining) >= rank:
        order = np.array(remaining + [c for c in range(n) if c not in set(remaining)])
        reduced, pivots = rref(basis[:, order])
        if len(pivots) < rank or max(pivots) >= len(remaining):
            break  # leftover columns no longer support a full information set
        systematic = np.zeros_like(basis)
        systematic[:, order] = reduced
        bases.append(systematic)
        pivot_cols = {int(order[p]) for p in pivots}
        remaining = [c for c in remaining if c not in pivot_cols]
    return bases


def _bounded_weight_classes(
    basis: np.ndarray, n: int, size_cap: int, work_cap: int
) -> tuple[np.ndarray, bool, int, str]:
    """``(codeword bits, spans, weight_cap, note)`` via information sets.

    Enumerates messages of weight ``<= t`` on each of ``h`` disjoint
    systematic bases; by the pigeonhole over disjoint supports this yields
    every codeword of weight ``<= h*t + h - 1``, so exactly the weight
    classes inside that bound are COMPLETE and eligible for selection.
    ``t`` grows until the selected classes span, or a cap calls it off.
    """

    rank = basis.shape[0]
    if rank == 0:
        return np.zeros((0, n), np.uint8), True, 0, ""
    bases = _systematic_bases(basis, n)
    h = len(bases)
    packed = [_pack_basis(b, n) for b in bases]
    lanes = packed[0].shape[1]
    found: dict[bytes, np.ndarray] = {}
    work = 0
    depth = 0
    while True:
        depth += 1
        step = h * comb(rank, depth)
        if work + step > work_cap:
            depth -= 1
            note = f"work cap at t = {depth}: classes past weight {h * depth + h - 1} unknown"
            break
        for rows in packed:
            for picks in combinations(range(rank), depth):
                word = rows[picks[0]].copy()
                for index in picks[1:]:
                    word ^= rows[index]
                found.setdefault(word.tobytes(), word)
        work += step
        complete_bound = h * depth + h - 1
        words = np.frombuffer(b"".join(found.keys()), dtype=np.uint64).reshape(-1, lanes)
        weights = _word_weights(words)
        selected: list[int] = []
        spans = False
        weight_cap = 0
        for weight in np.unique(weights):
            if weight == 0 or weight > complete_bound:
                continue
            members = np.flatnonzero(weights == weight)
            if len(selected) + len(members) > size_cap:
                break
            selected.extend(int(index) for index in members)
            weight_cap = int(weight)
            bits = _unpack(words[np.array(selected)], n)
            if row_basis(bits, ncols=n).shape[0] == rank:
                spans = True
                break
        if spans:
            return _unpack(words[np.array(selected)], n), True, weight_cap, f"{h} info sets, t = {depth}"
    # capped out: report the best complete-class selection we have
    if not found:
        return np.zeros((0, n), np.uint8), False, 0, note
    words = np.frombuffer(b"".join(found.keys()), dtype=np.uint64).reshape(-1, lanes)
    weights = _word_weights(words)
    complete_bound = h * depth + h - 1
    selected = []
    weight_cap = 0
    for weight in np.unique(weights):
        if weight == 0 or weight > complete_bound:
            continue
        members = np.flatnonzero(weights == weight)
        if len(selected) + len(members) > size_cap:
            break
        selected.extend(int(index) for index in members)
        weight_cap = int(weight)
    bits = _unpack(words[np.array(selected)], n) if selected else np.zeros((0, n), np.uint8)
    return bits, False, weight_cap, note


@dataclass(frozen=True)
class CodewordAutomorphisms:
    """Certified CSS permutation automorphism group from characteristic sets."""

    qubit_generators: tuple[np.ndarray, ...]
    generators: tuple[AutomorphismGenerator, ...]
    group_order: int
    exact: bool  # complete enumeration + spanning classes + all gens certified
    weight_cap_x: int
    weight_cap_z: int
    codewords_x: int
    codewords_z: int
    notes: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return all(all(g.certificate.values()) for g in self.generators)


def analyze_codeword_automorphisms(
    code: CSSCode,
    *,
    size_cap: int = _SIZE_CAP,
    work_cap: int = _WORK_CAP,
    full_enum_rank: int = _FULL_ENUM_RANK,
) -> CodewordAutomorphisms:
    """Row-space permutation automorphism group of a CSS code.

    Ranks up to ``full_enum_rank`` enumerate the whole space; larger ranks
    take the bounded-weight information-set route (see the module
    docstring), whose completeness bound makes the same exactness argument
    go through.  Exact whenever both characteristic selections consist of
    complete classes AND span AND every BLISS generator passes
    certification; any shortfall degrades to a certified subgroup with
    ``exact = False`` and a note saying why.
    """

    require_igraph()
    n = code.n
    notes: list[str] = []
    rank_x, rank_z = code.c_x.shape[0], code.c_z.shape[0]

    def classes(basis: np.ndarray, side: str):
        if basis.shape[0] <= full_enum_rank:
            bits, _, spans, cap = _characteristic_classes(basis, n, size_cap)
            return bits, spans, cap
        bits, spans, cap, note = _bounded_weight_classes(basis, n, size_cap, work_cap)
        if note:
            notes.append(f"{side}: {note}")
        return bits, spans, cap

    bits_x, spans_x, cap_x = classes(code.c_x, "C_X")
    bits_z, spans_z, cap_z = classes(code.c_z, "C_Z")
    if not spans_x:
        notes.append("C_X classes truncated before spanning: subgroup scope")
    if not spans_z:
        notes.append("C_Z classes truncated before spanning: subgroup scope")
    if rank_x and bits_x.shape[0] == 0 or rank_z and bits_z.shape[0] == 0:
        # No codeword vertices to constrain the coordinates: BLISS would
        # return the symmetric group and certification would grind through
        # meaningless candidates.  Decline honestly instead.
        return CodewordAutomorphisms(
            qubit_generators=(),
            generators=(),
            group_order=1,
            exact=False,
            weight_cap_x=cap_x,
            weight_cap_z=cap_z,
            codewords_x=int(bits_x.shape[0]),
            codewords_z=int(bits_z.shape[0]),
            notes=tuple(notes + ["smallest weight class exceeds size_cap: no verdict"]),
        )

    graph = igraph.Graph(n + bits_x.shape[0] + bits_z.shape[0])
    edges = []
    for index, row in enumerate(bits_x):
        for coordinate in np.flatnonzero(row):
            edges.append((int(coordinate), n + index))
    offset = n + bits_x.shape[0]
    for index, row in enumerate(bits_z):
        for coordinate in np.flatnonzero(row):
            edges.append((int(coordinate), offset + index))
    graph.add_edges(edges)
    # one color per (side, weight) class, disjoint from the coordinate color
    weight_of = lambda row: int(row.sum())  # tiny, local
    palette: dict[tuple[int, int], int] = {}

    def color(side: int, weight: int) -> int:
        return palette.setdefault((side, weight), len(palette) + 1)

    colors = (
        [0] * n
        + [color(1, weight_of(row)) for row in bits_x]
        + [color(2, weight_of(row)) for row in bits_z]
    )

    identity = np.arange(n)
    qubit_perms: list[np.ndarray] = []
    seen: set[bytes] = set()
    for generator in graph.automorphism_group(color=colors):
        perm = np.asarray(generator, dtype=int)[:n]
        key = perm.tobytes()
        if not np.array_equal(perm, identity) and key not in seen:
            seen.add(key)
            qubit_perms.append(perm)

    described = tuple(describe_permutation(code, perm) for perm in qubit_perms)
    certified = tuple(g for g in described if all(g.certificate.values()))
    if len(certified) < len(described):
        notes.append(
            f"{len(described) - len(certified)} graph generators failed row-space "
            "certification and were dropped (expected only in subgroup scope)"
        )
    kept = tuple(g.qubit_permutation for g in certified)
    order = permutation_group_order(list(kept)) if kept else 1
    exact = spans_x and spans_z and len(certified) == len(described)
    return CodewordAutomorphisms(
        qubit_generators=kept,
        generators=certified,
        group_order=order,
        exact=exact,
        weight_cap_x=cap_x,
        weight_cap_z=cap_z,
        codewords_x=int(bits_x.shape[0]),
        codewords_z=int(bits_z.shape[0]),
        notes=tuple(notes),
    )


__all__ = ["CodewordAutomorphisms", "analyze_codeword_automorphisms"]
