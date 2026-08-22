"""Automorphism (SWAP-class) gates and duality existence via Tanner graphs.

The colored Tanner graph of a CSS code has qubit vertices (color 0),
X-check vertices (color 1), and Z-check vertices (color 2).  Its
automorphism group — computed exactly by BLISS through ``python-igraph`` —
gives qubit permutations that preserve both check sets; each is a
SWAP-transversal logical gate (a depth-one permutation circuit).  Swapping
the two check colors instead and asking for a graph *isomorphism* decides
whether any permutation ZX-duality exists at all.

Scope: the group is exact for the Tanner graph of the check matrices as
given (dependent rows included, which preserves natural symmetries such as
lattice translations).  Every generator is additionally certified directly
against the row spaces, so no reported gate depends on the graph encoding.
Code automorphisms that do not fix the given check sets row-by-row are
outside this notion, as everywhere in the Tanner-graph literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..codes.css import CSSCode
from ..logical.action import project_to_logical
from ..logical.group import logical_group_summary
from ..utils.gf2 import gf2_matmul, row_basis, rowspace_residues
from ..utils.graph import igraph, require_igraph
from ..utils.permutations import permutation_group_order

#: Backwards-compatible private alias of :func:`..utils.graph.require_igraph`.
_require_igraph = require_igraph

def _tanner_graph(code: CSSCode, *, swap_colors: bool = False):
    n = code.n
    rows_x, rows_z = code.h_x.shape[0], code.h_z.shape[0]
    graph = igraph.Graph(n + rows_x + rows_z)
    edges = []
    for r, row in enumerate(code.h_x):
        for q in np.flatnonzero(row):
            edges.append((int(q), n + r))
    for r, row in enumerate(code.h_z):
        for q in np.flatnonzero(row):
            edges.append((int(q), n + rows_x + r))
    graph.add_edges(edges)
    x_color, z_color = (2, 1) if swap_colors else (1, 2)
    colors = [0] * n + [x_color] * rows_x + [z_color] * rows_z
    return graph, colors


@dataclass(frozen=True)
class AutomorphismGenerator:
    qubit_permutation: np.ndarray
    logical_symplectic: np.ndarray
    certificate: dict[str, bool]
    #: The dense ``2n x 2n`` physical symplectic matrix of the permutation.
    matrix: np.ndarray | None = None

    @property
    def is_logical_identity(self) -> bool:
        return np.array_equal(
            self.logical_symplectic,
            np.eye(self.logical_symplectic.shape[0], dtype=np.uint8),
        )


def describe_permutation(code: CSSCode, perm: np.ndarray) -> AutomorphismGenerator:
    """Certified SWAP-class gate record of a qubit permutation.

    Independent of how ``perm`` was found (Tanner-graph automorphism,
    characteristic-codeword automorphism, structural discovery,
    caller-supplied): the certificate re-verifies from scratch that both
    rowspaces are preserved and that the logical residue lies in the
    stabilizer, so an unsound candidate fails here rather than poisoning a
    downstream group order.
    """

    n = code.n
    matrix = np.eye(n, dtype=np.uint8)[perm]
    preserves_x = not rowspace_residues(gf2_matmul(code.c_x, matrix), code.c_x).any()
    preserves_z = not rowspace_residues(gf2_matmul(code.c_z, matrix), code.c_z).any()

    images = np.hstack(
        [gf2_matmul(code.logical[:, :n], matrix), gf2_matmul(code.logical[:, n:], matrix)]
    )
    logical, residue_ok = project_to_logical(
        images,
        code.logical,
        qubits=n,
        logical_qubits=code.k,
        stabilizer_rref=code._stabilizer_rref,
    )
    certificate = {
        "preserves_C_X": bool(preserves_x),
        "preserves_C_Z": bool(preserves_z),
        "logical_residue_in_stabilizer": bool(residue_ok),
    }
    permutation_symplectic = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    permutation_symplectic[:n, :n] = matrix
    permutation_symplectic[n:, n:] = matrix
    return AutomorphismGenerator(
        qubit_permutation=np.asarray(perm, dtype=int).copy(),
        logical_symplectic=logical,
        certificate=certificate,
        matrix=permutation_symplectic,
    )

class AutomorphismAnalysis:
    """Exact Tanner-graph automorphism group and its SWAP-class gates."""

    def __init__(self, code: CSSCode):
        require_igraph()
        self.code = code
        graph, colors = _tanner_graph(code)
        raw_generators = graph.automorphism_group(color=colors)
        n = code.n
        qubit_perms: list[np.ndarray] = []
        seen: set[bytes] = set()
        for generator in raw_generators:
            perm = np.asarray(generator, dtype=int)[:n]
            if np.array_equal(perm, np.arange(n)):
                continue
            key = perm.tobytes()
            if key not in seen:
                seen.add(key)
                qubit_perms.append(perm)
        self.qubit_generators = qubit_perms
        self.group_order = permutation_group_order(qubit_perms) if qubit_perms else 1
        self.generators = tuple(self._describe(perm) for perm in qubit_perms)

        # Duality existence: G is isomorphic to its color-swapped copy iff
        # Aut(G disjoint-union G_swapped) moves a vertex across components
        # (components are connected, so any crossing automorphism carries
        # the whole component).  BLISS computes that group exactly; VF2's
        # nonexistence proofs can blow up exponentially, so it is avoided.
        swapped, swapped_colors = _tanner_graph(code, swap_colors=True)
        total = graph.vcount()
        union = graph.disjoint_union(swapped)
        union_colors = list(colors) + list(swapped_colors)
        union_generators = [
            np.asarray(g, dtype=int) for g in union.automorphism_group(color=union_colors)
        ]
        # orbit of vertex 0 with witness permutations (orbit size <= 2N)
        witness: np.ndarray | None = None
        orbit: dict[int, np.ndarray] = {0: np.arange(2 * total)}
        queue = [0]
        while queue and witness is None:
            vertex = queue.pop()
            carrier = orbit[vertex]
            for generator in union_generators:
                image = int(generator[vertex])
                if image not in orbit:
                    orbit[image] = generator[carrier]
                    if image >= total:
                        witness = orbit[image]
                        break
                    queue.append(image)
        self.tanner_connected = bool(graph.is_connected())
        # Tri-state: witness None proves nonexistence (sound even when the
        # graph is disconnected); a witness on a disconnected graph only
        # maps one component, so the question is then left undecided.
        self.duality_decided = bool(witness is None or self.tanner_connected)
        self.duality: np.ndarray | None = None
        self.duality_certified = False
        if witness is not None and self.tanner_connected:
            candidate = (witness[:n] - total).astype(int)
            for tau in (candidate, np.argsort(candidate)):
                image = row_basis(
                    gf2_matmul(code.c_x, np.eye(n, dtype=np.uint8)[tau]), ncols=n
                )
                if image.shape[0] == code.c_z.shape[0] and not rowspace_residues(
                    image, code.c_z
                ).any():
                    self.duality_certified = True
                    self.duality = tau
                    break

    def _describe(self, perm: np.ndarray) -> AutomorphismGenerator:
        return describe_permutation(self.code, perm)

    def involutive_duality(self, *, search_limit: int = 20_000) -> np.ndarray | None:
        """An involutive certified duality: the found one if it squares to
        the identity, else a breadth-first search of its automorphism coset."""

        if self.duality is None:
            return None
        n = self.code.n
        identity = np.arange(n)
        if np.array_equal(self.duality[self.duality], identity):
            return self.duality
        seen = {self.duality.tobytes()}
        queue = [self.duality]
        tries = 0
        while queue and tries < search_limit:
            current = queue.pop()
            for generator in self.qubit_generators:
                candidate = current[generator]
                key = candidate.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                tries += 1
                if np.array_equal(candidate[candidate], identity):
                    return candidate
                queue.append(candidate)
        return None

    @property
    def certified(self) -> bool:
        return all(all(g.certificate.values()) for g in self.generators)

    def to_dict(self) -> dict[str, Any]:
        nontrivial = [
            g.logical_symplectic for g in self.generators if not g.is_logical_identity
        ]
        return {
            "generator_count": len(self.generators),
            "qubit_group_order": int(self.group_order),
            "nontrivial_logical_generators": len(nontrivial),
            "logical_group": logical_group_summary(nontrivial, self.code.k),
            "duality_exists": bool(self.duality_certified) if self.duality_decided else None,
            "duality_decided": bool(self.duality_decided),
            "tanner_connected": bool(self.tanner_connected),
            "certified": self.certified,
        }


def analyze_automorphisms(code: CSSCode) -> AutomorphismAnalysis:
    """Exact Tanner-graph automorphism analysis (needs python-igraph)."""

    return AutomorphismAnalysis(code)
