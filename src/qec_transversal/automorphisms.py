"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.automorphisms`` is now
:mod:`qec_transversal.ansatz.permutation`; ``permutation_group_order``
moved to :mod:`qec_transversal.utils.permutations`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.permutation import (
    AutomorphismAnalysis,
    AutomorphismGenerator,
    analyze_automorphisms,
    describe_permutation,
)
from .utils.graph import igraph
from .utils.graph import require_igraph as _require_igraph
from .utils.permutations import permutation_group_order


__all__ = [
    "AutomorphismAnalysis",
    "AutomorphismGenerator",
    "_require_igraph",
    "analyze_automorphisms",
    "describe_permutation",
    "igraph",
    "permutation_group_order",
]
