"""The optional graph-isomorphism backend.

Three engines in this package hand a vertex-coloured incidence graph to BLISS
through ``python-igraph``: the Tanner-graph automorphism solver, the
characteristic-codeword row-space solver, and the monomial GF(4) solver.  They
share the import guard here so an optional dependency is reported once, in one
voice.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised through the import guard test
    import igraph
except ImportError:  # pragma: no cover
    igraph = None


def require_igraph() -> None:
    """Raise a helpful ``ImportError`` when the BLISS backend is missing."""

    if igraph is None:
        raise ImportError(
            "graph-automorphism analysis needs python-igraph "
            "(pip install 'qec-transversal[automorphism]')"
        )


#: Backwards-compatible private alias (pre-refactor name).
_require_igraph = require_igraph
