"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.dualities`` is now :mod:`qec_transversal.ansatz.dualities`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .ansatz.dualities import *  # noqa: F403
from .ansatz.dualities import (
    candidates_for,
    gb_inversion,
    hgp_transpose,
    kasai_block_negation,
    two_block_inversion,
    two_block_reflection,
    two_block_swap_xy,
)


__all__ = [
    "candidates_for",
    "gb_inversion",
    "hgp_transpose",
    "kasai_block_negation",
    "two_block_inversion",
    "two_block_reflection",
    "two_block_swap_xy",
]
