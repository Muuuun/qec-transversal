"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.unitgroup`` split into :mod:`qec_transversal.algebra`:
``finite_algebra`` (the algebra object), ``radical`` (the Cohen-Ivanyos-
Wales radical and its verified peeling), ``wedderburn`` (the constructive
semisimple split) and ``unit_group`` (the driver).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .algebra.finite_algebra import AlgebraF2
from .algebra.radical import _char2_radical, _is_nilpotent_ideal
from .algebra.unit_group import UnitGroupResult, _gl_order, unit_group
from .algebra.wedderburn import _wedderburn
from .utils.polynomials import _berlekamp_factor, _charpoly_f2


__all__ = [
    "AlgebraF2",
    "UnitGroupResult",
    "_berlekamp_factor",
    "_char2_radical",
    "_charpoly_f2",
    "_gl_order",
    "_is_nilpotent_ideal",
    "_wedderburn",
    "unit_group",
]
