"""Exact computational algebra over ``F_2``.

The pipeline implemented here is the mathematical core of the package:

    preservation constraints  ->  finite algebra ``A``
                              ->  Jacobson radical ``J(A)``
                              ->  semisimple quotient ``A / J(A)``
                              ->  Wedderburn split ``prod_i M_{d_i}(F_{q_i})``
                              ->  unit group ``A^x`` with its exact order.

None of the individual algorithms is new mathematics; the contribution of
this package is that every stage carries a machine-checked certificate and
that a failed verification degrades to ``unknown`` instead of to a wrong
answer.
"""

from .finite_algebra import AlgebraF2
from .preservation import local_clifford_algebra, partition_algebra
from .unit_group import UnitGroupResult, unit_group

__all__ = [
    "AlgebraF2",
    "UnitGroupResult",
    "local_clifford_algebra",
    "partition_algebra",
    "unit_group",
]
