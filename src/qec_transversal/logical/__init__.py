"""From a physical code-preserving gate to its logical symplectic image.

Every solver in :mod:`qec_transversal.ansatz` produces physical symplectic
matrices; this subpackage turns them into elements of ``Sp(2k, 2)``, computes
exact orders of the groups they generate, decides membership, and factors a
requested logical target into an explicit word.
"""

from .action import project_to_logical
from .group import (
    GroupOrder,
    generated_group_order,
    logical_group_summary,
    schreier_sims_order,
)

__all__ = [
    "GroupOrder",
    "generated_group_order",
    "logical_group_summary",
    "project_to_logical",
    "schreier_sims_order",
]
