"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.stabilizer`` split into
:mod:`qec_transversal.codes.stabilizer` (the code object),
:mod:`qec_transversal.algebra.preservation` (the preservation algebra),
:mod:`qec_transversal.ansatz.strict` and
:mod:`qec_transversal.ansatz.partition` (the solvers).

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .algebra.preservation import (
    local_clifford_algebra,
    partition_algebra,
)
from .ansatz.partition import (
    PartitionCliffordAnalysis,
    analyze_partition_clifford,
    partition_units_via_structure,
)
from .ansatz.strict import (
    LocalCliffordAnalysis,
    LocalCliffordGenerator,
    analyze_local_clifford,
)
from .codes.stabilizer import StabilizerCode, five_qubit_code
from .utils.symplectic import symplectic_gram_schmidt


__all__ = [
    "LocalCliffordAnalysis",
    "LocalCliffordGenerator",
    "PartitionCliffordAnalysis",
    "StabilizerCode",
    "analyze_local_clifford",
    "analyze_partition_clifford",
    "five_qubit_code",
    "local_clifford_algebra",
    "partition_algebra",
    "partition_units_via_structure",
    "symplectic_gram_schmidt",
]
