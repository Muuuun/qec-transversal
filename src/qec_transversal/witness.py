"""Compatibility alias for the pre-0.2 flat module layout.

``qec_transversal.witness`` is now :mod:`qec_transversal.certificates.witness`.

The names below are re-exported unchanged; new code should import from the
module shown above.  See ``docs/refactor_report.md`` for the full mapping.
"""

from .certificates.witness import (
    export_stabilizer_witness,
    export_strict_witness,
    write_witness,
)


__all__ = [
    "export_stabilizer_witness",
    "export_strict_witness",
    "write_witness",
]
