"""Proof-carrying computation: witnesses and independent verifiers.

Every headline result of this package can be exported as a *witness* -- a
self-contained document that an independent checker re-verifies without
importing any of this code.  Two flavours of witness are produced here (CSS
strict and general stabilizer), together with the Smith-form completeness
certificate for the ``Z_{2^L}`` hierarchy kernels and the sign-exact
circuit-level verifier that removes the "modulo Pauli" fine print.

The standalone checkers live in ``tools/`` and import nothing from this
package on purpose.
"""

from .hierarchy import check_kernel_certificate, smith_kernel_certificate
from .witness import export_stabilizer_witness, export_strict_witness, write_witness

__all__ = [
    "check_kernel_certificate",
    "export_stabilizer_witness",
    "export_strict_witness",
    "smith_kernel_certificate",
    "write_witness",
]
