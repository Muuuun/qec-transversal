r"""Extraction of the logical symplectic action of a physical gate.

A code-preserving physical symplectic matrix ``M`` acts on the normaliser
``S^\perp`` and fixes ``S`` setwise, so it descends to the quotient
``S^\perp / S \cong F_2^{2k}``.  With a symplectically paired logical basis
``(\bar X_1..\bar X_k, \bar Z_1..\bar Z_k)`` the induced matrix is read off
by symplectic pairing, and the residue of the image against that basis must
lie in ``S`` -- which is exactly the certificate that the descent is
well defined.

Before the 2026 refactor this eleven-line computation was copied into seven
solver modules.  It lives here once; every backend calls it, so a single
convention change cannot desynchronise them.
"""

from __future__ import annotations

import numpy as np

from ..utils.gf2 import BinaryMatrix, gf2_matmul, reduce_rows
from ..utils.symplectic import symplectic_product


def project_to_logical(
    images: BinaryMatrix,
    logical_basis: BinaryMatrix,
    *,
    qubits: int,
    logical_qubits: int,
    stabilizer_rref: tuple,
) -> tuple[BinaryMatrix, bool]:
    """Return ``(logical action, residue_in_stabilizer)``.

    ``images`` are the images of the ``2k`` paired logical rows under the
    physical gate, ``logical_basis`` the paired logical rows themselves
    (``\bar X`` block then ``\bar Z`` block), and ``stabilizer_rref`` the
    precomputed ``rref`` of the stabilizer rows.  The boolean is ``False``
    exactly when the gate does not descend, i.e. the claimed logical action is
    not the whole story -- callers must treat that as a failed certificate.
    """

    if logical_qubits == 0:
        return np.zeros((0, 0), dtype=np.uint8), True
    x_coefficients = symplectic_product(images, logical_basis[logical_qubits:], qubits=qubits)
    z_coefficients = symplectic_product(images, logical_basis[:logical_qubits], qubits=qubits)
    action = np.hstack([x_coefficients, z_coefficients]).astype(np.uint8)
    residue = (images ^ gf2_matmul(action, logical_basis)) & 1
    return action, not reduce_rows(residue, *stabilizer_rref).any()
