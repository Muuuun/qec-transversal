"""Quantum code objects and the named-code registry.

Representation only: a code knows its checks, its rank-derived ``[[n, k]]``
parameters, and a symplectically paired logical basis.  What gates it admits
is the business of :mod:`qec_transversal.ansatz`.
"""

from .css import CSSCode
from .families import *  # noqa: F403
from .families import __all__ as _FAMILY_ALL
from .registry import REGISTRY, NamedCode
from .stabilizer import StabilizerCode, five_qubit_code

__all__ = list(_FAMILY_ALL) + [
    "CSSCode",
    "NamedCode",
    "REGISTRY",
    "StabilizerCode",
    "five_qubit_code",
]
