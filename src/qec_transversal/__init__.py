"""Automated transversal Clifford analysis for CSS quantum codes."""

from .codes import REGISTRY, NamedCode
from .css import CSSCode, ParameterSpace, TransversalAnalysis, TransversalGenerator
from .gf2 import gf2_inverse, nullspace, rank, row_basis, rref, symplectic_form

__all__ = [
    "REGISTRY",
    "CSSCode",
    "NamedCode",
    "ParameterSpace",
    "TransversalAnalysis",
    "TransversalGenerator",
    "gf2_inverse",
    "nullspace",
    "rank",
    "row_basis",
    "rref",
    "symplectic_form",
]

__version__ = "0.1.0"

