"""Exact and certified analysis of depth-one code-preserving gates.

Start at :mod:`qec_transversal.api`::

    from qec_transversal import REGISTRY, CSSCode, strict_transversal_clifford

    code = CSSCode(*REGISTRY["steane"].build())
    result = strict_transversal_clifford(code)
    print(result.completeness, result.logical_group_order)

The package is organised by concept, not by development history:

``codes``         code objects and the named-code registry
``ansatz``        one module per physical gate class being searched
``algebra``       preservation algebras and certified finite-algebra solvers
``hierarchy``     diagonal gates in the Clifford hierarchy over Z_{2^L}
``logical``       logical action, group orders, recognition, synthesis
``certificates``  witnesses, verifiers, and sign-exact circuit checks
``utils``         GF(2), symplectic, permutation, polynomial primitives
"""

from .ansatz.strict_css import ParameterSpace, TransversalAnalysis, TransversalGenerator
from .api import (
    Completeness,
    GateSearchResult,
    SignCertificate,
    certify_signs,
    css_strict_transversal_clifford,
    diagonal_transversal_gates,
    matching_clifford_group,
    monomial_clifford_group,
    one_block_clifford_group,
    partition_clifford_group,
    permutation_automorphism_group,
    strict_transversal_clifford,
    transversal_clifford_across_blocks,
)
from .codes.css import CSSCode
from .codes.registry import REGISTRY, NamedCode
from .codes.stabilizer import StabilizerCode, five_qubit_code
from .utils.gf2 import gf2_inverse, nullspace, rank, row_basis, rref
from .utils.symplectic import symplectic_form

__all__ = [
    "REGISTRY",
    "CSSCode",
    "Completeness",
    "GateSearchResult",
    "NamedCode",
    "SignCertificate",
    "certify_signs",
    "ParameterSpace",
    "StabilizerCode",
    "TransversalAnalysis",
    "TransversalGenerator",
    "css_strict_transversal_clifford",
    "diagonal_transversal_gates",
    "five_qubit_code",
    "gf2_inverse",
    "matching_clifford_group",
    "monomial_clifford_group",
    "nullspace",
    "one_block_clifford_group",
    "partition_clifford_group",
    "permutation_automorphism_group",
    "rank",
    "row_basis",
    "rref",
    "strict_transversal_clifford",
    "symplectic_form",
    "transversal_clifford_across_blocks",
]

__version__ = "0.2.1"
