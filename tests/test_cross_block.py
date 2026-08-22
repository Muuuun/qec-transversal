r"""Gates *across* code blocks -- the classical notion of "transversal".

An ``l``-block transversal gate is a depth-one layer whose cells each hold one
qubit from every block, acting on the joint code ``S^{\otimes l}``.  So it is
the prescribed-partition problem again, with no new mathematics: the point of
these tests is that the reduction is right and that the answers are the ones
the physics says they should be.

The headline checks are two *certified* facts of opposite sign:

* two Steane blocks realise the **entire** ``Sp(4,2)`` logical Clifford group
  on their two logical qubits, from depth-one two-block gates alone;
* two ``[[5,1,3]]`` blocks reach order 18 of that same 720 and no more --
  a certified negative, not an exhausted search.
"""

import numpy as np
import pytest

from qec_transversal import (
    REGISTRY,
    Completeness,
    CSSCode,
    five_qubit_code,
    partition_clifford_group,
    strict_transversal_clifford,
    transversal_clifford_across_blocks,
)
from qec_transversal.codes.stabilizer import (
    corresponding_qubit_cells,
    tensor_power,
    tensor_product,
)
from qec_transversal.utils.gf2 import gf2_matmul, rowspace_residues
from qec_transversal.utils.symplectic import symplectic_group_order, symplectic_product


def _steane():
    return CSSCode(*REGISTRY["steane"].build()).to_stabilizer_code()


# ---------------------------------------------------------------------------
# the joint-code construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_tensor_power_adds_parameters(blocks: int) -> None:
    single = _steane()
    joint = tensor_power(single, blocks)
    assert joint.n == blocks * single.n
    assert joint.k == blocks * single.k
    assert joint.rank == blocks * single.rank
    # still a genuine stabilizer code: all rows commute
    assert not symplectic_product(joint.h, joint.h, qubits=joint.n).any()


def test_tensor_product_of_different_codes() -> None:
    joint = tensor_product([_steane(), five_qubit_code()])
    assert (joint.n, joint.k) == (12, 2)
    assert not symplectic_product(joint.h, joint.h, qubits=joint.n).any()


def test_each_block_stabilizer_embeds_in_the_joint_code() -> None:
    single = _steane()
    joint = tensor_power(single, 2)
    n = single.n
    for block in range(2):
        for row in single.h:
            padded = np.zeros(2 * joint.n, dtype=np.uint8)
            padded[block * n : (block + 1) * n] = row[:n]
            padded[joint.n + block * n : joint.n + (block + 1) * n] = row[n:]
            assert not rowspace_residues(padded[None, :], joint.h).any()


def test_corresponding_qubit_cells_partition_the_register() -> None:
    cells = corresponding_qubit_cells(12, 3)
    assert cells == [(0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11)]
    assert sorted(q for cell in cells for q in cell) == list(range(12))
    with pytest.raises(ValueError, match="equal blocks"):
        corresponding_qubit_cells(7, 2)


# ---------------------------------------------------------------------------
# the reduction is exactly the partition solver
# ---------------------------------------------------------------------------


def test_one_block_reduces_to_the_strict_transversal_group() -> None:
    single = _steane()
    across = transversal_clifford_across_blocks(single, blocks=1)
    strict = strict_transversal_clifford(single, method="general")
    assert across.completeness is Completeness.COMPLETE
    assert across.group_order == strict.group_order == 6


def test_across_blocks_agrees_with_the_partition_solver_directly() -> None:
    single = _steane()
    joint = tensor_power(single, 2)
    direct = partition_clifford_group(
        joint, corresponding_qubit_cells(joint.n, 2), method="enumeration"
    )
    wrapped = transversal_clifford_across_blocks(single, blocks=2, method="enumeration")
    assert wrapped.group_order == direct.group_order
    assert wrapped.logical_group_order == direct.logical_group_order
    assert "2 code blocks" in wrapped.ansatz


# ---------------------------------------------------------------------------
# the physics
# ---------------------------------------------------------------------------


def test_two_steane_blocks_realise_the_full_two_qubit_logical_clifford() -> None:
    """Certified positive: depth-one two-block gates suffice, completely."""

    result = transversal_clifford_across_blocks(_steane(), blocks=2)
    assert result.completeness is Completeness.COMPLETE
    assert result.metadata["joint_k"] == 2
    assert result.logical_group_order == symplectic_group_order(2) == 720
    assert result.logical_group_order_is_exact


def test_two_perfect_code_blocks_fall_short_and_that_is_certified() -> None:
    """Certified negative: 18 of 720, and the 18 is exact, not a floor."""

    result = transversal_clifford_across_blocks(five_qubit_code(), blocks=2)
    assert result.completeness is Completeness.COMPLETE
    assert result.metadata["joint_k"] == 2
    assert result.logical_group_order == 18
    assert result.logical_group_order_is_exact
    assert result.logical_group_order < symplectic_group_order(2)


def test_every_cross_block_generator_preserves_the_joint_code() -> None:
    """No generator is taken on trust, however it was produced."""

    single = _steane()
    joint = tensor_power(single, 2)
    result = transversal_clifford_across_blocks(single, blocks=2, method="enumeration")
    assert result.generators
    checked = 0
    for record in result.generators:
        matrix = getattr(record, "matrix", None)
        if matrix is None:
            continue
        assert not rowspace_residues(gf2_matmul(joint.h, matrix), joint.h).any()
        checked += 1
    assert checked == len(result.generators)


def test_blocks_must_be_positive() -> None:
    with pytest.raises(ValueError, match="blocks"):
        transversal_clifford_across_blocks(_steane(), blocks=0)
