"""General diagonal solver + axis-frame sweep tests."""

import itertools

import numpy as np
import pytest

pytest.importorskip("stim")

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.axes import (
    _css_split,
    _z_subspace_with_signs,
    axis_frame_group,
    diagonal_kernel_general,
    diagonal_kernel_general_exact,
)
from qec_transversal.gf2 import rank, symplectic_product
from qec_transversal.stabilizer import StabilizerCode, five_qubit_code

#: Level-3 witness where the sound engine is incomplete: the residual
#: phase of e.g. t = (1, 0, 7) vanishes on the support coset (u_0 = u_2)
#: only, so the sound kernel spans 8 of the 16 legal diagonal gates.
WITNESS_ROWS = np.array([[1, 1, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1]], dtype=np.uint8)


def _projector(code: StabilizerCode) -> np.ndarray:
    n = code.n
    dim = 1 << n
    projector = np.eye(dim, dtype=complex)
    for row in code.h:
        a, b = row[:n], row[n:]
        S = np.zeros((dim, dim), dtype=complex)
        for u in range(dim):
            bits = np.array([(u >> i) & 1 for i in range(n)], dtype=np.int64)
            v = u ^ int(sum(int(a[i]) << i for i in range(n)))
            S[v, u] = (-1) ** int((b @ bits) % 2) * (1j) ** int((a @ b) % 2)
        projector = projector @ (np.eye(dim) + S) / 2
    return projector


def _legal_brute(code: StabilizerCode, level: int) -> set:
    n, dim, mod = code.n, 1 << code.n, 1 << level
    omega = np.exp(2j * np.pi / mod)
    projector = _projector(code)
    legal = set()
    for t in itertools.product(range(mod), repeat=n):
        phases = np.array(
            [omega ** sum(t[i] * ((u >> i) & 1) for i in range(n)) for u in range(dim)]
        )
        U = np.diag(phases)
        if np.allclose(U @ projector @ U.conj().T, projector, atol=1e-9):
            legal.add(t)
    return legal


def _span(kernel: np.ndarray, n: int, level: int) -> set:
    mod = 1 << level
    out = {tuple([0] * n)}
    frontier = [np.zeros(n, dtype=np.int64)]
    while frontier:
        current = frontier.pop()
        for g in kernel:
            candidate = tuple((current + g) % mod)
            if candidate not in out:
                out.add(candidate)
                frontier.append(np.array(candidate, dtype=np.int64))
    return out


def test_general_solver_is_sound_and_usually_complete() -> None:
    rng = np.random.default_rng(31)
    checked = 0
    complete_count = 0
    nontrivial_kernels = 0
    for n in (2, 3):
        for _ in range(6):
            rows = np.zeros((0, 2 * n), dtype=np.uint8)
            for _ in range(200):
                if rows.shape[0] >= rng.integers(1, n + 1):
                    break
                c = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
                if not c.any():
                    continue
                if rows.shape[0] and symplectic_product(rows, c[None, :], qubits=n).any():
                    continue
                stacked = np.vstack([rows, c])
                if rank(stacked) > rows.shape[0]:
                    rows = stacked
            if rows.shape[0] == 0:
                continue
            code = StabilizerCode(rows)
            checked += 1
            legal = _legal_brute(code, 3)
            sound_span = _span(diagonal_kernel_general(code, level=3), n, 3)
            assert sound_span <= legal  # soundness is non-negotiable
            z_dim = _z_subspace_with_signs(code)[0].shape[0]
            if z_dim == 0 or _css_split(code) is not None:
                # proved completeness rule for the sound engine
                assert sound_span == legal
            kernel, certified = diagonal_kernel_general_exact(code, level=3)
            got = _span(kernel, n, 3)
            assert got <= legal  # the exact engine must stay sound too
            complete = got == legal
            if certified:  # a claimed certificate must be honest
                assert complete
            complete_count += complete
            nontrivial_kernels += len(got) > 1
    assert checked == 12
    # the capped exact engine covers every seed-31 code (spec floor: >= 11;
    # the sound engine alone achieved 11/12, missing only the witness)
    assert complete_count == checked
    # guards against a vacuous pass: a stubbed solver returning an empty
    # kernel would span {0} everywhere and fail here
    assert nontrivial_kernels >= 1


def test_witness_code_exact_engine_spans_all_sixteen_legal_gates() -> None:
    # regression for the sound engine's known incompleteness: the capped
    # exact support-coset enumeration must recover the full legal group.
    witness = StabilizerCode(WITNESS_ROWS)
    legal = _legal_brute(witness, 3)
    assert len(legal) == 16
    sound_span = _span(diagonal_kernel_general(witness, level=3), 3, 3)
    assert sound_span < legal and len(sound_span) == 8
    kernel, certified = diagonal_kernel_general_exact(witness, level=3)
    assert certified
    got = _span(kernel, 3, 3)
    assert got == legal
    assert (1, 0, 7) in got  # the gate the sound engine misses


def test_general_route_certifies_complete_on_noncss_zdim0_code() -> None:
    # X (x) Z: a single mixed row, so no frame trickery is needed — the Z
    # frame itself is non-CSS with z_dim == 0 and takes the general route.
    code = StabilizerCode(np.array([[1, 0, 0, 1]], dtype=np.uint8))
    assert _css_split(code) is None
    assert _z_subspace_with_signs(code)[0].shape[0] == 0
    kernel, certified = diagonal_kernel_general_exact(code, level=3)
    assert certified is True
    assert _span(kernel, 2, 3) == _legal_brute(code, 3)
    result = axis_frame_group(code, level=3)
    by_axes = {axes: complete for axes, _, complete in result.nontrivial_frames}
    assert by_axes[("Z", "Z")] is True  # reported complete via the general route


def test_y_stabilized_qubit_found_via_its_frame() -> None:
    code = StabilizerCode(np.array([[1, 1]], dtype=np.uint8))
    result = axis_frame_group(code, level=3).to_dict()
    assert result["nontrivial_frames"] == [
        {"axes": "Y", "kernel_generators": 1, "complete": True}
    ]


def _stacked(css: CSSCode) -> StabilizerCode:
    return StabilizerCode(
        np.vstack(
            [
                np.hstack([css.c_x, np.zeros_like(css.c_x)]),
                np.hstack([np.zeros_like(css.c_z), css.c_z]),
            ]
        )
    )


def test_steane_has_exactly_the_three_uniform_diagonal_frames() -> None:
    result = axis_frame_group(_stacked(CSSCode(*REGISTRY["steane"].build())), level=3)
    assert result.exhaustive and result.frames_tested == 3**7
    axes = sorted("".join(a) for a, _, _ in result.nontrivial_frames)
    assert axes == ["X" * 7, "Y" * 7, "Z" * 7]
    assert all(complete for _, _, complete in result.nontrivial_frames)


def test_five_qubit_code_has_no_diagonal_frame_gates() -> None:
    # exhaustive over all 243 frames: the [[5,1,3]] transversal group is
    # purely the Clifford C3 - no diagonal-frame (hence no T-type) gate in
    # any local Clifford frame, matching Zeng-Cross-Chuang.
    result = axis_frame_group(five_qubit_code(), level=3)
    assert result.exhaustive
    assert result.nontrivial_frames == []


def test_five_qubit_code_has_no_diagonal_frame_gates_at_level_four() -> None:
    # companion to the level-3 sweep: the exact engine covers every frame
    # ([[5,1,3]] has z_dim == 0, so rank(A) + dim(T) <= 9), hence this is
    # an exhaustive *certified* no-go at level 4 as well.
    result = axis_frame_group(five_qubit_code(), level=4)
    assert result.exhaustive
    assert result.nontrivial_frames == []
