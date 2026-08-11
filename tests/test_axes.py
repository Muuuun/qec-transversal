"""General diagonal solver + axis-frame sweep tests."""

import itertools

import numpy as np
import pytest

pytest.importorskip("stim")

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.axes import axis_frame_group, diagonal_kernel_general
from qec_transversal.gf2 import rank, symplectic_product
from qec_transversal.stabilizer import StabilizerCode, five_qubit_code


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
    sound_failures = 0
    checked = 0
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
            legal = _legal_brute(code, 3)
            got = _span(diagonal_kernel_general(code, level=3), n, 3)
            checked += 1
            if not got <= legal:  # soundness is non-negotiable
                sound_failures += 1
    assert checked >= 8 and sound_failures == 0


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
