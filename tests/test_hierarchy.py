"""Level-3 diagonal (transversal-T) certifier tests."""

import itertools

import numpy as np

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.codes import quantum_reed_muller_15, steane_code
from qec_transversal.gf2 import nullspace, reduce_rows, rref
from qec_transversal.hierarchy import analyze_hierarchy


def _span_f2(basis, n):
    vectors = [np.zeros(n, dtype=np.int64)]
    for mask in range(1, 1 << len(basis)):
        v = np.zeros(n, dtype=np.int64)
        for i in range(len(basis)):
            if (mask >> i) & 1:
                v ^= basis[i]
        vectors.append(v)
    return vectors


def _legal_by_definition(code: CSSCode) -> set:
    """All t in Z_8^n with branch phase constant on every C_X-coset."""

    kernel_rows = [row.astype(np.int64) for row in nullspace(code.h_z)]
    normalizer = _span_f2(kernel_rows, code.n)
    reduced, pivots = rref(code.c_x)
    cosets: dict[bytes, list] = {}
    for u in normalizer:
        key = reduce_rows(u.astype(np.uint8)[None, :], reduced, pivots).tobytes()
        cosets.setdefault(key, []).append(u)
    legal = set()
    for t in itertools.product(range(8), repeat=code.n):
        tv = np.array(t, dtype=np.int64)
        if all(len({int(tv @ u) % 8 for u in members}) == 1 for members in cosets.values()):
            legal.add(t)
    return legal


def _span_z8(generators, n):
    out = {tuple([0] * n)}
    frontier = [np.zeros(n, dtype=np.int64)]
    while frontier:
        current = frontier.pop()
        for g in generators:
            candidate = tuple((current + g) % 8)
            if candidate not in out:
                out.add(candidate)
                frontier.append(np.array(candidate, dtype=np.int64))
    return out


def test_kernel_matches_raw_definition_on_random_codes() -> None:
    rng = np.random.default_rng(31337)
    for n in range(2, 6):
        for _ in range(4):
            rows_x = rng.integers(0, n + 1)
            h_x = rng.integers(0, 2, size=(rows_x, n), dtype=np.uint8)
            x_perp = nullspace(h_x)
            z = rng.integers(0, 2, size=(rng.integers(0, n + 1), x_perp.shape[0]), dtype=np.uint8)
            h_z = (z.astype(np.int64) @ x_perp.astype(np.int64) % 2).astype(np.uint8)
            code = CSSCode(h_x, h_z, n=n)
            analysis = analyze_hierarchy(code, "Z")
            assert _span_z8(list(analysis.kernel), n) == _legal_by_definition(code)
            assert analysis.certified


def test_qrm15_certifies_transversal_t() -> None:
    report = analyze_hierarchy(CSSCode(*quantum_reed_muller_15()), "Z").to_dict()
    assert report["has_t_level_gate"] is True
    assert report["max_level"] == 3
    assert report["levels_complete"] and report["certified"]


def test_cube_code_certifies_ccz() -> None:
    code = CSSCode(*REGISTRY["cube-832"].build())
    analysis = analyze_hierarchy(code, "Z")
    report = analysis.to_dict()
    assert report["has_t_level_gate"] is True  # via the CCZ-bar monomial
    # the all-T layer must be in the kernel
    ones = np.ones(code.n, dtype=np.int64)
    from qec_transversal.hierarchy import module_kernel  # noqa: F401

    span_contains = any(
        np.array_equal(g % 8, ones) for g in analysis.kernel
    ) or tuple(ones) in _span_z8(list(analysis.kernel), code.n)
    assert span_contains


def test_steane_tops_out_at_clifford_level() -> None:
    report = analyze_hierarchy(CSSCode(*steane_code()), "Z").to_dict()
    assert report["max_level"] == 2
    assert report["has_t_level_gate"] is False


def test_gross_code_has_pauli_level_only() -> None:
    code = CSSCode(*REGISTRY["gross"].build())
    report = analyze_hierarchy(code, "Z").to_dict()
    assert report["max_level"] <= 1
    assert report["has_t_level_gate"] is False
    assert report["certified"]
