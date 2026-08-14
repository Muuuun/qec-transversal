"""Bit-packed backend equivalence + Smith kernel-certificate tests."""

import numpy as np

from qec_transversal import REGISTRY, CSSCode, gf2
from qec_transversal.gf2 import _rref_packed, rref
from qec_transversal.hierarchy import (
    analyze_hierarchy,
    check_kernel_certificate,
    module_kernel,
    smith_kernel_certificate,
)


def test_packed_rref_equivalent_to_dense() -> None:
    rng = np.random.default_rng(64)
    for _ in range(40):
        rows = int(rng.integers(1, 40))
        cols = int(rng.integers(1, 300))
        matrix = (rng.random((rows, cols)) < rng.choice([0.05, 0.5])).astype(np.uint8)
        packed_result, packed_pivots = _rref_packed(matrix)
        saved = gf2._PACKED_MIN_COLS
        gf2._PACKED_MIN_COLS = 10**9
        try:
            dense_result, dense_pivots = rref(matrix)
        finally:
            gf2._PACKED_MIN_COLS = saved
        assert np.array_equal(packed_result, dense_result)
        assert packed_pivots == dense_pivots


def _span(generators, n, level):
    modulus = 1 << level
    out = {tuple([0] * n)}
    frontier = [np.zeros(n, dtype=np.int64)]
    while frontier:
        current = frontier.pop()
        for g in generators:
            candidate = tuple((current + g) % modulus)
            if candidate not in out:
                out.add(candidate)
                frontier.append(np.array(candidate, dtype=np.int64))
    return out


def test_smith_certificate_matches_kernel_and_checks() -> None:
    rng = np.random.default_rng(88)
    for trial in range(25):
        n = int(rng.integers(1, 5))
        level = int(rng.integers(2, 5))
        constraints = [
            (rng.integers(0, 1 << level, size=n).astype(np.int64), int(rng.choice([1, 2, 4])))
            for _ in range(int(rng.integers(0, 5)))
        ]
        certificate = smith_kernel_certificate(constraints, n, exponent=level)
        assert check_kernel_certificate(constraints, n, certificate, exponent=level)
        assert _span(list(certificate["kernel"]), n, level) == _span(
            list(module_kernel(constraints, n, exponent=level)), n, level
        )


def test_smith_certificate_rejects_mutations() -> None:
    import copy

    n, level = 4, 3
    constraints = [
        (np.array([2, 1, 0, 4], dtype=np.int64), 1),
        (np.array([0, 2, 2, 0], dtype=np.int64), 2),
    ]
    certificate = smith_kernel_certificate(constraints, n, exponent=level)
    assert check_kernel_certificate(constraints, n, certificate, exponent=level)

    m = copy.deepcopy(certificate)
    m["exponents"] = [min(3, e + 1) for e in m["exponents"]]
    assert not check_kernel_certificate(constraints, n, m, exponent=level)
    m = copy.deepcopy(certificate)
    m["V"] = (m["V"] + 1) % 8
    assert not check_kernel_certificate(constraints, n, m, exponent=level)
    m = copy.deepcopy(certificate)
    m["kernel"] = (
        np.vstack([m["kernel"], np.array([[1, 1, 1, 1]])])
        if m["kernel"].size
        else np.array([[1, 1, 1, 1]])
    )
    assert not check_kernel_certificate(constraints, n, m, exponent=level)


def test_hierarchy_analysis_exposes_verifiable_certificate() -> None:
    code = CSSCode(*REGISTRY["qrm15"].build())
    analysis = analyze_hierarchy(code, "Z", level=3)
    certificate = analysis.kernel_certificate()
    assert check_kernel_certificate(
        analysis._constraints, code.n, certificate, exponent=3
    )
    assert _span(list(certificate["kernel"]), code.n, 3) == _span(
        list(analysis.kernel), code.n, 3
    )
