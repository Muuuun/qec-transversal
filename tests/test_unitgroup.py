"""Certified unit-group solver tests: fuzz vs enumeration + integration."""

import numpy as np
import pytest

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.gf2 import row_basis
from qec_transversal.stabilizer import LocalCliffordAnalysis, StabilizerCode
from qec_transversal.unitgroup import AlgebraF2, unit_group


def _block_multiply(n):
    def mul(a, b):
        out = np.zeros(4 * n, dtype=np.uint8)
        for i in range(n):
            A = a[4 * i : 4 * i + 4].reshape(2, 2)
            B = b[4 * i : 4 * i + 4].reshape(2, 2)
            out[4 * i : 4 * i + 4] = (A @ B % 2).reshape(-1)
        return out

    return mul


def _identity_flat(n):
    v = np.zeros(4 * n, dtype=np.uint8)
    v[0::4] = 1
    v[3::4] = 1
    return v


def _random_closed_algebra(rng, n, n_gens):
    mul = _block_multiply(n)
    one = _identity_flat(n)
    gens = [one] + [rng.integers(0, 2, size=4 * n, dtype=np.uint8) for _ in range(n_gens)]
    basis = row_basis(np.asarray(gens, dtype=np.uint8), ncols=4 * n)
    for _ in range(4 * n + 1):  # closure stabilizes within dim steps
        products = [mul(a, b) for a in basis for b in basis]
        new = row_basis(
            np.vstack([basis] + [p[None, :] for p in products]), ncols=4 * n
        )
        if new.shape[0] == basis.shape[0]:
            return new, mul, one
        basis = new
    return None, mul, one


def _brute_unit_count(basis, dim, n):
    count = 0
    for mask in range(1 << dim):
        v = np.zeros(basis.shape[1], dtype=np.uint8)
        for b in range(dim):
            if (mask >> b) & 1:
                v ^= basis[b]
        if all(
            (int(v[4 * i]) & int(v[4 * i + 3])) ^ (int(v[4 * i + 1]) & int(v[4 * i + 2]))
            for i in range(n)
        ):
            count += 1
    return count


def test_unit_group_fuzz_never_wrong() -> None:
    rng = np.random.default_rng(2026)
    exact = failures = 0
    for trial in range(20):
        n = int(rng.integers(1, 4))
        basis, mul, one = _random_closed_algebra(rng, n, int(rng.integers(1, 3)))
        if basis is None or basis.shape[0] > 11:
            continue
        algebra = AlgebraF2(basis, mul, one)
        result = unit_group(algebra, seed=trial)
        if result.status != "exact":
            continue  # honest unknown is acceptable; wrong answers are not
        exact += 1
        want = _brute_unit_count(basis, basis.shape[0], n)
        if result.order != want:
            failures += 1
            continue
        # generation certificate: closure of generators is the full group
        seen = {algebra.one_coords.tobytes()}
        frontier = [algebra.one_coords]
        while frontier:
            current = frontier.pop()
            for g in result.generators:
                nxt = algebra.coords_multiply(current, g)
                if nxt.tobytes() not in seen:
                    seen.add(nxt.tobytes())
                    frontier.append(nxt)
        if len(seen) != want:
            failures += 1
    assert failures == 0 and exact >= 10


def _stacked(css: CSSCode) -> StabilizerCode:
    return StabilizerCode(
        np.vstack(
            [
                np.hstack([css.c_x, np.zeros_like(css.c_x)]),
                np.hstack([np.zeros_like(css.c_z), css.c_z]),
            ]
        )
    )


@pytest.mark.parametrize(
    "name", ["steane", "c4-22", "iceberg-8", "cube-832", "qrm15", "toric-4"]
)
def test_structured_route_matches_enumeration(name) -> None:
    code = _stacked(CSSCode(*REGISTRY[name].build()))
    structured = LocalCliffordAnalysis(code, dim_cap=0).to_dict()
    enumerated = LocalCliffordAnalysis(code).to_dict()
    assert structured["status"] == "exact"
    assert structured["physical_group_order"] == enumerated["physical_group_order"]
    assert (
        structured["logical_group"]["order"] == enumerated["logical_group"]["order"]
    )
