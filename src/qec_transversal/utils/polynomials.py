"""Univariate polynomial arithmetic over GF(2), in two representations.

Two independent representations are carried, each matched to its caller:

``integer bitmask``
    bit ``i`` is the coefficient of ``x^i``.  Compact and fast for the
    scalar-heavy factoring used by the classical-group recognition route
    (:mod:`..logical.recognition`).

``coefficient array``
    a low-degree-first ``uint8`` vector.  Used by the finite-algebra solver
    (:mod:`..algebra`), where polynomials are produced from and consumed by
    NumPy matrices.

The duplication is deliberate and documented rather than hidden: unifying the
two would mean rewriting two independently validated factoring routines.  See
``docs/refactor_report.md``.
"""

from __future__ import annotations

import numpy as np

from .gf2 import gf2_matmul as _gf2_matmul
from .gf2 import nullspace, rank
from .gf2 import solve_gf2 as _solve_gf2


def _solve_coords(basis: np.ndarray, vector: np.ndarray) -> np.ndarray | None:
    """Coordinates of ``vector`` over ``basis`` rows, or ``None``."""

    return _solve_gf2(np.asarray(basis, dtype=np.uint8).T, vector)


# ---------------------------------------------------------------------------
# GF(2)[x] as integer bitmasks (bit i = coefficient of x^i)
# ---------------------------------------------------------------------------

def _pdeg(p: int) -> int:
    return p.bit_length() - 1


def _pmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def _pmod(a: int, m: int) -> int:
    dm = _pdeg(m)
    while _pdeg(a) >= dm:
        a ^= m << (_pdeg(a) - dm)
    return a


def _pdivmod(a: int, m: int) -> tuple[int, int]:
    q = 0
    dm = _pdeg(m)
    while _pdeg(a) >= dm:
        shift = _pdeg(a) - dm
        q ^= 1 << shift
        a ^= m << shift
    return q, a


def _pgcd(a: int, b: int) -> int:
    while b:
        a, b = b, _pmod(a, b)
    return a


def _ppowmod(base: int, exponent: int, mod: int) -> int:
    result = _pmod(1, mod) if _pdeg(mod) == 0 else 1
    base = _pmod(base, mod)
    while exponent:
        if exponent & 1:
            result = _pmod(_pmul(result, base), mod)
        base = _pmod(_pmul(base, base), mod)
        exponent >>= 1
    return result


def _pderiv(a: int) -> int:
    # d/dx sum a_i x^i = sum over odd i of a_i x^(i-1) over GF(2)
    result = 0
    for i in range(1, a.bit_length(), 2):
        if (a >> i) & 1:
            result |= 1 << (i - 1)
    return result


def _psqrt(a: int) -> int:
    """Square root of a perfect square over GF(2) (coefficients at even powers)."""

    result = 0
    i = 0
    while (a >> (2 * i)) != 0:
        if (a >> (2 * i)) & 1:
            result |= 1 << i
        i += 1
    return result


def _primes_up_to(limit: int) -> list[int]:
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return [int(p) for p in np.flatnonzero(sieve)]


def _pirreducible(p: int) -> bool:
    """Rabin's deterministic irreducibility test over GF(2)."""

    d = _pdeg(p)
    if d <= 0:
        return False
    if d == 1:
        return True
    x = 2
    # x^(2^d) mod p must equal x
    frob = x
    for _ in range(d):
        frob = _pmod(_pmul(frob, frob), p)
    if frob != _pmod(x, p):
        return False
    for q in {prime for prime in _primes_up_to(d) if d % prime == 0}:
        e = d // q
        frob = x
        for _ in range(e):
            frob = _pmod(_pmul(frob, frob), p)
        if _pdeg(_pgcd(frob ^ x, p)) != 0:
            return False
    return True


def _equal_degree_split(product: int, degree: int, rng: np.random.Generator) -> list[int]:
    """Cantor-Zassenhaus splitting of a product of degree-``degree`` factors."""

    if _pdeg(product) == degree:
        return [product]
    total = _pdeg(product)
    for _ in range(64):
        coefficients = rng.integers(0, 2, size=total)
        bits = 0
        for index in np.flatnonzero(coefficients):
            bits |= 1 << int(index)
        if bits == 0:
            continue
        trace = 0
        current = _pmod(bits, product)
        for _ in range(degree):
            trace ^= current
            current = _pmod(_pmul(current, current), product)
        w = _pgcd(trace, product)
        if 0 < _pdeg(w) < _pdeg(product):
            quotient, remainder = _pdivmod(product, w)
            if remainder:
                raise AssertionError("internal error: gcd does not divide its argument")
            return _equal_degree_split(w, degree, rng) + _equal_degree_split(
                quotient, degree, rng
            )
    return []  # split not found within the attempt budget; caller treats as miss


def _pfactors(f: int, rng: np.random.Generator) -> list[int]:
    """Some (Rabin-verified) irreducible factors of ``f`` over GF(2).

    Repeated squares are peeled off first; the odd-multiplicity part is then
    factored by distinct-degree + equal-degree splitting.  The returned list
    may omit even-multiplicity factors — callers only need *witness* factors,
    and every returned polynomial is re-verified irreducible.
    """

    factors: set[int] = set()
    while _pdeg(f) >= 1:
        derivative = _pderiv(f)
        if derivative == 0:
            f = _psqrt(f)
            continue
        squarefree, _ = _pdivmod(f, _pgcd(f, derivative))
        g = squarefree
        h = 2  # the polynomial x
        d = 0
        while _pdeg(g) >= 1 and d < _pdeg(squarefree) + 1:
            d += 1
            h = _pmod(_pmul(h, h), g)
            w = _pgcd(h ^ 2, g)
            if _pdeg(w) > 0:
                for factor in _equal_degree_split(w, d, rng):
                    factors.add(factor)
                g, remainder = _pdivmod(g, w)
                if remainder:
                    raise AssertionError("internal error: DDF component does not divide")
                h = _pmod(h, g) if _pdeg(g) >= 1 else 0
        break
    return [factor for factor in sorted(factors) if _pirreducible(factor)]


# ---------------------------------------------------------------------------
# GF(2)[x] as low-degree-first coefficient arrays
# ---------------------------------------------------------------------------

def _hessenberg(matrix: np.ndarray) -> np.ndarray:
    """Upper-Hessenberg form of an F_2 matrix under GF(2) similarity."""

    h = (matrix % 2).astype(np.uint8).copy()
    d = h.shape[0]
    for m in range(d - 2):
        pivots = np.flatnonzero(h[m + 1 :, m])
        if pivots.size == 0:
            continue
        p = m + 1 + int(pivots[0])
        if p != m + 1:
            h[[m + 1, p]] = h[[p, m + 1]]
            h[:, [m + 1, p]] = h[:, [p, m + 1]]
        for r in np.flatnonzero(h[m + 2 :, m]) + (m + 2):
            # similarity by I + e_r e_{m+1}^T (self-inverse over F_2):
            # row operation followed by the matching column operation
            h[r] ^= h[m + 1]
            h[:, m + 1] ^= h[:, r]
    return h


def _charpoly_f2(matrix: np.ndarray) -> np.ndarray:
    """Characteristic polynomial over F_2, low-degree-first bit vector.

    Hessenberg reduction followed by the standard leading-principal-minor
    recurrence ``p_m = (lambda + h_mm) p_{m-1} + sum_i h_im (prod
    subdiagonal) p_{i-1}`` — exact over GF(2), O(dim^3).
    """

    d = matrix.shape[0]
    h = _hessenberg(matrix)
    polys = [np.ones(1, dtype=np.uint8)]
    for m in range(1, d + 1):
        prev = polys[m - 1]
        cur = np.zeros(m + 1, dtype=np.uint8)
        cur[1:] ^= prev  # lambda * p_{m-1}
        if h[m - 1, m - 1]:
            cur[:m] ^= prev
        subdiagonal = 1
        for i in range(m - 2, -1, -1):
            subdiagonal &= int(h[i + 1, i])
            if not subdiagonal:
                break
            if h[i, m - 1]:
                cur[: i + 1] ^= polys[i]
        polys.append(cur)
    return polys[d]


def _minimal_polynomial(matrix: np.ndarray) -> np.ndarray:
    """Minimal polynomial of an F2 matrix, low-degree-first bit vector."""

    d = matrix.shape[0]
    powers = [np.eye(d, dtype=np.uint8).reshape(-1)]
    current = np.eye(d, dtype=np.uint8)
    for _ in range(d):
        current = (current @ matrix) % 2
        powers.append(current.reshape(-1).copy())
    stack = np.zeros((0, d * d), dtype=np.uint8)
    for degree, row in enumerate(powers):
        if rank(np.vstack([stack, row[None, :]])) == stack.shape[0]:
            coeffs = _solve_coords(stack, row)
            poly = np.zeros(degree + 1, dtype=np.uint8)
            poly[degree] = 1
            poly[:degree] = coeffs[: degree]
            return poly
        stack = np.vstack([stack, row[None, :]])
    raise AssertionError("minimal polynomial not found")


def _poly_gcd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.trim_zeros(a, "b")
    b = np.trim_zeros(b, "b")
    while b.size:
        a, b = b, _poly_mod(a, b)
        b = np.trim_zeros(b, "b")
    return a


def _poly_mod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = list(np.trim_zeros(a, "b"))
    b = list(np.trim_zeros(b, "b"))
    while len(a) >= len(b) and a:
        if a[-1]:
            shift = len(a) - len(b)
            for i, coeff in enumerate(b):
                a[shift + i] ^= coeff
        a.pop()
    return np.asarray(a + [0], dtype=np.uint8) if a else np.zeros(1, dtype=np.uint8)


def _berlekamp_factor(poly: np.ndarray) -> list[np.ndarray]:
    """Distinct irreducible factors of a square-free F2 polynomial."""

    poly = np.trim_zeros(poly, "b")
    degree = poly.size - 1
    if degree <= 1:
        return [poly]
    # Berlekamp Q matrix: x^{2i} mod poly
    Q = np.zeros((degree, degree), dtype=np.uint8)
    for i in range(degree):
        x2i = np.zeros(2 * i + 1, dtype=np.uint8)
        x2i[2 * i] = 1
        rem = _poly_mod(x2i, poly)
        Q[: rem.size - 1 if rem.size > degree else rem.size, i] = 0
        rem_t = np.trim_zeros(rem, "b")
        for j, c in enumerate(rem_t):
            if j < degree:
                Q[j, i] = c
    kernel = nullspace((Q ^ np.eye(degree, dtype=np.uint8)).T)
    if kernel.shape[0] <= 1:
        return [poly]
    # split with a non-constant kernel element
    for v in kernel:
        if np.flatnonzero(v).size and not (np.flatnonzero(v).size == 1 and v[0]):
            g = np.trim_zeros(v, "b")
            for c in (0, 1):
                shifted = g.copy()
                shifted[0] ^= c
                factor = _poly_gcd(poly, shifted)
                if 0 < factor.size - 1 < degree:
                    left = _poly_div(poly, factor)
                    return _berlekamp_factor(factor) + _berlekamp_factor(left)
    return [poly]


def _poly_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = list(np.trim_zeros(a, "b"))
    b = list(np.trim_zeros(b, "b"))
    out = [0] * (len(a) - len(b) + 1)
    while len(a) >= len(b) and a:
        if a[-1]:
            shift = len(a) - len(b)
            out[shift] = 1
            for i, coeff in enumerate(b):
                a[shift + i] ^= coeff
        a.pop()
    return np.asarray(out, dtype=np.uint8)


def _poly_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.size + b.size - 1, dtype=np.uint8)
    for i, ca in enumerate(a):
        if ca:
            out[i : i + b.size] ^= b
    return out


# ---------------------------------------------------------------------------
# Matrix polynomial helpers (integer-bitmask representation)
# ---------------------------------------------------------------------------

def _vector_minpoly(matrix: np.ndarray, vector: np.ndarray) -> int:
    """Monic minimal polynomial of ``matrix`` on ``vector`` (row action).

    Krylov elimination with combination tracking: the first linear dependence
    among ``v, vM, vM^2, ...`` is the (monic) minimal annihilator of ``v``,
    which divides the minimal polynomial and hence the characteristic
    polynomial of ``M``.
    """

    rows: list[tuple[np.ndarray, int, int]] = []  # (reduced row, combination, pivot)
    current = np.asarray(vector, dtype=np.uint8).copy()
    power = 0
    while True:
        reduced = current.copy()
        combination = 1 << power
        for row, combo, pivot in rows:
            if reduced[pivot]:
                reduced ^= row
                combination ^= combo
        support = np.flatnonzero(reduced)
        if support.size == 0:
            return combination
        rows.append((reduced, combination, int(support[0])))
        current = _gf2_matmul(current[None, :], matrix)[0]
        power += 1


def _poly_eval_matrix(p: int, matrix: np.ndarray) -> np.ndarray:
    """Evaluate a GF(2)[x] polynomial at a square matrix (Horner)."""

    size = matrix.shape[0]
    result = np.zeros((size, size), dtype=np.uint8)
    identity = np.eye(size, dtype=np.uint8)
    for bit in range(_pdeg(p), -1, -1):
        result = _gf2_matmul(result, matrix)
        if (p >> bit) & 1:
            result ^= identity
    return result


def _matrix_order(matrix: np.ndarray, *, cap: int = 65_536) -> int:
    """Exact multiplicative order by iterated powering (raises past ``cap``)."""

    size = matrix.shape[0]
    identity = np.eye(size, dtype=np.uint8)
    power = matrix.copy()
    for exponent in range(1, cap + 1):
        if np.array_equal(power, identity):
            return exponent
        power = _gf2_matmul(power, matrix)
    raise RuntimeError(f"matrix order exceeds cap {cap}")
