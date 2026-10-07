"""Constructors for standard CSS and quantum LDPC code families.

Every constructor returns a pair ``(h_x, h_z)`` of binary ``uint8`` matrices
satisfying ``H_X H_Z^T = 0`` over GF(2), suitable for :class:`~.css.CSSCode`.

Monomial conventions follow arXiv:2308.07915: with cyclic shift matrices
``S_l`` and ``S_m``, set ``x = S_l (x) I_m`` and ``y = I_l (x) S_m``.  A
bivariate polynomial is given as an iterable of ``(i, j)`` exponent pairs
meaning ``x^i y^j``; a univariate polynomial as an iterable of integers.
"""


from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np

from ..utils.gf2 import BinaryMatrix, as_binary_matrix
from .stabilizer import five_qubit_code


def cyclic_shift(size: int) -> BinaryMatrix:
    """The ``size x size`` cyclic shift permutation matrix ``S``.

    ``S`` maps basis vector ``e_i`` to ``e_(i+1 mod size)`` under row-vector
    convention ``e_i S``.
    """

    if size < 1:
        raise ValueError("size must be positive")
    return np.eye(size, dtype=np.uint8)[:, np.roll(np.arange(size), 1)]


def circulant(size: int, exponents: Iterable[int]) -> BinaryMatrix:
    """The circulant matrix ``sum_e S^e`` over GF(2)."""

    matrix = np.zeros((size, size), dtype=np.uint8)
    rows = np.arange(size)
    seen: set[int] = set()
    for exponent in exponents:
        reduced = exponent % size
        if reduced in seen:
            raise ValueError(f"repeated exponent {exponent} modulo {size}")
        seen.add(reduced)
        matrix[rows, (rows + reduced) % size] ^= 1
    return matrix


def bivariate_monomial_sum(
    l: int, m: int, monomials: Iterable[Sequence[int]]
) -> BinaryMatrix:
    """The ``lm x lm`` matrix ``sum x^i y^j`` for exponent pairs ``(i, j)``."""

    matrix = np.zeros((l * m, l * m), dtype=np.uint8)
    block = np.arange(l * m).reshape(l, m)
    rows = block.reshape(-1)
    seen: set[tuple[int, int]] = set()
    for monomial in monomials:
        if len(monomial) != 2:
            raise ValueError(f"expected (i, j) exponent pairs, got {monomial!r}")
        i, j = int(monomial[0]) % l, int(monomial[1]) % m
        if (i, j) in seen:
            raise ValueError(f"repeated monomial x^{i} y^{j}")
        seen.add((i, j))
        columns = np.roll(np.roll(block, -i, axis=0), -j, axis=1).reshape(-1)
        matrix[rows, columns] ^= 1
    return matrix


def bivariate_bicycle(
    l: int,
    m: int,
    a_monomials: Iterable[Sequence[int]],
    b_monomials: Iterable[Sequence[int]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Bivariate bicycle code ``H_X = [A | B]``, ``H_Z = [B^T | A^T]``.

    ``n = 2 l m``.  See Bravyi et al., arXiv:2308.07915.
    """

    a = bivariate_monomial_sum(l, m, a_monomials)
    b = bivariate_monomial_sum(l, m, b_monomials)
    h_x = np.hstack([a, b])
    h_z = np.hstack([b.T, a.T])
    return h_x, h_z


def trivariate_tricycle(
    l: int,
    m: int,
    p: int,
    a_monomials: Iterable[Sequence[int]],
    b_monomials: Iterable[Sequence[int]],
    c_monomials: Iterable[Sequence[int]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Trivariate tricycle code of arXiv:2508.08191, Eqs. (13)-(14).

    ``A``, ``B``, ``C`` are sums of monomials ``x^i y^j z^k`` (exponent
    triples) in the commuting shifts ``x = S_l (x) 1 (x) 1``,
    ``y = 1 (x) S_m (x) 1``, ``z = 1 (x) 1 (x) S_p``.  ``H_X = [A | B | C]``
    and ``H_Z = [[0, C^T, B^T], [C^T, 0, A^T], [B^T, A^T, 0]]``;
    ``n = 3 l m p``.  The ``Z`` checks are redundant (meta-checks
    ``[A^T | B^T | C^T]``), so ``H_Z`` is not full rank.
    """

    if min(l, m, p) < 1:
        raise ValueError("l, m, p must be positive")
    size = l * m * p
    block = np.arange(size).reshape(l, m, p)
    rows = block.reshape(-1)

    def monomial_sum(monomials: Iterable[Sequence[int]]) -> BinaryMatrix:
        matrix = np.zeros((size, size), dtype=np.uint8)
        seen: set[tuple[int, int, int]] = set()
        for monomial in monomials:
            if len(monomial) != 3:
                raise ValueError(f"expected (i, j, k) exponent triples, got {monomial!r}")
            i, j, k = int(monomial[0]) % l, int(monomial[1]) % m, int(monomial[2]) % p
            if (i, j, k) in seen:
                raise ValueError(f"repeated monomial x^{i} y^{j} z^{k}")
            seen.add((i, j, k))
            columns = np.roll(block, (-i, -j, -k), axis=(0, 1, 2)).reshape(-1)
            matrix[rows, columns] ^= 1
        return matrix

    a, b, c = (monomial_sum(mono) for mono in (a_monomials, b_monomials, c_monomials))
    zero = np.zeros_like(a)
    h_x = np.hstack([a, b, c])
    h_z = np.vstack(
        [np.hstack([zero, c.T, b.T]), np.hstack([c.T, zero, a.T]), np.hstack([b.T, a.T, zero])]
    )
    return h_x, h_z


def _twisted_torus_basis(
    basis_1: Sequence[int], basis_2: Sequence[int]
) -> tuple[int, int, int]:
    """Upper-triangular (Hermite) basis ``(h11, h12), (0, h22)`` of the lattice
    spanned by ``basis_1`` and ``basis_2``.

    The quotient group has order ``h11 * h22 = |det|``, so every cell of the
    twisted torus has the canonical representative computed in
    :func:`twisted_torus_translation`.
    """

    (p, q), (r, s) = basis_1, basis_2
    determinant = p * s - q * r
    if determinant == 0:
        raise ValueError(f"degenerate torus basis {basis_1!r}, {basis_2!r}")
    # Extended Euclid on the first components: u*p + v*r = gcd(p, r).
    prev_remainder, remainder = p, r
    prev_u, u = 1, 0
    prev_v, v = 0, 1
    while remainder:
        quotient = prev_remainder // remainder
        prev_remainder, remainder = remainder, prev_remainder - quotient * remainder
        prev_u, u = u, prev_u - quotient * u
        prev_v, v = v, prev_v - quotient * v
    h11, sign = abs(prev_remainder), 1 if prev_remainder > 0 else -1
    h22 = abs(determinant) // h11
    h12 = sign * (prev_u * q + prev_v * s) % h22
    return h11, h12, h22


def twisted_torus_translation(
    monomials: Iterable[Sequence[int]],
    basis_1: Sequence[int],
    basis_2: Sequence[int],
) -> BinaryMatrix:
    """``sum x^i y^j`` as a matrix over the group ring of ``Z^2 / <a1, a2>``.

    Generalizes :func:`bivariate_monomial_sum` from the rectangular torus
    ``Z_l x Z_m`` to the *twisted* tori of arXiv:2510.05211, whose cells are
    the cosets of the lattice spanned by ``basis_1`` and ``basis_2``.
    """

    h11, h12, h22 = _twisted_torus_basis(basis_1, basis_2)
    size = h11 * h22

    def index(a: int, b: int) -> int:
        shift = a // h11
        return (a - shift * h11) * h22 + (b - shift * h12) % h22

    pairs = [(int(i), int(j)) for i, j in monomials]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"repeated monomial in {pairs!r}")
    matrix = np.zeros((size, size), dtype=np.uint8)
    for cell in range(size):
        a, b = divmod(cell, h22)
        for i, j in pairs:
            matrix[cell, index(a + i, b + j)] ^= 1
    return matrix


def self_dual_bicycle(
    monomials: Iterable[Sequence[int]],
    basis_1: Sequence[int],
    basis_2: Sequence[int],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Self-dual bivariate bicycle code of Liang-Chen, arXiv:2510.05211.

    With ``F`` the group-ring matrix of ``f(x, y)`` on the twisted torus and
    ``g := f`` conjugated by the antipode ``x^i y^j -> x^-i y^-j`` (so
    ``G = F^T``), Eq. (3) gives ``H_X = [F | G]`` and ``H_Z = [G^T | F^T]``,
    i.e. ``H_X = H_Z = [F | F^T]``: a genuinely self-dual CSS code.  For the
    weight-8 family of Eq. (14) the checks are doubly even, so the code carries
    a *strict* transversal H and S at LDPC check weight -- the registry's
    positive sparse control.
    """

    f = twisted_torus_translation(monomials, basis_1, basis_2)
    h = np.hstack([f, f.T])
    return h, h.copy()


def generalized_bicycle(
    size: int, a_exponents: Iterable[int], b_exponents: Iterable[int]
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Generalized bicycle code from two circulants of one cyclic group.

    ``H_X = [A | B]``, ``H_Z = [B^T | A^T]``, ``n = 2 * size``.  See
    Panteleev and Kalachev, arXiv:1904.02703.
    """

    a = circulant(size, a_exponents)
    b = circulant(size, b_exponents)
    return np.hstack([a, b]), np.hstack([b.T, a.T])


def hypergraph_product(
    h1: object, h2: object
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Hypergraph product of two classical parity-check matrices.

    With ``h1`` of shape ``(r1, n1)`` and ``h2`` of shape ``(r2, n2)``:

    ``H_X = [h1 (x) I_n2 | I_r1 (x) h2^T]``,
    ``H_Z = [I_n1 (x) h2 | h1^T (x) I_r2]``,

    acting on ``n = n1 n2 + r1 r2`` qubits.  See Tillich and Zemor,
    arXiv:0903.0566.
    """

    first = np.asarray(h1, dtype=np.uint8) & 1
    second = np.asarray(h2, dtype=np.uint8) & 1
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("expected two-dimensional parity-check matrices")
    r1, n1 = first.shape
    r2, n2 = second.shape
    h_x = np.hstack(
        [np.kron(first, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(r1, dtype=np.uint8), second.T)]
    )
    h_z = np.hstack(
        [np.kron(np.eye(n1, dtype=np.uint8), second), np.kron(first.T, np.eye(r2, dtype=np.uint8))]
    )
    return (h_x & 1), (h_z & 1)


def repetition_ring(size: int) -> BinaryMatrix:
    """Cyclic repetition-code checks ``1 + x`` (rank ``size - 1``)."""

    return circulant(size, [0, 1])


def toric_code(distance: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[2 d^2, 2, d]]`` toric code as a hypergraph product."""

    ring = repetition_ring(distance)
    return hypergraph_product(ring, ring)


def hamming_7_4() -> BinaryMatrix:
    """Parity checks of the classical ``[7, 4, 3]`` Hamming code."""

    return np.asarray(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def steane_code() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[7, 1, 3]]`` Steane code."""

    checks = hamming_7_4()
    return checks.copy(), checks.copy()


def iceberg(pairs: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[2m, 2m-2, 2]]`` iceberg error-detection code.

    One global X stabilizer and one global Z stabilizer; the transversal
    ``sqrt(Z)`` layer acts as the product of logical CZ on every pair of
    logical qubits.  Rate approaches 1 at fixed distance 2.
    """

    if pairs < 2:
        raise ValueError("pairs must be at least 2")
    row = np.ones((1, 2 * pairs), dtype=np.uint8)
    return row.copy(), row.copy()


def quantum_reed_muller_15() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[15, 1, 3]]`` punctured quantum Reed-Muller code.

    X checks are the four coordinate-bit vectors of ``1..15``; Z checks add
    their six pairwise coordinatewise products.  ``C_X`` is contained in
    ``C_Z``, so the code supports transversal diagonal gates.
    """

    columns = np.arange(1, 16)
    bits = np.stack([(columns >> shift) & 1 for shift in range(4)]).astype(np.uint8)
    products = [bits[i] & bits[j] for i in range(4) for j in range(i + 1, 4)]
    h_x = bits
    h_z = np.vstack([bits, np.asarray(products, dtype=np.uint8)])
    return h_x, h_z


def quantum_reed_muller_31() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[31, 1, 3]]`` punctured quantum Reed-Muller code.

    X checks are the five coordinate-bit vectors of ``1..31``; Z checks add
    all pairwise and triple coordinatewise products.  The code carries a
    transversal gate at the *fourth* Clifford-hierarchy level (the
    ``sqrt(T)`` family).
    """

    columns = np.arange(1, 32)
    bits = np.stack([(columns >> shift) & 1 for shift in range(5)]).astype(np.uint8)
    pairs = [bits[i] & bits[j] for i in range(5) for j in range(i + 1, 5)]
    triples = [
        bits[i] & bits[j] & bits[l]
        for i in range(5)
        for j in range(i + 1, 5)
        for l in range(j + 1, 5)
    ]
    h_x = bits
    h_z = np.vstack([bits, np.asarray(pairs, np.uint8), np.asarray(triples, np.uint8)])
    return h_x, h_z


# Cyclic triorthogonal codes, arXiv:2610.08012, construction A.  A binary
# cyclic code of odd length n is fixed by its spectral support S, the j in
# Z_n with g(alpha^j) != 0 for its generator polynomial g and a primitive
# n-th root of unity alpha in GF(2^m), m = ord_n(2) (Definition 1); S is a
# union of 2-cyclotomic cosets and dim C = |S|.  Field elements and binary
# polynomials are both int bit masks below.


def _gf2_poly_mod(a: int, b: int) -> int:
    width = b.bit_length()
    while a and a.bit_length() >= width:
        a ^= b << (a.bit_length() - width)
    return a


def _gf2_poly_mulmod(a: int, b: int, modulus: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a = _gf2_poly_mod(a << 1, modulus)
    return result


def _gf2_poly_powmod(a: int, exponent: int, modulus: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf2_poly_mulmod(result, a, modulus)
        a = _gf2_poly_mulmod(a, a, modulus)
        exponent >>= 1
    return result


def _irreducible_polynomial(degree: int) -> int:
    """The lexicographically first irreducible binary polynomial of ``degree`` (Ben-Or test)."""

    for candidate in range(1 << degree | 1, 1 << (degree + 1), 2):
        power, irreducible = 2, True
        for _ in range(degree // 2):
            power = _gf2_poly_mulmod(power, power, candidate)
            a, b = candidate, power ^ 2
            while b:
                a, b = b, _gf2_poly_mod(a, b)
            if a != 1:
                irreducible = False
                break
        if irreducible:
            return candidate
    raise ValueError(f"no irreducible polynomial of degree {degree}")


def cyclic_triorthogonal(length: int, support: Iterable[int]) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[n, 1]]`` cyclic triorthogonal code with spectral support ``S``.

    Construction A of arXiv:2610.08012 (Proposition 12): X checks are the
    ``|S|`` cyclic shifts of the generator polynomial of the binary cyclic
    code ``C`` of odd length ``n`` with spectral support ``S``, the logical
    X is all-ones, and Z checks are the ``n - |S| - 1`` cyclic shifts of the
    generator of ``C^perp`` intersected with the even-weight vectors, whose
    zeros are ``-S`` and ``0`` (Lemma 2).  ``0 not in S + S + S`` makes
    ``C`` triply even, hence the pair Bravyi-Haah triorthogonal with a
    transversal T (Theorem 16); it is required here.  ``alpha`` is taken in
    the field cut out by the first irreducible polynomial of degree
    ``ord_n(2)``; the paper's GAP root may differ from it by a multiplier
    ``u``, which maps ``S`` to ``uS`` and the code to a coordinate
    permutation of itself (Appendix C).  ``S = {1, 2, 4, 8}`` at ``n = 15``
    is the ``[[15,1,3]]`` punctured Reed-Muller code; the paper's records
    are the ``[[85,1,5]]``, ``[[127,1,7]]`` and ``[[223,1,9]]`` supports of
    its Appendix C.
    """

    if length < 3 or length % 2 == 0:
        raise ValueError("length must be odd and at least 3")
    spectrum = sorted({int(j) % length for j in support})
    if not spectrum or any((2 * j) % length not in spectrum for j in spectrum):
        raise ValueError("support must be a nonempty union of 2-cyclotomic cosets")
    if any((a + b + c) % length == 0 for a in spectrum for b in spectrum for c in spectrum):
        raise ValueError("0 lies in S + S + S: the cyclic code is not triply even")
    degree, power = 1, 2 % length
    while power != 1:
        power, degree = 2 * power % length, degree + 1
    modulus = _irreducible_polynomial(degree)
    cofactor = ((1 << degree) - 1) // length
    factors = {p for p in range(2, length + 1) if length % p == 0 and all(p % q for q in range(2, p))}
    for seed in range(2, 1 << degree):
        alpha = _gf2_poly_powmod(seed, cofactor, modulus)
        if alpha != 1 and all(_gf2_poly_powmod(alpha, length // p, modulus) != 1 for p in factors):
            break
    roots = [_gf2_poly_powmod(alpha, j, modulus) for j in range(length)]

    def binary_generator(zeros: Sequence[int]) -> list[int]:
        poly = [1]
        for j in zeros:
            poly = [a ^ b for a, b in zip([0] + poly, [_gf2_poly_mulmod(c, roots[j], modulus) for c in poly] + [0])]
        if any(c not in (0, 1) for c in poly):
            raise AssertionError("generator polynomial is not binary")
        return poly

    def cyclic_rows(poly: Sequence[int], count: int) -> BinaryMatrix:
        rows = np.zeros((count, length), dtype=np.uint8)
        for shift in range(count):
            for i, c in enumerate(poly):
                if c:
                    rows[shift, (i + shift) % length] = 1
        return rows

    g_x = binary_generator([j for j in range(length) if j not in spectrum])
    g_z = binary_generator([(-j) % length for j in spectrum] + [0])
    return cyclic_rows(g_x, len(spectrum)), cyclic_rows(g_z, length - len(spectrum) - 1)


def reed_muller_generator(order: int, variables: int) -> BinaryMatrix:
    """Generator matrix of the classical Reed-Muller code ``RM(order, m)``."""

    if order < 0 or variables < 1:
        raise ValueError("expected order >= 0 and variables >= 1")
    points = np.arange(2**variables)
    coordinates = [((points >> index) & 1).astype(np.uint8) for index in range(variables)]
    rows: list[BinaryMatrix] = []
    for degree in range(order + 1):
        for combo in combinations(range(variables), degree):
            row = np.ones(2**variables, dtype=np.uint8)
            for coordinate in combo:
                row &= coordinates[coordinate]
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def middle_reed_muller(variables: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The self-dual CSS code with ``C_X = C_Z = RM(m/2 - 1, m)`` for even m.

    Parameters ``[[2^m, C(m, m/2), 2^(m/2)]]``; ``m = 4`` is the ``[[16,6,4]]``
    tesseract code.  See Albert, arXiv:2608.05688.
    """

    if variables % 2 != 0 or variables < 2:
        raise ValueError("variables must be even and at least 2")
    checks = reed_muller_generator(variables // 2 - 1, variables)
    return checks.copy(), checks.copy()


def bipartite_grid(a: int, b: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Albert's bipartite-grid code on an ``a x b`` cell grid.

    ``C_X = C_Z`` is spanned by all row-plus-column indicators; for even
    ``a, b`` with ``a + b = 2 mod 4`` this is a doubly-even self-dual
    ``[[ab, (a-2)(b-2), 4]]`` code with full transversal Clifford group.
    """

    if a < 2 or b < 2:
        raise ValueError("expected a, b >= 2")
    checks = []
    for i in range(a):
        for j in range(b):
            cell = np.zeros((a, b), dtype=np.uint8)
            cell[i, :] ^= 1
            cell[:, j] ^= 1
            checks.append(cell.reshape(-1))
    matrix = np.asarray(checks, dtype=np.uint8)
    return matrix.copy(), matrix.copy()


def surface_code(distance: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The open-boundary ``[[d^2 + (d-1)^2, 1, d]]`` surface code.

    Hypergraph product of the ``(d-1) x d`` repetition-chain matrix with
    itself.
    """

    if distance < 2:
        raise ValueError("distance must be at least 2")
    chain = np.zeros((distance - 1, distance), dtype=np.uint8)
    for i in range(distance - 1):
        chain[i, i] = chain[i, i + 1] = 1
    return hypergraph_product(chain, chain)


def la_cross(size: int, reach: int, *, periodic: bool = False) -> tuple[BinaryMatrix, BinaryMatrix]:
    """La-cross code from the seed polynomial ``1 + x + x^reach``.

    Hypergraph product of the seed with itself: the open-boundary form gives
    ``[[size^2 + (size-reach)^2, reach^2, d]]``, the periodic form
    ``[[2 size^2, 2 reach^2, d]]``.  See Pecorari et al., arXiv:2404.13010.
    """

    if not 0 < reach < size:
        raise ValueError("expected 0 < reach < size")
    if periodic:
        seed = circulant(size, [0, 1, reach])
    else:
        seed = np.zeros((size - reach, size), dtype=np.uint8)
        for i in range(size - reach):
            seed[i, i] = seed[i, i + 1] = seed[i, i + reach] = 1
    return hypergraph_product(seed, seed)


def lifted_product_b1() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The Panteleev-Kalachev ``[[882, 24]]`` generalized hypergraph product.

    Over ``R = F2[x]/(x^63 - 1)`` with the ``7 x 7`` block matrix
    ``A[i,i] = x^27``, ``A[i,i+5] = 1``, ``A[i,i+6] = x^54`` (indices mod 7)
    and ``B = (1 + x + x^6) I_7``.  See arXiv:1904.02703, Appendix B.
    """

    ell, blocks = 63, 7
    a = np.zeros((blocks * ell, blocks * ell), dtype=np.uint8)
    b = np.zeros((blocks * ell, blocks * ell), dtype=np.uint8)
    for i in range(blocks):
        for j, exponent in (((i + 0) % blocks, 27), ((i + 5) % blocks, 0), ((i + 6) % blocks, 54)):
            a[i * ell : (i + 1) * ell, j * ell : (j + 1) * ell] ^= circulant(ell, [exponent])
        b[i * ell : (i + 1) * ell, i * ell : (i + 1) * ell] = circulant(ell, [0, 1, 6])
    h_x = np.hstack([a, b])
    h_z = np.hstack([b.T, a.T])
    return h_x, h_z


def lifted_product_monomial(
    lift: int, exponents: Sequence[Sequence[int]]
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Lifted product ``LP(A, A^dagger)`` of a monomial protograph over a cyclic group.

    ``A`` is the ``r x c`` matrix over ``R = F_2[x]/(x^lift - 1)`` with
    ``A[i][j] = x^exponents[i][j]``, and ``B = A^dagger`` (transpose with
    ``x -> x^-1``).  Following arXiv:2609.39874, Eq. (S37), with the qubit
    ordering ``(A_1 (x) B_0) + (A_0 (x) B_1)``:

    ``H_X = (A (x) I_c | I_r (x) A^dagger)``,
    ``H_Z = (I_c (x) A | A^dagger (x) I_r)``,

    tensor products over ``R``, and ``x^s`` lifted to the cyclic permutation
    ``P(s)[a, a + s] = 1``.  ``n = lift (c^2 + r^2)``; every check has weight
    ``r + c``.  Commutation is checked here rather than assumed.
    """

    rows = [[int(e) for e in row] for row in exponents]
    r = len(rows)
    c = len(rows[0]) if r else 0
    if lift < 1 or r < 1 or c < 1 or any(len(row) != c for row in rows):
        raise ValueError("expected a positive lift and a rectangular exponent matrix")

    def lifted(n_rows: int, n_cols: int, entries: dict[tuple[int, int], int]) -> BinaryMatrix:
        out = np.zeros((n_rows * lift, n_cols * lift), dtype=np.uint8)
        for (i, j), shift in entries.items():
            out[i * lift : (i + 1) * lift, j * lift : (j + 1) * lift] = circulant(lift, [shift])
        return out

    cells = [(i, j) for i in range(r) for j in range(c)]
    a_i = {(i * c + t, j * c + t): rows[i][j] for i, j in cells for t in range(c)}
    i_ad = {(s * c + j, s * r + i): -rows[i][j] for i, j in cells for s in range(r)}
    i_a = {(t * r + i, t * c + j): rows[i][j] for i, j in cells for t in range(c)}
    ad_i = {(j * r + s, i * r + s): -rows[i][j] for i, j in cells for s in range(r)}
    h_x = np.hstack([lifted(r * c, c * c, a_i), lifted(r * c, r * r, i_ad)])
    h_z = np.hstack([lifted(c * r, c * c, i_a), lifted(c * r, r * r, ad_i)])
    if ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any():
        raise ValueError("lifted product checks do not commute")
    return h_x, h_z


def kasai_binary_pair(width: int, lift: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The binary orthogonal quasi-cyclic pair underlying Kasai-style codes.

    Definition 6 of Komoto and Kasai, arXiv:2501.13444 (``J = 2`` rows of
    circulant permutation blocks, column weight two): with ``H = width/2``,
    shift amounts ``f_l = 2^l`` and ``g_l = 2^(l + H)``,

    - ``H_X`` block ``(j, l)`` shifts by ``f[(l - j) mod H]`` for ``l < H``
      and ``g[(l - j - H) mod H]`` otherwise;
    - ``H_Z`` block ``(j, l)`` shifts by ``-g[(j - l) mod H]`` for ``l < H``
      and ``-f[(j - l + H) mod H]`` otherwise.

    Orthogonality holds for every ``lift``; girth 12 requires ``lift`` at or
    above the published threshold (49 for ``width = 6``, 138 for ``8``).
    The resulting CSS code has ``n = width * lift`` and
    ``k = (width - 4) * lift + 2``.
    """

    if width < 6 or width % 2 != 0:
        raise ValueError("width must be an even integer of at least 6")
    if lift < 2:
        raise ValueError("lift must be at least 2")
    half = width // 2
    f_shifts = [pow(2, l, lift) for l in range(half)]
    g_shifts = [pow(2, l + half, lift) for l in range(half)]
    h_x = np.zeros((2 * lift, width * lift), dtype=np.uint8)
    h_z = np.zeros((2 * lift, width * lift), dtype=np.uint8)
    for j in range(2):
        for l in range(width):
            if l < half:
                shift_x = f_shifts[(l - j) % half]
                shift_z = -g_shifts[(j - l) % half]
            else:
                shift_x = g_shifts[(l - j - half) % half]
                shift_z = -f_shifts[(j - l + half) % half]
            h_x[j * lift : (j + 1) * lift, l * lift : (l + 1) * lift] = circulant(lift, [shift_x])
            h_z[j * lift : (j + 1) * lift, l * lift : (l + 1) * lift] = circulant(lift, [shift_z])
    return h_x, h_z


def subset_inclusion(m: int, s: int, alpha: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Subset-inclusion (generalized WZL) quantum locally recoverable code.

    ``H_X = H_Z = H`` where ``H`` has one row per ``(s - alpha)``-subset
    ``E`` of ``{1..m}``, one column per ``s``-subset ``F``, and
    ``H[E, F] = 1`` iff ``E`` is contained in ``F``.  Dual containment holds
    exactly when ``C(m - u, s - u)`` is even for every ``u`` in
    ``s - alpha .. min(2(s - alpha), m)``.  See arXiv:2608.10912, Sec. IV;
    the ``(s, alpha) = (3, 2)``, ``m = 4l + 2`` members form the exact pure
    ``[[C(m,3), C(m,3) - 2m, 4]]`` subfamily of Example 1.
    """

    rows = list(combinations(range(m), s - alpha))
    cols = list(combinations(range(m), s))
    h = np.zeros((len(rows), len(cols)), dtype=np.uint8)
    for i, row_subset in enumerate(rows):
        for j, col_subset in enumerate(cols):
            if set(row_subset) <= set(col_subset):
                h[i, j] = 1
    return h, h.copy()


def helper_qss_css(parties: int) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The ``[[2m + 1, 1]]`` blind-helper CSS code on ``m`` odd parties.

    ``H_X = H_Z = [I_m | 1 | J_m]``: row ``i`` carries a single check qubit
    of the helper's first ``m`` qubits, the helper's shared last qubit, and
    every party qubit except the anti-diagonal one.  The helper holds the
    first ``m + 1`` columns and the ``m`` parties one column each; the
    logical pair ``X-bar = X...X``, ``Z-bar = Z...Z`` is supported on the
    party qubits alone, which is what makes the helper blind.  Row weight is
    ``m + 1``, even exactly when ``m`` is odd -- the paper's parity
    condition.  ``m = 3`` is the Steane code.  See arXiv:2609.00220,
    Example 6.
    """

    if parties < 3 or parties % 2 == 0:
        raise ValueError("parties must be an odd integer at least 3")
    width = 2 * parties + 1
    h = np.zeros((parties, width), dtype=np.uint8)
    for i in range(parties):
        h[i, i] = 1
        h[i, parties] = 1
        h[i, parties + 1:] = 1
        h[i, width - 1 - i] = 0
    return h, h.copy()


def _support_matrix(width: int, supports: Sequence[Sequence[int]]) -> BinaryMatrix:
    """Binary rows of the given width from qubit-support lists.

    A repeated qubit cancels, so a support may be handed the concatenated
    supports of a product of Pauli operators.
    """

    matrix = np.zeros((len(supports), width), dtype=np.uint8)
    for row, support in zip(matrix, supports):
        for qubit in support:
            row[qubit] ^= 1
    return matrix


def symplectic_double(x_part: BinaryMatrix, z_part: BinaryMatrix) -> tuple[BinaryMatrix, BinaryMatrix]:
    """The symplectic double ``D(H)`` of the stabilizer code ``H = (X | Z)``.

    ``D(H)`` is the CSS code on ``2n`` qubits with ``H_X = [X | Z]`` and
    ``H_Z = [Z | X]``; ``H_X H_Z^T = X Z^T + Z X^T`` vanishes exactly because
    ``H`` is symplectically self-orthogonal.  Qubit ``i`` and qubit ``i + n``
    are exchanged by the ZX-duality of the double.  See arXiv:2609.03194,
    Eq. (A2).
    """

    x_part = as_binary_matrix(x_part)
    z_part = as_binary_matrix(z_part)
    if x_part.shape != z_part.shape:
        raise ValueError("the X and Z halves must have the same shape")
    return np.hstack([x_part, z_part]), np.hstack([z_part, x_part])


# The [[12, 2, 4]] Carbon code, arXiv:2404.02280 Table IV.
_CARBON_X = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
             (0, 1, 5, 7, 8, 11), (0, 3, 4, 5, 9, 11))
_CARBON_Z = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
             (0, 2, 6, 7, 8, 11), (0, 3, 4, 6, 10, 11))
_CARBON_LOGICALS = ((0, 3, 10, 11), (0, 3, 9, 11), (1, 3, 9, 10), (0, 1, 9, 10))

# The [[20, 2, 6]] C4-Helix code, arXiv:2609.03194 Table I (right panel).
_C4_HELIX_LOGICALS = ((2, 3, 8, 11, 12, 15), (0, 3, 6, 7, 18, 19),
                      (0, 3, 6, 7, 18, 19), (2, 3, 8, 11, 12, 15))


def _helix_inner(inner: str) -> tuple[BinaryMatrix, BinaryMatrix, tuple[tuple[int, ...], ...]]:
    """``(H_X, H_Z, (X0, Z0, X1, Z1))`` for one helix inner code."""

    if inner == "c4":
        # The [[4, 2, 2]] code: XXXX / ZZZZ with logicals XIIX, IIZZ, IIXX, ZIIZ.
        block = _support_matrix(4, ((0, 1, 2, 3),))
        return block, block.copy(), ((0, 3), (2, 3), (2, 3), (0, 3))
    if inner == "carbon":
        return (_support_matrix(12, _CARBON_X), _support_matrix(12, _CARBON_Z),
                _CARBON_LOGICALS)
    if inner == "c4-helix":
        h_x, h_z = helix_code("c4")
        return h_x, h_z, _C4_HELIX_LOGICALS
    raise ValueError(f"unknown helix inner code {inner!r}")


def helix_code(inner: str = "c4") -> tuple[BinaryMatrix, BinaryMatrix]:
    """A concatenated symplectic double ("Helix") code, arXiv:2609.03194.

    The outer code is the ``[[10, 2, 3]]`` twisted toric code, the symplectic
    double of the perfect ``[[5, 1, 3]]`` code.  Its ZX-duality pairs qubit
    ``i`` with qubit ``i + 5``; the pair becomes one block of an
    ``[[m, 2, d]]`` inner code, the lower-indexed qubit taking logical 0.
    Every outer check then lifts to the product of the inner logicals sitting
    in its support, and the five inner check sets come along, for
    ``[[5m, 2, 3d]]``.  ``inner`` selects the inner code:

    ``"c4"``
        the ``[[4, 2, 2]]`` code, giving the ``[[20, 2, 6]]`` C4-Helix code
        (arXiv:2510.18753, Sec. 3.1.2; matrices in arXiv:2609.03194 Table I);
    ``"carbon"``
        the ``[[12, 2, 4]]`` Carbon code, giving ``[[60, 2, 12]]``;
    ``"c4-helix"``
        the ``[[20, 2, 6]]`` code itself, giving ``[[100, 2, 18]]``.

    The published generators multiply some lifted checks by an inner check;
    that leaves the stabilizer group, and so the code, unchanged.
    """

    # five_qubit_code() carries exactly the generators of arXiv:2609.03194
    # Eq. (A1); row reduction inside it leaves both halves of D(H) spanning
    # the same two check spaces.
    perfect = five_qubit_code().h
    outer_x, outer_z = symplectic_double(perfect[:, :5], perfect[:, 5:])
    inner_x, inner_z, (l_x0, l_z0, l_x1, l_z1) = _helix_inner(inner)
    width = inner_x.shape[1]
    x_rows: list[tuple[int, ...]] = []
    z_rows: list[tuple[int, ...]] = []
    for block in range(5):
        offset = block * width
        x_rows += [tuple(offset + np.flatnonzero(row)) for row in inner_x]
        z_rows += [tuple(offset + np.flatnonzero(row)) for row in inner_z]

    def lift(row: BinaryMatrix, logical_0: Sequence[int], logical_1: Sequence[int]):
        support: list[int] = []
        for qubit in np.flatnonzero(row):
            logical = logical_0 if qubit < 5 else logical_1
            support += [(qubit % 5) * width + q for q in logical]
        return tuple(support)

    x_rows += [lift(row, l_x0, l_x1) for row in outer_x]
    z_rows += [lift(row, l_z0, l_z1) for row in outer_z]
    return _support_matrix(5 * width, x_rows), _support_matrix(5 * width, z_rows)


def doubled_color_41() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The doubly-even self-orthogonal ``[[41, 1, 9]]`` doubled code.

    ``H_X = H_Z = G`` with ``G`` stacked from the all-even ``[9, 8, 2]``
    code repeated on two nine-qubit blocks, the doubly-even ``[23, 11, 8]``
    even subcode of the Golay code (generator polynomial
    ``1 + x + x^2 + x^3 + x^4 + x^7 + x^10 + x^12``), and one all-ones row
    on the last 32 qubits.  Every row weight is divisible by four.  See
    arXiv:2608.11160, Example III.5 (their Eq. (III.1) row
    ``(0_9 | 1_9 | v)`` with ``v`` a weight-7 logical spans the same
    stabilizer group because ``1_23 + v`` lies in the Golay subcode).
    """

    e1 = np.zeros((8, 9), dtype=np.uint8)
    for i in range(8):
        e1[i, i] = e1[i, i + 1] = 1
    e2 = np.zeros((11, 23), dtype=np.uint8)
    for i in range(11):
        for exponent in (0, 1, 2, 3, 4, 7, 10, 12):
            e2[i, (i + exponent) % 23] = 1
    g = np.zeros((20, 41), dtype=np.uint8)
    g[0:8, 0:9] = e1
    g[0:8, 9:18] = e1
    g[8:19, 18:41] = e2
    g[19, 9:41] = 1
    return g, g.copy()


# Depth-one universal codes, arXiv:2610.06730: per database label, the qubit
# count and the X and Z check supports (``ldpc_generators``) stored in the
# paper's ancillary file depth_one_universal_codes.json, qubits numbered as
# there.
_DEPTH_ONE_UNIVERSAL = {
    "ghz37": (
        37,
        ((25, 26), (2, 3), (35, 36), (0, 1), (5, 6), (3, 4), (10, 11), (20, 21), (15, 16),
         (30, 31), (12, 13, 17, 19, 32, 34), (7, 9, 22, 23, 32, 33),
         (12, 14, 22, 24, 27, 29), (7, 8, 17, 18, 27, 28), (3, 5, 12, 15, 17, 21, 27, 30),
         (0, 6, 11, 16, 21, 25, 30, 36), (1, 4, 11, 12, 21, 22, 31, 32),
         (4, 6, 7, 10, 17, 20, 32, 35)),
        ((8, 28), (14, 29), (13, 34), (19, 34), (18, 28), (14, 24), (9, 33), (23, 33),
         (0, 1, 15, 16, 20, 21, 35, 36), (0, 1, 10, 11, 25, 26, 35, 36),
         (2, 3, 4, 7, 17, 24, 29, 32), (5, 6, 12, 22, 23, 34, 35, 36),
         (0, 1, 10, 11, 13, 14, 17, 27), (7, 15, 16, 19, 28, 30, 31, 32),
         (10, 11, 12, 20, 21, 22, 33, 34), (0, 1, 5, 6, 30, 31, 35, 36),
         (5, 6, 7, 25, 26, 27, 29, 33), (9, 22, 25, 26, 27, 28, 30, 31)),
    ),
    "width3-59-w8": (
        59,
        ((25, 26, 46, 52, 55), (23, 24, 45, 54, 57), (17, 18, 25, 26, 45, 56, 57),
         (12, 40, 46, 47, 50, 53, 58), (11, 29, 30, 38, 47, 51, 52),
         (0, 12, 15, 40, 45, 56, 57), (3, 6, 21, 27, 48, 50, 58),
         (3, 8, 21, 32, 39, 40, 46), (7, 14, 30, 43, 46, 47, 55),
         (19, 20, 21, 22, 27, 28, 54, 56), (17, 18, 37, 38, 43, 44, 54, 57),
         (2, 17, 18, 20, 31, 32, 48, 52), (1, 4, 5, 8, 9, 13, 15, 40),
         (1, 8, 18, 32, 37, 38, 51, 56), (14, 17, 18, 27, 28, 43, 52, 55),
         (0, 4, 5, 13, 18, 32, 34, 39), (0, 1, 2, 16, 17, 19, 55, 56),
         (5, 6, 19, 20, 25, 27, 54, 55), (12, 35, 36, 39, 45, 52, 55, 56),
         (1, 4, 17, 24, 47, 50, 51, 58), (5, 10, 13, 16, 20, 22, 31, 37),
         (1, 3, 6, 7, 10, 13, 15, 40), (8, 31, 39, 40, 48, 51, 52, 53),
         (5, 13, 25, 35, 36, 42, 47, 58), (1, 2, 3, 7, 8, 14, 23, 42),
         (7, 12, 29, 39, 41, 42, 54, 58), (0, 17, 22, 27, 31, 33, 38, 44),
         (7, 13, 29, 42, 46, 48, 49, 53), (2, 7, 10, 19, 29, 36, 53, 57)),
        ((11, 29, 35, 36, 41, 49), (1, 3, 9, 21, 24, 45, 56), (4, 9, 16, 17, 18, 37, 44),
         (20, 24, 31, 38, 41, 51, 54), (14, 36, 41, 42, 43, 45, 49, 57),
         (14, 33, 42, 43, 44, 49, 50, 58), (7, 10, 23, 27, 31, 43, 48, 54),
         (1, 2, 9, 10, 11, 31, 33, 51), (5, 6, 11, 13, 31, 37, 38, 48),
         (7, 8, 12, 35, 40, 47, 51, 53), (1, 5, 8, 13, 19, 21, 50, 53),
         (17, 19, 22, 25, 36, 37, 51, 52), (22, 28, 30, 33, 37, 38, 43, 44),
         (20, 21, 27, 28, 32, 34, 37, 44), (11, 32, 36, 39, 48, 51, 53, 58),
         (19, 22, 31, 32, 39, 54, 56, 57), (4, 5, 19, 24, 35, 37, 56, 57),
         (7, 10, 14, 22, 28, 33, 41, 49), (9, 10, 11, 13, 29, 34, 50, 58),
         (19, 28, 34, 39, 41, 46, 53, 55), (4, 5, 22, 24, 44, 50, 54, 58),
         (3, 7, 11, 15, 40, 47, 53, 58), (3, 6, 11, 14, 21, 27, 30, 33),
         (19, 22, 30, 31, 35, 36, 52, 55), (25, 26, 27, 28, 30, 33, 47, 50),
         (0, 2, 18, 23, 30, 38, 43, 57), (0, 1, 32, 33, 40, 42, 48, 58),
         (11, 14, 23, 27, 44, 47, 54, 58), (6, 10, 16, 19, 22, 31, 48, 49)),
    ),
}


def depth_one_universal(label: str) -> tuple[BinaryMatrix, BinaryMatrix]:
    """A depth-one universal CSS code of arXiv:2610.06730, by database label.

    The paper pushes a seed code with a depth-one non-Clifford gate, tensored
    with a kernel code, through a CNOT circuit, so that depth-one layers of
    few-qubit gates on several qubit partitions generate a universal logical
    gate set.  ``"ghz37"`` is its ``[[37, 1, 7]]`` code ``lift:ghz37`` -- the
    15-qubit Reed-Muller seed, a kernel of four GHZ triples and ten ``|+>``
    qubits, and 22 coupling CNOTs (App. B.1) -- and ``"width3-59-w8"`` its
    ``[[59, 1, 9]]`` code ``lift:width3-59-w8``.  Both have checks of weight
    at most eight (Table 1).
    """

    if label not in _DEPTH_ONE_UNIVERSAL:
        raise ValueError(f"unknown depth-one universal code {label!r}")
    width, x_rows, z_rows = _DEPTH_ONE_UNIVERSAL[label]
    return _support_matrix(width, x_rows), _support_matrix(width, z_rows)


def apm_kasai(
    p: int,
    f_maps: Sequence[tuple[int, int]],
    g_maps: Sequence[tuple[int, int]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Affine-permutation-matrix Kasai-template CSS code (arXiv:2604.16209).

    ``f_maps`` and ``g_maps`` are six ``(a, b)`` pairs defining affine
    permutations ``x -> a x + b (mod p)`` (``gcd(a, p) = 1``), realized as
    ``p x p`` matrices with entry 1 at ``(a x + b mod p, x)``.  The code
    keeps ``J = 3`` active block rows of the ``6 x 12`` block-circulant
    parent: ``H_X`` block row ``r`` holds ``F_{(i-r) mod 6}`` on data block
    ``i`` and ``G_{(i-r) mod 6}`` on block ``6+i``; ``H_Z`` block row ``r``
    holds ``G_{(r-i) mod 6}^T`` and ``F_{(r-i) mod 6}^T``.  ``n = 12 p``.
    See arXiv:2604.16209, Appendix A, Table A1.
    """

    def affine(a: int, b: int) -> BinaryMatrix:
        matrix = np.zeros((p, p), dtype=np.uint8)
        for x in range(p):
            matrix[(a * x + b) % p, x] = 1
        return matrix

    f = [affine(a, b) for a, b in f_maps]
    g = [affine(a, b) for a, b in g_maps]
    h_x = np.zeros((3 * p, 12 * p), dtype=np.uint8)
    h_z = np.zeros((3 * p, 12 * p), dtype=np.uint8)
    for r in range(3):
        for i in range(6):
            h_x[r * p : (r + 1) * p, i * p : (i + 1) * p] = f[(i - r) % 6]
            h_x[r * p : (r + 1) * p, (6 + i) * p : (7 + i) * p] = g[(i - r) % 6]
            h_z[r * p : (r + 1) * p, i * p : (i + 1) * p] = g[(r - i) % 6].T
            h_z[r * p : (r + 1) * p, (6 + i) * p : (7 + i) * p] = f[(r - i) % 6].T
    return h_x, h_z


def cpm_pair_partition(
    lift: int,
    e_x: Sequence[Sequence[int]],
    e_z: Sequence[Sequence[int]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """CPM-based pair-partition CSS code (arXiv:2609.30069, Definition 1).

    ``e_x`` and ``e_z`` are ``J x L`` exponent arrays over ``Z_lift``; block
    ``(r, c)`` of ``H_X`` (``H_Z``) is the circulant permutation matrix
    ``C(s)`` whose row ``a`` has its one in column ``a - s``, so check
    ``(r, a)`` acts on qubit ``a - E[r, c]`` of block ``c`` (Eq. (78)).
    Orthogonality comes from the pair-partition condition, which is checked
    here rather than assumed.  ``n = L * lift``.
    """

    x = np.asarray(e_x, dtype=np.int64)
    z = np.asarray(e_z, dtype=np.int64)
    if x.ndim != 2 or x.shape != z.shape:
        raise ValueError("exponent arrays must share one J x L shape")

    def lift_array(exponents: np.ndarray) -> BinaryMatrix:
        return np.block([[circulant(lift, [-int(s)]) for s in row] for row in exponents]).astype(np.uint8)

    h_x, h_z = lift_array(x), lift_array(z)
    if ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any():
        raise ValueError("exponent arrays do not satisfy the pair-partition condition")
    return h_x, h_z


def cpm_pair_partition_f4(
    lift: int,
    e_x: Sequence[Sequence[int]],
    e_z: Sequence[Sequence[int]],
    c_x: Sequence[Sequence[int]],
    c_z: Sequence[Sequence[int]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Quaternary-coefficient CPM pair-partition CSS code (arXiv:2609.35601).

    ``e_x``/``e_z`` are ``J x L`` exponent arrays over ``Z_lift`` as in
    :func:`cpm_pair_partition`; ``c_x``/``c_z`` give the nonzero ``F_4``
    coefficient of each block as the exponent ``t`` of ``omega^t``
    (``omega^2 = omega + 1``).  Block ``(r, c)`` is ``C(s) (x) rho(omega^t)``
    with ``rho(omega) = [[0, 1], [1, 1]]`` the symmetric companion matrix of
    Eq. (18), so ``n = 2 * L * lift``.  Orthogonality is checked here rather
    than assumed.
    """

    arrays = [np.asarray(a, dtype=np.int64) for a in (e_x, e_z, c_x, c_z)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("exponent and coefficient arrays must share one J x L shape")
    omega = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    rho = [np.eye(2, dtype=np.uint8), omega, (omega @ omega) % 2]

    def lift_array(exponents: np.ndarray, coefficients: np.ndarray) -> BinaryMatrix:
        return np.block(
            [
                [np.kron(circulant(lift, [-int(s)]), rho[int(t) % 3]) for s, t in zip(row, coeffs)]
                for row, coeffs in zip(exponents, coefficients)
            ]
        ).astype(np.uint8)

    h_x, h_z = lift_array(arrays[0], arrays[2]), lift_array(arrays[1], arrays[3])
    if ((h_x.astype(np.int64) @ h_z.T.astype(np.int64)) % 2).any():
        raise ValueError("arrays do not satisfy the pair-partition conditions")
    return h_x, h_z


def permutation_group_table(
    generators: Sequence[Sequence[Sequence[int]]], degree: int
) -> tuple[list[list[int]], tuple[int, ...]]:
    """Cayley table of the permutation group generated by ``generators``.

    Each generator is given in 1-indexed disjoint-cycle notation on
    ``1..degree``.  Returns ``(table, generator_indices)``: ``table[g][h]`` is
    the index, in the sorted element list, of the product ``g h`` --
    permutations act from right to left, so ``(g h)(i) = g(h(i))`` -- and
    ``generator_indices[j]`` is the index of the ``j``-th generator.  The
    identity is index 0.
    """

    images = [_permutation_from_cycles(cycles, degree) for cycles in generators]
    elements = _permutation_closure(images, degree)
    index = {element: position for position, element in enumerate(elements)}
    table = [[index[tuple(g[h[point]] for point in range(degree))] for h in elements] for g in elements]
    return table, tuple(index[image] for image in images)


def two_block_group_algebra(
    table: Sequence[Sequence[int]], a_terms: Sequence[int], b_terms: Sequence[int]
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Two-block group-algebra CSS code over ``F_2[G]`` (arXiv:2609.36213, Sec. 2).

    ``table`` is the Cayley table of ``G`` (``table[g][h]`` is the index of
    ``g h``); ``a_terms`` and ``b_terms`` list the group elements, as
    indices, of the seeds ``a`` and ``b``.  ``A = rho(a)`` is right
    multiplication (``rho(z) e_h = sum_g z_g e_{hg}``) and ``B = lambda(b)``
    is left multiplication (``lambda(z) e_h = sum_g z_g e_{gh}``), Eq. (2),
    so ``A`` and ``B`` commute by associativity for any group, abelian or
    not; ``H_X = [A | B]`` and ``H_Z = [B^T | A^T]`` (Eq. (1)).
    ``n = 2 |G|``.  Commutation is checked here rather than assumed.
    """

    cayley = np.asarray(table, dtype=np.int64)
    if cayley.ndim != 2 or cayley.shape[0] != cayley.shape[1]:
        raise ValueError("table must be a square Cayley table")
    order = cayley.shape[0]
    a = np.zeros((order, order), dtype=np.uint8)
    b = np.zeros((order, order), dtype=np.uint8)
    for g in a_terms:
        for h in range(order):
            a[cayley[h, g], h] ^= 1
    for g in b_terms:
        for h in range(order):
            b[cayley[g, h], h] ^= 1
    a64, b64 = a.astype(np.int64), b.astype(np.int64)
    if ((a64 @ b64 + b64 @ a64) % 2).any():
        raise ValueError("left and right multiplications do not commute: table is not a group")
    h_x = np.concatenate([a, b], axis=1).astype(np.uint8)
    h_z = np.concatenate([b.T, a.T], axis=1).astype(np.uint8)
    return h_x, h_z


def weighted_shift_c3_code() -> tuple[BinaryMatrix, BinaryMatrix]:
    """The [[18,4,3]] weighted-shift BBGA code of arXiv:2609.36213, Sec. 5.

    Over ``F_2[C_3]`` with ``c = r + r^2`` (multiplication matrix ``C``, the
    complement of ``I_3``), the seeds ``a = (e, e, c)``, ``b = (c)`` of
    Eq. (18) lift to the binary weighted shifts ``P = S_3(e, e, c)`` and
    ``Q = lambda(c)`` of Eq. (19); ``A = P^2 + Q`` and
    ``B = P^2 + P Q + P^2 Q`` (Eq. (20)) then enter the two-block form
    ``H_X = [A | B]``, ``H_Z = [B^T | A^T]`` of Eq. (1).
    """

    c = (np.ones((3, 3), dtype=np.int64) - np.eye(3, dtype=np.int64)) % 2
    identity = np.eye(3, dtype=np.int64)
    zero = np.zeros((3, 3), dtype=np.int64)
    p = np.block([[zero, identity, zero], [zero, zero, identity], [c, zero, zero]])
    q = np.block([[c, zero, zero], [zero, c, zero], [zero, zero, c]])
    p2 = (p @ p) % 2
    a = (p2 + q) % 2
    b = (p2 + p @ q + p2 @ q) % 2
    h_x = np.concatenate([a, b], axis=1).astype(np.uint8)
    h_z = np.concatenate([b.T, a.T], axis=1).astype(np.uint8)
    return h_x, h_z


def cornucopia(
    q: int, a_shifts: Sequence[int], b_shifts: Sequence[int]
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Cornucopia block-convolutional code (arXiv:2608.02773).

    Each of twelve data blocks holds ``P = 3q`` qubits indexed by
    ``(x, y)`` in ``Z_3 x Z_q``, flattened as ``v = x q + y``.  The block
    permutations are ``A_1, B_3: (x, y) -> (2x+2, y+s)`` (row-inverting),
    ``A_0, B_2: (x, y) -> (x+1, y+s)`` (row-shifting), and pure column
    translations otherwise, with per-matrix column shifts ``s`` given by
    ``a_shifts``/``b_shifts``.  ``H_X`` block ``(r, j)`` holds
    ``A_{(j-r) mod 6}`` on ``L_j`` and ``B_{(j-r) mod 6}`` on ``R_j``;
    ``H_Z`` holds ``B_{(r-j) mod 6}^T`` and ``A_{(r-j) mod 6}^T``.
    ``n = 12 P``.  See arXiv:2608.02773, Methods and Extended Data Tab. 1.
    """

    p = 3 * q

    def perm(kind: str, s: int) -> BinaryMatrix:
        matrix = np.zeros((p, p), dtype=np.uint8)
        for x in range(3):
            new_x = {"invert": (2 * x + 2) % 3, "shift": (x + 1) % 3}.get(kind, x)
            for y in range(q):
                matrix[new_x * q + (y + s) % q, x * q + y] = 1
        return matrix

    a_kinds = ["shift", "invert", "column", "column", "column", "column"]
    b_kinds = ["column", "column", "shift", "invert", "column", "column"]
    a = [perm(kind, s) for kind, s in zip(a_kinds, a_shifts)]
    b = [perm(kind, s) for kind, s in zip(b_kinds, b_shifts)]
    h_x = np.zeros((3 * p, 12 * p), dtype=np.uint8)
    h_z = np.zeros((3 * p, 12 * p), dtype=np.uint8)
    for r in range(3):
        for j in range(6):
            h_x[r * p : (r + 1) * p, j * p : (j + 1) * p] = a[(j - r) % 6]
            h_x[r * p : (r + 1) * p, (6 + j) * p : (7 + j) * p] = b[(j - r) % 6]
            h_z[r * p : (r + 1) * p, j * p : (j + 1) * p] = b[(r - j) % 6].T
            h_z[r * p : (r + 1) * p, (6 + j) * p : (7 + j) * p] = a[(r - j) % 6].T
    return h_x, h_z


def gala_abelian(
    moduli: Sequence[int],
    rungs: int,
    active: int,
    f_terms: Sequence[Sequence[Sequence[int]]],
    g_terms: Sequence[Sequence[Sequence[int]]],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """GALA group-action-lift CSS code over an abelian lift group.

    The GALA construction of arXiv:2608.07431 is the Kasai template of
    Definition 11 with the lift taken in the group ring of a product group.
    ``moduli`` gives the cyclic factors ``C_{m_1} x ... x C_{m_r}``, whose
    regular representation is the Kronecker product of the individual shift
    matrices, so each group element is a ``P x P`` permutation with
    ``P = prod(moduli)``.  ``f_terms[i]`` lists the exponent tuples summed to
    form the group-ring element ``F_i`` (one tuple per element of ``moduli``);
    a length-two list is the paper's ``x^a + x^b`` polynomial lift.

    With ``L = rungs`` and ``J = active``, the parent matrices are the
    ``L/2 x L/2`` block circulants ``[H_X]_{i,j} = F_{j-i}``,
    ``[H_X]_{i,j+L/2} = G_{j-i}`` and ``[H_Z]_{i,j} = G^T_{i-j}``,
    ``[H_Z]_{i,j+L/2} = F^T_{i-j}``, of which only the first ``J`` block rows
    are kept.  ``n = L P``.  Abelian lifts commute, so every ``Psi_r`` of
    Eq. (S16) vanishes and orthogonality is automatic -- the non-abelian
    ``H_k`` factors of the paper's other instances exist to break exactly
    that on the latent rows, and are not covered here.  See arXiv:2608.07431,
    Tables S3 and S5.
    """

    if rungs < 2 or rungs % 2 != 0:
        raise ValueError("rungs (L) must be an even integer of at least 2")
    half = rungs // 2
    if not 1 <= active <= half:
        raise ValueError("active (J) must satisfy 1 <= J <= L/2")
    if len(f_terms) != half or len(g_terms) != half:
        raise ValueError(f"f_terms and g_terms must each hold L/2 = {half} entries")
    size = 1
    for modulus in moduli:
        size *= modulus

    def element(shifts: Sequence[int]) -> BinaryMatrix:
        if len(shifts) != len(moduli):
            raise ValueError("each exponent tuple needs one entry per cyclic factor")
        matrix = np.ones((1, 1), dtype=np.uint8)
        for modulus, shift in zip(moduli, shifts):
            matrix = np.kron(matrix, circulant(modulus, [shift]))
        return matrix

    def lift(terms: Sequence[Sequence[int]]) -> BinaryMatrix:
        total = np.zeros((size, size), dtype=np.uint8)
        for shifts in terms:
            total ^= element(shifts)
        return total

    f = [lift(terms) for terms in f_terms]
    g = [lift(terms) for terms in g_terms]
    h_x = np.zeros((active * size, rungs * size), dtype=np.uint8)
    h_z = np.zeros((active * size, rungs * size), dtype=np.uint8)
    for r in range(active):
        for j in range(half):
            rows = slice(r * size, (r + 1) * size)
            left = slice(j * size, (j + 1) * size)
            right = slice((half + j) * size, (half + j + 1) * size)
            h_x[rows, left] = f[(j - r) % half]
            h_x[rows, right] = g[(j - r) % half]
            h_z[rows, left] = g[(r - j) % half].T
            h_z[rows, right] = f[(r - j) % half].T
    return h_x, h_z


# The three local codes that arXiv:2608.12509 writes out explicitly, each
# exactly as printed (Appendix A.1 gives generators for [7,3,4] and [9,5,3],
# A.2 gives a parity check for [6,3,3]).  Tables 1 and 3 name local codes only
# by their [n, k, d] label, so only instances built from these three can be
# rebuilt exactly.
_QT_LOCAL_SOURCE: dict[str, tuple[str, list[list[int]]]] = {
    "633": (
        "check",
        [[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]],
    ),
    "734": (
        "generator",
        [[1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]],
    ),
    "953": (
        "generator",
        [
            [1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 1, 1],
        ],
    ),
}


def qt_local_code(label: str) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Check and generator matrices of a local code of arXiv:2608.12509.

    Whichever of the two the paper prints is returned verbatim; the other is
    a dual basis.  The lifted code depends only on the two row spaces, but
    keeping the published matrix reproduces the authors' own ``H_X``,
    ``H_Z`` row for row.
    """

    from ..utils.gf2 import nullspace

    kind, rows = _QT_LOCAL_SOURCE[label]
    matrix = np.asarray(rows, dtype=np.uint8)
    dual = np.asarray(nullspace(matrix), dtype=np.uint8)
    return (matrix, dual) if kind == "check" else (dual, matrix)


def _permutation_from_cycles(cycles: Sequence[Sequence[int]], degree: int) -> tuple[int, ...]:
    """Convert 1-indexed disjoint cycle notation into an image tuple."""

    image = list(range(degree))
    for cycle in cycles:
        points = [point - 1 for point in cycle]
        if any(not 0 <= point < degree for point in points):
            raise ValueError(f"cycle {cycle} moves a point outside 1..{degree}")
        for position, point in enumerate(points):
            image[point] = points[(position + 1) % len(points)]
    return tuple(image)


def _permutation_closure(
    generators: Sequence[tuple[int, ...]], degree: int
) -> list[tuple[int, ...]]:
    """The subgroup of ``S_degree`` generated by ``generators``, sorted."""

    def compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(left[point] for point in right)

    identity = tuple(range(degree))
    elements = {identity}
    frontier = [identity]
    while frontier:
        grown = []
        for element in frontier:
            for generator in generators:
                product = compose(generator, element)
                if product not in elements:
                    elements.add(product)
                    grown.append(product)
        frontier = grown
    return sorted(elements)


def quantum_tanner_lift(
    degree: int,
    multiset_a: Sequence[Sequence[Sequence[int]]],
    multiset_b: Sequence[Sequence[Sequence[int]]],
    local_a: tuple[BinaryMatrix, BinaryMatrix],
    local_b: tuple[BinaryMatrix, BinaryMatrix],
    pi_a: Sequence[int],
    pi_b: Sequence[int],
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """Lifted quantum Tanner code of Leverrier-Zemor type (arXiv:2608.12509).

    The group ``G`` is the subgroup of ``S_degree`` generated by the two
    multisets, each element written in 1-indexed disjoint cycle notation
    (``[]`` is the identity), exactly as tabulated in the paper.  Qubits are
    triples ``(i, j, g)`` in ``[n_A] x [n_B] x G``, so ``n = n_A n_B |G|``.

    ``local_a`` and ``local_b`` are ``(check, generator)`` pairs as returned
    by :func:`qt_local_code`.  With ``H_0, G_0`` and their column-permuted
    copies ``H_1 = H_0[:, pi_a]``, ``G_1 = G_0[:, pi_a]`` on the ``A`` side,
    and ``H_0', G_0', H_1' = H_0'[:, pi_b]``, ``G_1' = G_0'[:, pi_b]`` on the
    ``B`` side, Eq. (34) of the paper reads

    ``H_X = [H_0 (x) G_0' (x) I ; (H_1 (x) G_1' (x) I) L_A R_B]``,
    ``H_Z = [(G_0 (x) H_1' (x) I) R_B ; (G_1 (x) H_0' (x) I) L_A]``,

    where ``L_A`` permutes the fibre of ``(i, j, .)`` by left multiplication
    with ``a_i`` and ``R_B`` by right multiplication with ``b_j``.  Both
    ``pi_a`` and ``pi_b`` are 1-indexed column permutations.

    The code is independent of the dual bases chosen: changing basis
    multiplies a block of ``H_X`` or ``H_Z`` on the left by an invertible
    matrix, leaving its row space unchanged.
    """

    elements_a = [_permutation_from_cycles(cycles, degree) for cycles in multiset_a]
    elements_b = [_permutation_from_cycles(cycles, degree) for cycles in multiset_b]
    group = _permutation_closure(elements_a + elements_b, degree)
    index = {element: position for position, element in enumerate(group)}
    order = len(group)
    n_a, n_b = len(elements_a), len(elements_b)

    def permute_columns(matrix: BinaryMatrix, permutation: Sequence[int]) -> BinaryMatrix:
        columns = [entry - 1 for entry in permutation]
        if sorted(columns) != list(range(matrix.shape[1])):
            raise ValueError("column permutation does not match the local code length")
        return matrix[:, columns]

    h_0, g_0 = (np.asarray(matrix, dtype=np.uint8) for matrix in local_a)
    h_0p, g_0p = (np.asarray(matrix, dtype=np.uint8) for matrix in local_b)
    if h_0.shape[1] != n_a or h_0p.shape[1] != n_b:
        raise ValueError("local code lengths must match the multiset sizes")
    h_1 = permute_columns(h_0, pi_a)
    h_1p = permute_columns(h_0p, pi_b)
    g_1 = permute_columns(g_0, pi_a)
    g_1p = permute_columns(g_0p, pi_b)

    identity = np.eye(order, dtype=np.uint8)

    def lift(left: BinaryMatrix, right: BinaryMatrix) -> BinaryMatrix:
        return np.kron(np.kron(left, right), identity).astype(np.uint8)

    def fibre_permutation(elements: Sequence[tuple[int, ...]], on_left: bool) -> np.ndarray:
        """Column destination of every qubit under ``L_A`` or ``R_B``."""

        destination = np.empty(n_a * n_b * order, dtype=np.int64)
        for i in range(n_a):
            for j in range(n_b):
                shift = elements[i] if on_left else elements[j]
                base = (i * n_b + j) * order
                for position, element in enumerate(group):
                    moved = (
                        tuple(shift[point] for point in element)
                        if on_left
                        else tuple(element[point] for point in shift)
                    )
                    destination[base + position] = base + index[moved]
        return destination

    left_action = fibre_permutation(elements_a, on_left=True)
    right_action = fibre_permutation(elements_b, on_left=False)

    def act(matrix: BinaryMatrix, destination: np.ndarray) -> BinaryMatrix:
        moved = np.zeros_like(matrix)
        moved[:, destination] = matrix
        return moved

    h_x = np.vstack([lift(h_0, g_0p), act(act(lift(h_1, g_1p), left_action), right_action)])
    h_z = np.vstack([act(lift(g_0, h_1p), right_action), act(lift(g_1, h_0p), left_action)])
    return h_x, h_z


def _gf2e_multiplication_matrices(extension: int, modulus: int) -> list[BinaryMatrix]:
    """Matrices of multiplication by ``alpha^m`` on ``F_2^e``.

    ``modulus`` is the primitive polynomial bitmask (bit ``i`` is the
    coefficient of ``x^i``).  The returned list has ``2^e - 1`` entries and
    satisfies ``M[a] @ M[b] = M[(a + b) mod (2^e - 1)]``.
    """

    span = (1 << extension) - 1
    alpha = np.zeros((extension, extension), dtype=np.uint8)
    tail = [(modulus >> index) & 1 for index in range(extension)]
    for column in range(extension - 1):
        alpha[column + 1, column] = 1
    # x * x^(e-1) = x^e reduces to the modulus tail.
    for row in range(extension):
        alpha[row, extension - 1] = tail[row]
    powers = [np.eye(extension, dtype=np.uint8)]
    for _ in range(span - 1):
        powers.append((alpha @ powers[-1]) & 1)
    return powers


def kasai_nonbinary(
    width: int,
    lift: int,
    *,
    extension: int = 8,
    modulus: int | None = None,
    seed: int = 101,
) -> tuple[BinaryMatrix, BinaryMatrix]:
    """A full Kasai-style non-binary quasi-cyclic CSS code, binary-expanded.

    Follows Komoto and Kasai, arXiv:2412.21171: the binary orthogonal pair of
    :func:`kasai_binary_pair` is lifted to ``GF(2^extension)`` labels via the
    canonical separable assignment of arXiv:2510.25583
    (``gamma[i,j] = alpha^(A_i + C_j)``, ``delta[i,j] = alpha^(B_i - C_j)``,
    which preserves orthogonality because binary row overlaps are even), then
    expanded to binary through multiplication matrices, transposed on the Z
    side.  ``n = extension * width * lift``.

    Separable labels are diagonal row and column scalings, so the GF(2^e)
    rank equals the binary rank ``2 * lift - 1`` and
    ``k = extension * ((width - 4) * lift + 2)``.  The published instances
    instead draw a random solution of the label congruence system, which
    generically reaches full rank and the slightly smaller
    ``k = extension * (width - 4) * lift``.
    """

    if extension < 2:
        raise ValueError("extension must be at least 2")
    if modulus is None:
        defaults = {2: 0b111, 3: 0b1011, 4: 0b10011, 8: 0b100011101}
        if extension not in defaults:
            raise ValueError(f"no default primitive polynomial for extension {extension}")
        modulus = defaults[extension]
    base_x, base_z = kasai_binary_pair(width, lift)
    span = (1 << extension) - 1
    powers = _gf2e_multiplication_matrices(extension, modulus)
    rng = np.random.default_rng(seed)
    row_x_labels = rng.integers(0, span, size=base_x.shape[0])
    row_z_labels = rng.integers(0, span, size=base_z.shape[0])
    column_labels = rng.integers(0, span, size=base_x.shape[1])

    rows, columns = base_x.shape
    h_x = np.zeros((rows * extension, columns * extension), dtype=np.uint8)
    h_z = np.zeros((rows * extension, columns * extension), dtype=np.uint8)
    for i, j in zip(*np.nonzero(base_x)):
        exponent = int(row_x_labels[i] + column_labels[j]) % span
        h_x[i * extension : (i + 1) * extension, j * extension : (j + 1) * extension] = powers[
            exponent
        ]
    for i, j in zip(*np.nonzero(base_z)):
        exponent = int(row_z_labels[i] - column_labels[j]) % span
        h_z[i * extension : (i + 1) * extension, j * extension : (j + 1) * extension] = powers[
            exponent
        ].T
    return h_x, h_z


__all__ = [
    "apm_kasai",
    "bipartite_grid",
    "bivariate_bicycle",
    "bivariate_monomial_sum",
    "circulant",
    "cornucopia",
    "cpm_pair_partition",
    "cpm_pair_partition_f4",
    "cyclic_shift",
    "cyclic_triorthogonal",
    "depth_one_universal",
    "doubled_color_41",
    "gala_abelian",
    "generalized_bicycle",
    "hamming_7_4",
    "helix_code",
    "helper_qss_css",
    "hypergraph_product",
    "iceberg",
    "kasai_binary_pair",
    "kasai_nonbinary",
    "la_cross",
    "lifted_product_b1",
    "lifted_product_monomial",
    "middle_reed_muller",
    "permutation_group_table",
    "qt_local_code",
    "quantum_reed_muller_15",
    "quantum_reed_muller_31",
    "quantum_tanner_lift",
    "reed_muller_generator",
    "repetition_ring",
    "self_dual_bicycle",
    "steane_code",
    "subset_inclusion",
    "surface_code",
    "symplectic_double",
    "toric_code",
    "trivariate_tricycle",
    "twisted_torus_translation",
    "two_block_group_algebra",
    "weighted_shift_c3_code",
]
