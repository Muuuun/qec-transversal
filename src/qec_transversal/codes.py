"""Constructors for standard CSS and quantum LDPC code families.

Every constructor returns a pair ``(h_x, h_z)`` of binary ``uint8`` matrices
satisfying ``H_X H_Z^T = 0`` over GF(2), suitable for :class:`~.css.CSSCode`.

Monomial conventions follow arXiv:2308.07915: with cyclic shift matrices
``S_l`` and ``S_m``, set ``x = S_l (x) I_m`` and ``y = I_l (x) S_m``.  A
bivariate polynomial is given as an iterable of ``(i, j)`` exponent pairs
meaning ``x^i y^j``; a univariate polynomial as an iterable of integers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .gf2 import BinaryMatrix


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

    from .gf2 import nullspace

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


@dataclass(frozen=True)
class NamedCode:
    """A registry entry: a constructor plus its published parameters."""

    name: str
    family: str
    build: Callable[[], tuple[BinaryMatrix, BinaryMatrix]]
    n: int
    k: int
    d: int | None = None
    d_is_upper_bound: bool = False
    source: str = ""


def _bb(l: int, m: int, a: list, b: list) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: bivariate_bicycle(l, m, a, b)


def _sdbb(
    monomials: list, basis_1: Sequence[int], basis_2: Sequence[int]
) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: self_dual_bicycle(monomials, basis_1, basis_2)


def _qt(
    degree: int,
    multiset_a: list,
    multiset_b: list,
    label_a: str,
    label_b: str,
    pi_a: Sequence[int],
    pi_b: Sequence[int],
) -> Callable[[], tuple[BinaryMatrix, BinaryMatrix]]:
    return lambda: quantum_tanner_lift(
        degree,
        multiset_a,
        multiset_b,
        qt_local_code(label_a),
        qt_local_code(label_b),
        pi_a,
        pi_b,
    )


REGISTRY: dict[str, NamedCode] = {
    code.name: code
    for code in [
        # Positive controls: codes with known nontrivial strict-transversal gates.
        NamedCode("steane", "color", steane_code, 7, 1, 3, source="self-dual doubly-even"),
        NamedCode(
            "c4-22", "small", lambda: (np.ones((1, 4), np.uint8), np.ones((1, 4), np.uint8)),
            4, 2, 2, source="[[4,2,2]] iceberg code",
        ),
        NamedCode(
            "c6-22", "small",
            lambda: (
                np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]], np.uint8),
                np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]], np.uint8),
            ),
            6, 2, 2, source="Albert running example",
        ),
        NamedCode(
            "cube-832", "small",
            lambda: (
                np.ones((1, 8), np.uint8),
                np.asarray(
                    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0]],
                    np.uint8,
                ),
            ),
            8, 3, 2, source="[[8,3,2]] cube code, transversal CCZ",
        ),
        NamedCode("iceberg-8", "iceberg", lambda: iceberg(4), 8, 6, 2, source="[[2m,2m-2,2]] error-detection family"),
        NamedCode("iceberg-12", "iceberg", lambda: iceberg(6), 12, 10, 2, source="[[2m,2m-2,2]] error-detection family"),
        NamedCode("qrm15", "reed-muller", quantum_reed_muller_15, 15, 1, 3, source="arXiv:1403.2734"),
        NamedCode("qrm31", "reed-muller", quantum_reed_muller_31, 31, 1, 3, source="punctured RM(1,5); level-4 transversal gate"),
        NamedCode("tesseract", "reed-muller", lambda: middle_reed_muller(4), 16, 6, 4, source="arXiv:2608.05688"),
        NamedCode("rm64", "reed-muller", lambda: middle_reed_muller(6), 64, 20, 8, source="arXiv:2608.05688"),
        NamedCode("rm256", "reed-muller", lambda: middle_reed_muller(8), 256, 70, 16, source="arXiv:2608.05688"),
        NamedCode("grid-4x6", "grid", lambda: bipartite_grid(4, 6), 24, 8, 4, source="arXiv:2608.05688"),
        NamedCode("grid-6x8", "grid", lambda: bipartite_grid(6, 8), 48, 24, 4, source="arXiv:2608.05688"),
        # Topological negative controls.
        NamedCode("toric-4", "toric", lambda: toric_code(4), 32, 2, 4),
        NamedCode("toric-10", "toric", lambda: toric_code(10), 200, 2, 10),
        NamedCode("surface-5", "surface", lambda: surface_code(5), 41, 1, 5),
        # Bivariate bicycle codes, Bravyi et al. arXiv:2308.07915 Table 3.
        NamedCode("bb72", "bivariate-bicycle", _bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 72, 12, 6, source="arXiv:2308.07915"),
        NamedCode("bb90", "bivariate-bicycle", _bb(15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)]), 90, 8, 10, source="arXiv:2308.07915"),
        NamedCode("bb108", "bivariate-bicycle", _bb(9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 108, 8, 10, source="arXiv:2308.07915"),
        NamedCode("gross", "bivariate-bicycle", _bb(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 144, 12, 12, source="arXiv:2308.07915"),
        NamedCode("two-gross", "bivariate-bicycle", _bb(12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]), 288, 12, 18, source="arXiv:2308.07915"),
        NamedCode("bb360", "bivariate-bicycle", _bb(30, 6, [(9, 0), (0, 1), (0, 2)], [(0, 3), (25, 0), (26, 0)]), 360, 12, 24, d_is_upper_bound=True, source="arXiv:2308.07915"),
        NamedCode("bb756", "bivariate-bicycle", _bb(21, 18, [(3, 0), (0, 10), (0, 17)], [(0, 5), (3, 0), (19, 0)]), 756, 16, 34, d_is_upper_bound=True, source="arXiv:2308.07915"),
        NamedCode("bb54", "bivariate-bicycle", _bb(3, 9, [(0, 0), (0, 2), (0, 4)], [(0, 3), (1, 0), (2, 0)]), 54, 8, 6, source="arXiv:2408.10001"),
        # Symmetric BB codes with rich fold-transversal groups.
        NamedCode("bb98-symmetric", "bivariate-bicycle", _bb(7, 7, [(1, 0), (0, 3), (0, 4)], [(0, 1), (3, 0), (4, 0)]), 98, 6, 12, source="arXiv:2407.03973"),
        NamedCode("bb162-symmetric", "bivariate-bicycle", _bb(9, 9, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]), 162, 8, 12, source="arXiv:2407.03973"),
        # Coprime bivariate bicycle codes, Wang-Mueller arXiv:2408.10001.
        NamedCode("coprime30", "coprime-bb", _bb(3, 5, [(0, 0), (1, 1), (2, 2)], [(0, 0), (2, 2), (1, 2)]), 30, 4, 6, source="arXiv:2408.10001"),
        NamedCode("coprime42", "coprime-bb", _bb(3, 7, [(0, 0), (2, 2), (0, 3)], [(0, 0), (2, 2), (1, 3)]), 42, 6, 6, source="arXiv:2408.10001"),
        NamedCode("coprime70", "coprime-bb", _bb(5, 7, [(0, 0), (1, 1), (0, 5)], [(0, 0), (1, 1), (2, 5)]), 70, 6, 8, source="arXiv:2408.10001"),
        NamedCode("coprime126", "coprime-bb", _bb(7, 9, [(0, 0), (1, 1), (2, 4)], [(0, 0), (6, 4), (6, 5)]), 126, 12, 10, source="arXiv:2408.10001"),
        NamedCode("coprime154", "coprime-bb", _bb(7, 11, [(0, 0), (1, 1), (3, 9)], [(0, 0), (5, 8), (4, 9)]), 154, 6, 16, source="arXiv:2408.10001"),
        # Trivariate bicycle, arXiv:2406.19151 (weight-5 checks).
        NamedCode("trivariate30", "trivariate-bicycle", _bb(3, 5, [(1, 0), (1, 4)], [(1, 0), (0, 2), (2, 2)]), 30, 4, 5, source="arXiv:2406.19151"),
        # Generalized bicycle codes, Panteleev-Kalachev arXiv:1904.02703 App. B.
        NamedCode("gb48", "generalized-bicycle", lambda: generalized_bicycle(24, [0, 2, 8, 15], [0, 2, 12, 17]), 48, 6, 8, source="arXiv:1904.02703 A3"),
        NamedCode("gb46", "generalized-bicycle", lambda: generalized_bicycle(23, [0, 5, 8, 12], [0, 1, 5, 7]), 46, 2, 9, source="arXiv:1904.02703 A4"),
        NamedCode("gb126", "generalized-bicycle", lambda: generalized_bicycle(63, [0, 1, 14, 16, 22], [0, 3, 13, 20, 42]), 126, 28, 8, source="arXiv:1904.02703 A2"),
        NamedCode("gb66-2608.09115", "generalized-bicycle", lambda: generalized_bicycle(33, [1, 2, 3, 5, 10, 27, 32], [0, 1, 3, 4, 5, 7, 13, 15, 22, 24, 30, 32]), 66, 20, 7, source="arXiv:2608.09115 Table I"),
        NamedCode("gb46-2608.09115", "generalized-bicycle", lambda: generalized_bicycle(23, [0, 1, 12, 13, 17, 19], [0, 1, 4, 5, 8, 9, 12, 13]), 46, 2, 8, source="arXiv:2608.09115 Table I"),
        # Multi-agent search bicycle codes, Qian-Li arXiv:2608.08996 (SI construction data).
        # Abelian coset-orbit balanced products: bb288 is the paper's [[288,16,18]] over
        # Z12 x Z48 with normal K = <y^12>, reduced (as the paper states) to its lifted
        # product over Z12 x Z12; the other two have K = {e} and are two-block codes as given.
        NamedCode("bb288-2608.08996", "bivariate-bicycle", _bb(12, 12, [(1, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0), (5, 9)]), 288, 16, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("bb234-2608.08996", "bivariate-bicycle", _bb(13, 9, [(0, 0), (0, 2), (0, 8), (4, 4), (6, 8)], [(2, 8), (5, 4), (10, 2), (11, 5), (12, 0)]), 234, 28, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("bb372-2608.08996", "bivariate-bicycle", _bb(31, 6, [(5, 1), (5, 3), (7, 2), (18, 2), (30, 2)], [(10, 1), (10, 3), (21, 5), (24, 5), (26, 5)]), 372, 44, 18, source="arXiv:2608.08996 Table 1"),
        NamedCode("wzl20-2608.10912", "subset-inclusion", lambda: subset_inclusion(6, 3, 2), 20, 8, 4, source="arXiv:2608.10912 Sec. IV"),
        NamedCode("wzl120-2608.10912", "subset-inclusion", lambda: subset_inclusion(10, 3, 2), 120, 100, 4, source="arXiv:2608.10912 Sec. IV"),
        NamedCode("doubled41-2608.11160", "doubled", doubled_color_41, 41, 1, 9, source="arXiv:2608.11160 Ex. III.5"),
        # Self-dual bivariate bicycle codes: sparse (weight-8, doubly even) codes
        # that DO carry strict transversal H and S -- the registry's positive LDPC
        # control against reading the qLDPC census as a statement about sparsity.
        # f(x, y) per Eq. (14) on the twisted torus of Tables I-II; distances are
        # the paper's exact integer-programming values.
        NamedCode("sdbb16-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 4), (2, 2)), 16, 4, 4, source="arXiv:2510.05211 Table I"),
        NamedCode("sdbb64-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (0, 1), (0, -1)], (0, 8), (4, 4)), 64, 8, 8, source="arXiv:2510.05211 Table I, Fig. 1"),
        NamedCode("sdbb152-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (2, 1), (-1, 1)], (0, 19), (4, 6)), 152, 6, 16, source="arXiv:2510.05211 Table II"),
        NamedCode("sdbb160-2510.05211", "self-dual-bb", _sdbb([(0, 0), (1, 0), (2, 2), (-1, 1)], (0, 10), (8, 0)), 160, 8, 16, source="arXiv:2510.05211 Table II"),
        NamedCode("apm1152-2604.16209", "apm-kasai", lambda: apm_kasai(96, [(5, 41), (85, 77), (73, 66), (1, 0), (1, 72), (37, 9)], [(61, 15), (1, 24), (89, 62), (25, 22), (85, 93), (25, 78)]), 1152, 580, 12, d_is_upper_bound=True, source="arXiv:2604.16209 Table A1"),
        NamedCode("apm2304-2604.16209", "apm-kasai", lambda: apm_kasai(192, [(71, 127), (97, 80), (67, 117), (163, 165), (25, 60), (187, 33)], [(163, 165), (55, 183), (167, 79), (139, 41), (109, 78), (31, 27)]), 2304, 1156, 14, d_is_upper_bound=True, source="arXiv:2604.16209 Table A1"),
        NamedCode("cornucopia252-2608.02773", "cornucopia", lambda: cornucopia(7, [2, 1, 1, 1, 4, 5], [5, 3, 0, 5, 2, 3]), 252, 130, 6, source="arXiv:2608.02773 Ext. Tab. 1"),
        NamedCode("cornucopia1044-2608.02773", "cornucopia", lambda: cornucopia(29, [2, 22, 20, 22, 18, 6], [27, 11, 12, 18, 21, 26]), 1044, 526, 12, source="arXiv:2608.02773 Ext. Tab. 1"),
        NamedCode("cornucopia2844-2608.02773", "cornucopia", lambda: cornucopia(79, [6, 49, 55, 18, 40, 7], [24, 41, 78, 53, 68, 21]), 2844, 1426, 18, source="arXiv:2608.02773 Ext. Tab. 1"),
        # Lifted quantum Tanner codes, Mian et al. arXiv:2608.12509.  Only the
        # instances whose two local codes are among the three the paper prints
        # in full are rebuildable; the [8,4,4] and [7,4,3] rows are named by
        # parameters only, so their column order -- which the lift depends on --
        # is not published.  d is min(d_X, d_Z) of the paper's sQetch bounds.
        # qt504 and qt756 additionally reproduce the rank and row-weight
        # multiset of the authors' own H_X, H_Z (Zenodo 10.5281/zenodo.21904804).
        NamedCode("qt720-2608.12509", "quantum-tanner", _qt(5, [[], [[2, 4, 5, 3]], [[1, 5, 4, 3, 2]], [[1, 5, 2, 3]], [[1, 3, 4, 2]], [[1, 2, 3, 4, 5]]], [[], [[2, 4, 5, 3]], [[1, 5, 4, 3, 2]], [[1, 5, 2, 3]], [[1, 3, 4, 2]], [[1, 2, 3, 4, 5]]], "633", "633", [1, 2, 3, 4, 5, 6], [1, 2, 5, 4, 6, 3]), 720, 6, 30, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 3, 10, 11"),
        NamedCode("qt504-2608.12509", "quantum-tanner", _qt(7, [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]]], [[], [[1, 3], [2, 4], [5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]], [[1, 4, 3, 2], [5, 6]], [[1, 2, 3, 4], [6, 7]]], "734", "633", [1, 2, 3, 4, 5, 6, 7], [1, 2, 6, 3, 4, 5]), 504, 4, 12, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 1, 9, 12"),
        NamedCode("qt756-2608.12509", "quantum-tanner", _qt(7, [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]]], [[], [], [[5, 6, 7]], [[5, 6, 7]], [[1, 4, 3, 2], [6, 7]], [[1, 4, 3, 2], [5, 7]], [[1, 4, 3, 2], [5, 6]], [[1, 2, 3, 4], [6, 7]], [[1, 2, 3, 4], [5, 7]]], "734", "953", [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7, 8, 9, 6]), 756, 10, 9, d_is_upper_bound=True, source="arXiv:2608.12509 Tables 1, 9, 12"),
        # Hypergraph and lifted products.
        NamedCode("hgp-hamming", "hypergraph-product", lambda: hypergraph_product(hamming_7_4(), hamming_7_4()), 58, 16, 3, source="arXiv:0903.0566"),
        NamedCode("lacross65", "la-cross", lambda: la_cross(7, 3), 65, 9, 4, source="arXiv:2404.13010"),
        NamedCode("lacross400", "la-cross", lambda: la_cross(16, 4), 400, 16, 8, source="arXiv:2404.13010"),
        NamedCode("lifted-b1", "lifted-product", lifted_product_b1, 882, 24, 24, d_is_upper_bound=True, source="arXiv:1904.02703 B1"),
        # Kasai-style quasi-cyclic CSS codes.
        NamedCode("kasai-binary-294", "kasai", lambda: kasai_binary_pair(6, 49), 294, 100, None, source="arXiv:2501.13444"),
        NamedCode("kasai-binary-1104", "kasai", lambda: kasai_binary_pair(8, 138), 1104, 554, None, source="arXiv:2501.13444"),
        NamedCode("kasai-gf256-2352", "kasai", lambda: kasai_nonbinary(6, 49), 2352, 800, None, source="arXiv:2412.21171 (CSA labels)"),
    ]
}

