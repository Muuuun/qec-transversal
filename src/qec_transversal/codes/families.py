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
    "cyclic_shift",
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
    "middle_reed_muller",
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
    "twisted_torus_translation",
]
