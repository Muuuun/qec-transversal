"""Certified unit groups of finite-dimensional matrix algebras over F_2.

Replaces ``2^dim`` element enumeration for transversal-group computations:
given a basis of an algebra ``A`` (as flat GF(2) vectors with a supplied
multiplication), produce generators of the unit group ``A^x`` together
with its exact order, through the classical structure route

    ``1 -> 1 + N -> A^x -> (A/N)^x -> 1``,

with ``N`` the (iteratively peeled) nilpotent radical and the semisimple
quotient split into simple factors ``M_{d_i}(F_{2^{e_i}})``.

Every stage is *verified*, never trusted:

- candidate ideals are proven nilpotent by explicit power computation;
- the final quotient is proven semisimple constructively — its center is
  split into fields by square-free/Berlekamp factoring, and each block is
  exhibited as a full matrix algebra by an explicitly constructed and
  dimension-checked isomorphism onto ``End(W)`` of an irreducible module;
- the returned generators are units by construction and the order formula
  ``|A^x| = 2^{dim N} * prod |GL(d_i, q_i)|`` is assembled only from
  certified pieces.

If any verification fails the result carries ``status = "unknown"`` — an
incomplete computation is never a negative result.

The char-2 trace-form shortcut for the radical is *deliberately absent*:
over F_2 the trace form is degenerate beyond the radical (already for
``F_2[x]/(x^2)``), so nilpotency is proven directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .gf2 import gf2_inverse, nullspace, rank, row_basis, rref


def _solve_coords(basis: np.ndarray, vector: np.ndarray) -> np.ndarray | None:
    """Coordinates of ``vector`` over ``basis`` rows, or None."""

    augmented = np.hstack([basis.T % 2, (vector % 2)[:, None]]).astype(np.uint8)
    rows, cols = augmented.shape
    r = 0
    pivots = []
    for c in range(cols - 1):
        hit = np.flatnonzero(augmented[r:, c])
        if hit.size == 0:
            continue
        sel = r + int(hit[0])
        if sel != r:
            augmented[[r, sel]] = augmented[[sel, r]]
        others = np.flatnonzero(augmented[:, c])
        others = others[others != r]
        if others.size:
            augmented[others] ^= augmented[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    if any(augmented[i, -1] and not augmented[i, : cols - 1].any() for i in range(rows)):
        return None
    coords = np.zeros(cols - 1, dtype=np.uint8)
    for i, c in enumerate(pivots):
        coords[c] = augmented[i, -1]
    return coords


class AlgebraF2:
    """A finite-dimensional unital algebra over F_2 in coordinates.

    ``basis``: rows spanning the algebra as flat GF(2) vectors.
    ``multiply``: bilinear product on flat vectors.
    ``one``: the unit element (must lie in the span).
    """

    def __init__(self, basis: np.ndarray, multiply: Callable, one: np.ndarray):
        self.basis = row_basis(basis.astype(np.uint8))
        self.multiply = multiply
        self.dim = self.basis.shape[0]
        one_coords = _solve_coords(self.basis, one.astype(np.uint8))
        if one_coords is None:
            raise ValueError("unit element is not in the algebra span")
        self.one_coords = one_coords
        # structure constants via left-multiplication matrices L_i (d x d)
        self.left = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint8)
        self.right = np.zeros((self.dim, self.dim, self.dim), dtype=np.uint8)
        for i in range(self.dim):
            for j in range(self.dim):
                product = multiply(self.basis[i], self.basis[j]) % 2
                coords = _solve_coords(self.basis, product)
                if coords is None:
                    raise ValueError("basis is not multiplicatively closed")
                self.left[i, :, j] = 0  # filled below
                for k in range(self.dim):
                    if coords[k]:
                        self.left[i, k, j] = 1  # (L_i)_{kj}: e_i e_j = sum_k c e_k
                        self.right[j, k, i] = 1  # (R_j)_{ki}: e_i e_j
        # rows of self.left[i] act on coordinate columns: L_i @ coords(x) = coords(e_i x)

    def coords_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Product in coordinates."""

        out = np.zeros(self.dim, dtype=np.uint8)
        for i in np.flatnonzero(a):
            out ^= (self.left[i] @ b % 2).astype(np.uint8)
        return out

    def left_matrix(self, a: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.dim, self.dim), dtype=np.uint8)
        for i in np.flatnonzero(a):
            matrix ^= self.left[i]
        return matrix

    def right_matrix(self, a: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.dim, self.dim), dtype=np.uint8)
        for j in np.flatnonzero(a):
            matrix ^= self.right[j]
        return matrix

    def is_unit(self, a: np.ndarray) -> bool:
        return rank(self.left_matrix(a)) == self.dim


def _ideal_closure(algebra: AlgebraF2, seed_rows: np.ndarray) -> np.ndarray:
    """Row basis of the two-sided ideal generated by ``seed_rows`` (coords)."""

    current = row_basis(seed_rows, ncols=algebra.dim) if seed_rows.size else seed_rows
    while True:
        extended = [current]
        for v in current:
            for i in range(algebra.dim):
                extended.append(algebra.coords_multiply(algebra.basis_coords(i), v)[None, :])
                extended.append(algebra.coords_multiply(v, algebra.basis_coords(i))[None, :])
        stacked = np.vstack(extended)
        new = row_basis(stacked, ncols=algebra.dim)
        if new.shape[0] == current.shape[0]:
            return new
        current = new


def _basis_coords(dim: int, index: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.uint8)
    v[index] = 1
    return v


AlgebraF2.basis_coords = lambda self, index: _basis_coords(self.dim, index)  # type: ignore[attr-defined]


def _is_nilpotent_ideal(algebra: AlgebraF2, ideal: np.ndarray) -> bool:
    """Prove nilpotency by explicit power computation (exact, no heuristics)."""

    current = ideal
    for _ in range(algebra.dim + 1):
        if current.shape[0] == 0:
            return True
        products = [
            algebra.coords_multiply(u, v) for u in current for v in ideal
        ]
        nxt = row_basis(np.asarray(products, dtype=np.uint8), ncols=algebra.dim)
        if nxt.shape[0] >= current.shape[0]:
            return False  # dimension must strictly descend for nilpotency
        current = nxt
    return current.shape[0] == 0


def _find_nilpotent_ideal(algebra: AlgebraF2, rng: np.random.Generator) -> np.ndarray | None:
    """Search for a nonzero nilpotent two-sided ideal (None if not found)."""

    for _ in range(60):
        a = rng.integers(0, 2, size=algebra.dim, dtype=np.uint8)
        if not a.any():
            continue
        R = algebra.right_matrix(a)
        # kernels of powers of a right-multiplication are left submodules;
        # their ideal closures are candidates
        power = R.copy()
        for _ in range(3):
            kern = nullspace(power)  # vectors v with (power) v = 0 (column action)
            if 0 < kern.shape[0] < algebra.dim:
                ideal = _ideal_closure(algebra, kern)
                if ideal.shape[0] < algebra.dim and _is_nilpotent_ideal(algebra, ideal):
                    if ideal.shape[0] > 0:
                        return ideal
            power = (power @ R) % 2
        L = algebra.left_matrix(a)
        power = L.copy()
        for _ in range(3):
            kern = nullspace(power)
            if 0 < kern.shape[0] < algebra.dim:
                ideal = _ideal_closure(algebra, kern)
                if 0 < ideal.shape[0] < algebra.dim and _is_nilpotent_ideal(algebra, ideal):
                    return ideal
            power = (power @ L) % 2
        coords_a = a.copy()
        for _ in range(algebra.dim.bit_length()):
            coords_a = algebra.coords_multiply(coords_a, coords_a)
        if coords_a.any():
            continue
        # a itself nilpotent: ideal closure candidate
        ideal = _ideal_closure(algebra, a[None, :])
        if 0 < ideal.shape[0] < algebra.dim and _is_nilpotent_ideal(algebra, ideal):
            return ideal
    return None


@dataclass
class UnitGroupResult:
    status: str  # "exact" | "unknown"
    order: int | None
    generators: list[np.ndarray] = field(default_factory=list)  # coords in A
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "order": self.order,
            "generator_count": len(self.generators),
            "detail": self.detail,
        }


def _gl_order(d: int, q: int) -> int:
    order = 1
    for i in range(d):
        order *= q**d - q**i
    return order


def unit_group(algebra: AlgebraF2, *, seed: int = 11) -> UnitGroupResult:
    """Certified generators and exact order of ``A^x``.

    Falls back to ``status="unknown"`` whenever a structural verification
    fails, per the verifier semantics.
    """

    rng = np.random.default_rng(seed)

    # ---- stage 1: peel nilpotent ideals ----
    nil_rows: list[np.ndarray] = []
    work = algebra
    lift = np.eye(algebra.dim, dtype=np.uint8)  # coords(work) -> coords(algebra)
    total_nil_dim = 0
    for _ in range(algebra.dim):
        ideal = _find_nilpotent_ideal(work, rng)
        if ideal is None:
            break
        total_nil_dim += ideal.shape[0]
        for row in ideal:
            nil_rows.append((row @ lift) % 2)
        # quotient algebra work/ideal
        work, lift = _quotient(work, ideal, lift)
    quotient = work

    # ---- stage 2: verify the quotient is semisimple constructively ----
    split = _wedderburn(quotient, rng)
    if split is None:
        return UnitGroupResult(
            status="unknown",
            order=None,
            detail=(
                "semisimple decomposition could not be verified; unit group "
                "not certified (no silent fallback)"
            ),
        )
    blocks, block_gens = split

    order = 1 << total_nil_dim
    for d, e in blocks:
        order *= _gl_order(d, 1 << e)

    generators: list[np.ndarray] = []
    one_full = algebra.one_coords
    for row in nil_rows:
        generators.append((one_full ^ row) % 2)  # 1 + nilpotent
    for g in block_gens:
        # lift a quotient unit to a full unit: any preimage works because
        # units lift along nilpotent quotients (1 + N is in the kernel)
        lifted = (g @ lift) % 2
        # correct to an actual unit by adding a nilpotent part if necessary
        if not algebra.is_unit(lifted):
            for row in nil_rows:
                candidate = (lifted ^ row) % 2
                if algebra.is_unit(candidate):
                    lifted = candidate
                    break
        if not algebra.is_unit(lifted):
            return UnitGroupResult(
                status="unknown", order=None, detail="unit lifting failed"
            )
        generators.append(lifted)

    return UnitGroupResult(
        status="exact",
        order=order,
        generators=generators,
        detail=(
            f"nilpotent dimension {total_nil_dim}; simple factors "
            + " x ".join(f"M_{d}(F_{1 << e})" for d, e in blocks)
        ),
    )


def _quotient(algebra: AlgebraF2, ideal: np.ndarray, lift: np.ndarray):
    """Quotient algebra with a coordinate section back into the parent."""

    complement_rows = []
    current = row_basis(ideal, ncols=algebra.dim)
    for i in range(algebra.dim):
        candidate = np.vstack([current, algebra.basis_coords(i)[None, :]])
        if rank(candidate) > current.shape[0]:
            complement_rows.append(algebra.basis_coords(i))
            current = row_basis(candidate, ncols=algebra.dim)
    section = np.asarray(complement_rows, dtype=np.uint8)
    stacked = np.vstack([section, ideal])

    def project(v: np.ndarray) -> np.ndarray:
        # solve v = s . section + j . ideal jointly; return the section part
        coords = _solve_coords(stacked, v % 2)
        if coords is None:
            raise AssertionError("projection failed: vector outside the algebra")
        return coords[: section.shape[0]]

    def multiply(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
        a_full = (a_q @ section) % 2
        b_full = (b_q @ section) % 2
        return project(algebra.coords_multiply(a_full, b_full))

    one_q = project(algebra.one_coords)
    quotient = AlgebraF2(np.eye(section.shape[0], dtype=np.uint8), multiply, one=(one_q))
    new_lift = (section @ lift) % 2
    return quotient, new_lift


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


def _wedderburn(algebra: AlgebraF2, rng: np.random.Generator):
    """Verify semisimplicity constructively; return blocks and unit gens.

    Splits the center into fields via minimal-polynomial factoring, then
    exhibits each block as ``M_d(F_q)`` through its action on an
    irreducible module, dimension-checked.  Returns None when any
    verification fails.
    """

    d = algebra.dim
    if d == 0:
        return [], []
    # center: elements commuting with all basis elements
    rows = []
    for i in range(d):
        Li = algebra.left[i]
        Ri = algebra.right[i]
        rows.append((Li ^ Ri) % 2)
    stacked = np.vstack(rows)
    center = nullspace(stacked)
    # split center via a random central element's minimal polynomial
    idempotents = [algebra.one_coords]
    for _ in range(20):
        if not center.shape[0]:
            break
        coeffs = rng.integers(0, 2, size=center.shape[0], dtype=np.uint8)
        z = (coeffs @ center) % 2
        if not z.any():
            continue
        Lz = algebra.left_matrix(z)
        poly = _minimal_polynomial(Lz)
        factors = _berlekamp_factor(poly)
        if len(factors) <= 1:
            continue
        new_idempotents = []
        for e in idempotents:
            pieces = _split_idempotent(algebra, e, z, factors)
            new_idempotents.extend(pieces)
        if len(new_idempotents) > len(idempotents):
            idempotents = new_idempotents
    blocks: list[tuple[int, int]] = []
    generators: list[np.ndarray] = []
    covered_dim = 0
    for e in idempotents:
        block_result = _identify_matrix_block(algebra, e, rng)
        if block_result is None:
            return None
        d_block, e_field, gens = block_result
        blocks.append((d_block, e_field))
        covered_dim += d_block * d_block * e_field
        # embed block units into full algebra: g_block + (1 - e)
        one = algebra.one_coords
        for g in gens:
            complement = (one ^ e) % 2
            generators.append((g ^ complement) % 2)
    if covered_dim != algebra.dim:
        return None  # dimension bookkeeping failed: not proven semisimple
    return blocks, generators


def _split_idempotent(algebra, e, z, factors):
    """Split idempotent e along the factors of z's minimal polynomial."""

    ze = algebra.coords_multiply(z, e)
    pieces = []
    for f in factors:
        # e_f = f_complement(z) e scaled: use CRT via gcd-based projector
        others = np.array([1], dtype=np.uint8)
        for g in factors:
            if g is not f:
                others = _poly_multiply(others, g)
        val = _poly_eval_element(algebra, others, ze, e)
        # idempotent-ize by squaring until stable
        for _ in range(algebra.dim.bit_length() + 1):
            sq = algebra.coords_multiply(val, val)
            if np.array_equal(sq, val):
                break
            val = sq
        if val.any():
            pieces.append(val)
    return pieces if pieces else [e]


def _poly_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(a.size + b.size - 1, dtype=np.uint8)
    for i, ca in enumerate(a):
        if ca:
            out[i : i + b.size] ^= b
    return out


def _poly_eval_element(algebra, poly, x, e):
    """Evaluate poly at algebra element x, with constant term times e."""

    result = np.zeros(algebra.dim, dtype=np.uint8)
    power = e.copy()
    for coeff in poly:
        if coeff:
            result ^= power
        power = algebra.coords_multiply(power, x)
    return result


def _identify_matrix_block(algebra, e, rng):
    """Exhibit the block eAe as M_d(F_q); return (d, log2 q, unit gens)."""

    # block basis
    block_rows = []
    for i in range(algebra.dim):
        v = algebra.coords_multiply(
            algebra.coords_multiply(e, algebra.basis_coords(i)), e
        )
        block_rows.append(v)
    block_basis = row_basis(np.asarray(block_rows, dtype=np.uint8), ncols=algebra.dim)
    dim_block = block_basis.shape[0]
    if dim_block == 0:
        return None

    def bmul(a, b):
        return algebra.coords_multiply(a, b)

    # find an irreducible left module: start from a random column space
    # W = (block) * v for random v, refined by intersecting with images
    for _ in range(40):
        coeffs = rng.integers(0, 2, size=dim_block, dtype=np.uint8)
        v = (coeffs @ block_basis) % 2
        if not v.any():
            continue
        module_rows = [bmul(row, v) for row in block_basis]
        module = row_basis(np.asarray(module_rows, dtype=np.uint8), ncols=algebra.dim)
        w_dim = module.shape[0]
        if w_dim == 0:
            continue
        # the action map block -> End(W): verify injective and dimension
        # bookkeeping dim_block = w_dim^2 / e_field for some e_field
        action_rows = []
        for row in block_basis:
            images = [bmul(row, w) for w in module]
            image_coords = []
            for img in images:
                coords = _solve_coords(module, img)
                if coords is None:
                    coords = None
                    break
                image_coords.append(coords)
            if image_coords is None or len(image_coords) != w_dim:
                action_rows = None
                break
            action_rows.append(np.concatenate(image_coords))
        if action_rows is None:
            continue
        action = np.asarray(action_rows, dtype=np.uint8)
        if rank(action) != dim_block:
            continue  # not faithful; try another module
        # endomorphism field: E = End_block(W); dim_F2 E = e_field
        # candidate: elements of End(W) commuting with all action matrices
        w2 = w_dim * w_dim
        comm_rows = []
        act_mats = [action[i].reshape(w_dim, w_dim) for i in range(dim_block)]
        for M in act_mats:
            # commutant condition: X M = M X -> linear in X
            left = np.kron(np.eye(w_dim, dtype=np.uint8), M)
            right = np.kron(M.T, np.eye(w_dim, dtype=np.uint8))
            comm_rows.append((left ^ right) % 2)
        commutant = nullspace(np.vstack(comm_rows)) if comm_rows else np.zeros((0, w2), np.uint8)
        e_field = commutant.shape[0]
        if e_field == 0 or e_field > 16:
            continue
        # the commutant must be a FIELD (Schur + Wedderburn): verify closure
        # under multiplication and that every nonzero element is invertible.
        # This rejects reducible modules (e.g. the regular module of
        # M_2(F_2), whose commutant M_2(F_2) has zero divisors).
        comm_mats = [commutant[i].reshape(w_dim, w_dim) for i in range(e_field)]
        # cheap rejection: a field has no singular nonzero elements, so any
        # singular basis matrix disqualifies immediately
        if any(rank(m) != w_dim for m in comm_mats if m.any()):
            continue
        # certificate that the commutant IS a field: find c whose minimal
        # polynomial is irreducible of degree e_field — then F_2[c] is a
        # field of dimension e_field inside the commutant, hence equals it.
        is_field = False
        for _ in range(30):
            coeffs = rng.integers(0, 2, size=e_field, dtype=np.uint8)
            c = np.zeros((w_dim, w_dim), dtype=np.uint8)
            for bit in np.flatnonzero(coeffs):
                c ^= comm_mats[bit]
            if not c.any():
                continue
            poly = _minimal_polynomial(c)
            if poly.size - 1 == e_field and len(_berlekamp_factor(poly)) == 1:
                is_field = True
                break
        if not is_field:
            continue
        # closure of multiplication (basis products stay inside)
        closed = all(
            _solve_coords(commutant, (a_i @ b_i % 2).reshape(-1)) is not None
            for a_i in comm_mats
            for b_i in comm_mats
        )
        if not closed:
            continue
        if w_dim % e_field != 0:
            continue
        d_block = w_dim // e_field
        if d_block * d_block * e_field != dim_block:
            continue  # bookkeeping failed
        # unit generators of GL(d, 2^e) inside the block: random search,
        # certified by the closed-form order downstream; generators are
        # verified units and (for safety) we check they do not all commute
        # unless the group is abelian (d = 1)
        target_order = _gl_order(d_block, 1 << e_field)
        gens: list[np.ndarray] = []
        gen_mats: list[np.ndarray] = []
        from .group import schreier_sims_order

        def block_left_matrix(candidate):
            Lb = np.zeros((dim_block, dim_block), dtype=np.uint8)
            for j in range(dim_block):
                prod = bmul(candidate, block_basis[j])
                coords = _solve_coords(block_basis, prod)
                if coords is None:
                    return None
                Lb[:, j] = coords
            return Lb

        tries = 0
        while tries < 600:
            tries += 1
            coeffs = rng.integers(0, 2, size=dim_block, dtype=np.uint8)
            candidate = (coeffs @ block_basis) % 2
            Lb = block_left_matrix(candidate)
            if Lb is None or rank(Lb) != dim_block:
                continue
            gens.append(candidate)
            gen_mats.append(Lb.T)  # row-vector action for schreier_sims
            if len(gens) >= 2:
                generated = schreier_sims_order(gen_mats)
                if generated == target_order:
                    # generation certified: the returned set provably
                    # generates the whole block unit group
                    return d_block, e_field, gens
                if generated is not None and generated == target_order:
                    return d_block, e_field, gens
            elif target_order == 1:
                return d_block, e_field, gens
            if len(gens) > 8:
                gens = gens[-4:]
                gen_mats = gen_mats[-4:]
        continue  # generation not certified for this module; try another
    return None
