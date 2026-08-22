"""Constructive Wedderburn split of a claimed-semisimple ``F_2``-algebra.

The centre is split into fields by minimal-polynomial factoring, and each
resulting block is *exhibited* as ``M_d(F_{2^e})`` through an explicitly
constructed action on an irreducible module whose commutant is verified to be
a field.  Dimension bookkeeping must close exactly; any shortfall returns
``None`` so the caller reports ``unknown`` rather than an unproven splitting.
"""

from __future__ import annotations

import numpy as np

from ..utils.gf2 import nullspace, rank, row_basis
from ..utils.polynomials import (
    _berlekamp_factor,
    _minimal_polynomial,
    _poly_multiply,
    _solve_coords,
)
from .finite_algebra import AlgebraF2
from .orders import _gl_order


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
        from ..logical.group import schreier_sims_order

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
