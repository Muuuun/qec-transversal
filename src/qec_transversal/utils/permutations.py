"""Permutation groups: exact orders, sampling, and symmetric-group facts."""

from __future__ import annotations

import numpy as np

from .polynomials import _primes_up_to


def permutation_group_order(generators: list[np.ndarray]) -> int:
    """Exact order of a permutation group by point-based Schreier-Sims."""

    degree = len(generators[0]) if generators else 0
    gens = [np.asarray(g, dtype=int) for g in generators if not np.array_equal(g, np.arange(degree))]
    if not gens:
        return 1

    def orbit_transversal(point: int, group_gens):
        transversal = {point: np.arange(degree)}
        queue = [point]
        while queue:
            p = queue.pop()
            rep = transversal[p]
            for g in group_gens:
                image = int(g[p])
                if image not in transversal:
                    transversal[image] = g[rep]
                    queue.append(image)
        return transversal

    order = 1
    current = gens
    for point in range(degree):
        if not current:
            break
        moved = [g for g in current if any(int(g[p]) != p for p in range(degree))]
        if not moved:
            break
        base = next(
            (p for p in range(degree) if any(int(g[p]) != p for g in current)), None
        )
        if base is None:
            break
        transversal = orbit_transversal(base, current)
        order *= len(transversal)
        inverse = {p: np.argsort(rep) for p, rep in transversal.items()}
        stabilizer = []
        seen = set()
        for p, rep in transversal.items():
            for g in current:
                image = int(g[p])
                schreier = inverse[image][g[rep]]
                key = schreier.tobytes()
                if key not in seen and not np.array_equal(schreier, np.arange(degree)):
                    seen.add(key)
                    stabilizer.append(schreier)
        current = stabilizer
    return order


def random_involution(rng: np.random.Generator, n: int) -> np.ndarray:
    """A uniformly paired random matching of ``n`` points."""

    order = rng.permutation(n)
    tau = np.arange(n)
    for index in range(0, n - 1, 2):
        i, j = int(order[index]), int(order[index + 1])
        tau[i], tau[j] = j, i
    return tau


_SYMMETRIC_ORDER_CACHE: dict[int, frozenset[int]] = {}


def _symmetric_group_element_orders(m: int, *, cap: int = 4_000_000) -> frozenset[int]:
    """The exact set of element orders of ``S_m``.

    An element order is the lcm of a partition of ``m``; equivalently a
    product over distinct primes ``p`` of one power ``p^a``, with the sum of
    the chosen prime powers at most ``m``.  Dynamic programming over primes
    (order -> minimal spent budget) enumerates the set exactly.  This set is
    closed under divisors, which the recognition certificate relies on.
    """

    cached = _SYMMETRIC_ORDER_CACHE.get(m)
    if cached is not None:
        return cached
    best: dict[int, int] = {1: 0}
    for p in _primes_up_to(m):
        powers = []
        q = p
        while q <= m:
            powers.append(q)
            q *= p
        for order, spent in list(best.items()):
            for q in powers:
                cost = spent + q
                if cost <= m:
                    new_order = order * q
                    previous = best.get(new_order)
                    if previous is None or previous > cost:
                        best[new_order] = cost
            if len(best) > cap:
                raise RuntimeError("symmetric-order enumeration exceeded its cap")
    result = frozenset(best)
    _SYMMETRIC_ORDER_CACHE[m] = result
    return result
