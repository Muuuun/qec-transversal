"""Exhaustive fixed-partition sweep: for a code and a width cap W, enumerate
EVERY partition of the n qubits into cells of size <= W, solve the
preservation-algebra unit group exactly, and report the best logical image."""
import sys, time, itertools
from qec_transversal.codes.registry import REGISTRY
from qec_transversal.codes.css import CSSCode
from qec_transversal.api import partition_clifford_group, Completeness
from qec_transversal.utils.symplectic import symplectic_group_order


def partitions_max_cell(n, w):
    """All set partitions of range(n) with every cell of size <= w."""
    def rec(rest, acc):
        if not rest:
            yield [tuple(c) for c in acc]
            return
        first, tail = rest[0], rest[1:]
        for size in range(1, min(w, len(rest)) + 1):
            for extra in itertools.combinations(tail, size - 1):
                cell = (first,) + extra
                remaining = [q for q in tail if q not in extra]
                yield from rec(remaining, acc + [cell])
    yield from rec(list(range(n)), [])


def sweep(name, w, verbose=False):
    entry = REGISTRY[name]
    code = CSSCode(*entry.build())
    k = code.k
    target = symplectic_group_order(k)
    best, best_p, seen, incomplete = 1, None, 0, 0
    t0 = time.time()
    for p in partitions_max_cell(code.n, w):
        seen += 1
        r = partition_clifford_group(code, p)
        if r.completeness is not Completeness.COMPLETE:
            incomplete += 1
            continue
        order = r.logical_group_order or 1
        if not r.logical_group_order_is_exact:
            incomplete += 1
        if order > best:
            best, best_p = order, p
    dt = time.time() - t0
    print(f"{name:10s} [[{entry.n},{entry.k},{entry.d}]] width<={w}: "
          f"{seen} partitions, best logical order {best} / |Sp(2k,2)|={target} "
          f"{'FULL' if best == target else 'short'}   "
          f"(non-COMPLETE: {incomplete})  {dt:.1f}s")
    if best_p:
        print(f"           argmax partition: {best_p}")
    return best, target


if __name__ == "__main__":
    for name, w in [(a.split(':')[0], int(a.split(':')[1])) for a in sys.argv[1:]]:
        sweep(name, w)
