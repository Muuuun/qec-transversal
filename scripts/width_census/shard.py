"""One shard of an exhaustive fixed-partition sweep, with live progress."""
import json, sys, time
from staircase import partitions_max_cell
from qec_transversal.codes.registry import REGISTRY
from qec_transversal.codes.css import CSSCode
from qec_transversal.api import partition_clifford_group, Completeness
from qec_transversal.utils.symplectic import symplectic_group_order

name, width, shard, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
entry = REGISTRY[name]
code = CSSCode(*entry.build())
target = symplectic_group_order(code.k)
mine = [p for i, p in enumerate(partitions_max_cell(code.n, width)) if i % shards == shard]
best, best_p, unknown, done = 1, None, 0, 0
gens, seen = [], set()
t0 = time.time()
for p in mine:
    r = partition_clifford_group(code, p)
    if r.completeness is not Completeness.COMPLETE or not r.logical_group_order_is_exact:
        unknown += 1
    else:
        o = r.logical_group_order or 1
        if o > best:
            best, best_p = o, p
    for g in r.logical_generators:
        key = g.astype(int).tobytes()
        if key not in seen:
            seen.add(key)
            gens.append(g.astype(int).tolist())
    done += 1
    if done % 20 == 0 or done == len(mine):
        rate = (time.time() - t0) / done
        print(f"shard {shard}: {done}/{len(mine)} best={best} unknown={unknown} "
              f"eta={rate*(len(mine)-done)/60:.1f}min", flush=True)
json.dump({"shard": shard, "count": len(mine), "best": best, "best_partition": best_p,
           "unknown": unknown, "target": target, "generators": gens,
           "seconds": time.time() - t0},
          open(f"{sys.argv[5]}/shard_{name}_{width}_{shard}.json", "w"))
print(f"shard {shard} DONE best={best} unknown={unknown} {time.time()-t0:.0f}s", flush=True)
