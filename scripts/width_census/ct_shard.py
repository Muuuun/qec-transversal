"""One shard of an exhaustive fixed-partition sweep over a codetables code."""
import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ct", ROOT / "scripts" / "codetables_n7_census.py")
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)

from staircase import partitions_max_cell
from qec_transversal.codes.stabilizer import StabilizerCode
from qec_transversal.api import partition_clifford_group, Completeness
from qec_transversal.utils.symplectic import symplectic_group_order

n, k, width, shard, shards = (int(a) for a in sys.argv[1:6])
out = sys.argv[6]
html = ct.fetch_page(n, k)
code = StabilizerCode(ct.parse_stabilizer_matrix(html))
d = ct.parse_d_bounds(html)
target = symplectic_group_order(code.k)
mine = [p for i, p in enumerate(partitions_max_cell(n, width)) if i % shards == shard]
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
            seen.add(key); gens.append(g.astype(int).tolist())
    done += 1
    if done % 25 == 0 or done == len(mine):
        rate = (time.time() - t0) / done
        print(f"[[{n},{k},{d}]] w{width} shard {shard}: {done}/{len(mine)} best={best} "
              f"unknown={unknown} eta={rate*(len(mine)-done)/60:.1f}min", flush=True)
json.dump({"n": n, "k": k, "d": d, "width": width, "shard": shard, "count": len(mine),
           "best": best, "best_partition": best_p, "unknown": unknown, "target": target,
           "generators": gens, "seconds": time.time() - t0},
          open(f"{out}/ct_{n}_{k}_{width}_{shard}.json", "w"))
print(f"shard {shard} DONE best={best} unknown={unknown} {time.time()-t0:.0f}s", flush=True)
