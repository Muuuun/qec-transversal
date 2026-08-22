import sys, time, resource
from qec_transversal.codes.registry import REGISTRY
from qec_transversal.codes.css import CSSCode
from qec_transversal.api import _as_stabilizer_code
from qec_transversal.ansatz.partition import partition_units_via_structure
from qec_transversal.utils.symplectic import symplectic_group_order
name = sys.argv[1]; cap = int(sys.argv[2])
code = _as_stabilizer_code(CSSCode(*REGISTRY[name].build()))
target = symplectic_group_order(code.k)
for a in sys.argv[3:]:
    cells = [tuple(c) for c in eval(a)]
    t0 = time.time()
    s = partition_units_via_structure(code, cells, method="phi", orbit_cap=cap)
    lg = s.get("logical_group", {}); orb = s.get("orbit") or {}
    print(f"{cells} -> {s['status']} |A^x|={s.get('unit_group_order')} orbit={orb.get('orbit_size')} "
          f"|G|={s.get('symplectic_group_order')} logical={lg.get('order') or lg.get('lower_bound')}"
          f"/{target} {'FULL' if lg.get('order')==target else ''} exact={lg.get('exact')} "
          f"complete={orb.get('generators_complete')} {time.time()-t0:.0f}s "
          f"rssGB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f}", flush=True)
    if s['status'] != 'exact': print("   ", s.get('detail'), flush=True)
