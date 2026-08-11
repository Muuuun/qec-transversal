"""Regenerates zoo_data.json by analyzing every registry code.

Run from the repo root with the project installed:
    python docs/zoo/make_data.py
"""

import json
import time
from pathlib import Path

from qec_transversal import CSSCode, REGISTRY

HERE = Path(__file__).resolve().parent

out = []
for name, entry in REGISTRY.items():
    h_x, h_z = entry.build()
    code = CSSCode(h_x, h_z)
    start = time.time()
    analysis = code.analyze_transversal()
    report = analysis.to_dict()
    elapsed = time.time() - start
    generators = [
        {
            "family": g.family,
            "support": g.support,
            "weight": len(g.support),
            "logical_identity": bool(g.is_logical_identity),
            "logical": g.logical_symplectic.astype(int).tolist() if code.k <= 2 else None,
        }
        for g in analysis.generators
    ]
    out.append(
        {
            "name": name,
            "family": entry.family,
            "source": entry.source,
            "n": code.n,
            "k": code.k,
            "d": entry.d,
            "d_ub": entry.d_is_upper_bound,
            "rank_MZ": int(analysis.a_z.constraints.shape[0]),
            "rank_MX": int(analysis.a_x.constraints.shape[0]),
            "dim_AZ": analysis.a_z.dimension,
            "dim_AX": analysis.a_x.dimension,
            "structure": report["structure"],
            "order": report["logical_group"]["order"],
            "is_full": report["logical_group"]["is_full_logical_clifford"],
            "certified": report["certificate"]["certified"],
            "seconds": round(elapsed, 2),
            "generators": generators,
            "checks_weight_X": sorted({int(w) for w in h_x.sum(axis=1)}),
            "checks_weight_Z": sorted({int(w) for w in h_z.sum(axis=1)}),
        }
    )
    print(name, "done", flush=True)

(HERE / "zoo_data.json").write_text(json.dumps(out, indent=0))
print("saved", HERE / "zoo_data.json")
