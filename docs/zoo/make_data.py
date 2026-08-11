"""Regenerates zoo_data.json by analyzing every registry code.

Run from the repo root with the project installed:
    python docs/zoo/make_data.py
"""

import json
import time
from pathlib import Path

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.automorphisms import analyze_automorphisms
from qec_transversal.dualities import candidates_for
from qec_transversal.hierarchy import analyze_hierarchy
from qec_transversal.matching import analyze_matching, logical_group_summary

HERE = Path(__file__).resolve().parent


def fold_report(name: str, code: CSSCode, discovered=None) -> dict:
    """Run structural duality candidates plus any BLISS-discovered involutive
    duality; combine certified folds."""

    candidates = list(candidates_for(name))
    if discovered is not None:
        tested = {tau.tobytes() for _, tau in candidates}
        if discovered.tobytes() not in tested:
            candidates.append(("BLISS-discovered duality", discovered))
    analyses = []
    combined_generators = []
    for label, tau in candidates:
        start = time.time()
        analysis = analyze_matching(code, tau)
        summary = analysis.to_dict()
        summary["label"] = label
        summary["seconds"] = round(time.time() - start, 2)
        analyses.append(summary)
        if summary["is_zx_duality"]:
            combined_generators += [
                g.logical_symplectic for g in analysis.generators if not g.is_logical_identity
            ]
            if analysis.fold_hadamard is not None and not analysis.fold_hadamard.is_logical_identity:
                combined_generators.append(analysis.fold_hadamard.logical_symplectic)
    certified_dualities = sum(1 for a in analyses if a["is_zx_duality"])
    nontrivial = sum(a["nontrivial_generator_count"] for a in analyses if a["is_zx_duality"])
    return {
        "candidates_tested": len(analyses),
        "certified_dualities": certified_dualities,
        "nontrivial_fold_generators": nontrivial,
        "combined_group": logical_group_summary(combined_generators, code.k),
        "all_certified": all(a["certified"] for a in analyses),
        "analyses": analyses,
    }

out = []
for name, entry in REGISTRY.items():
    h_x, h_z = entry.build()
    code = CSSCode(h_x, h_z)
    try:
        aut = analyze_automorphisms(code)
        aut_summary = aut.to_dict()
        discovered = aut.involutive_duality()
    except ImportError:
        aut_summary, discovered = None, None
    if candidates_for(name) or discovered is not None:
        fold = fold_report(name, code, discovered)
    else:
        fold = None
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
            "fold": fold,
            "automorphisms": aut_summary,
            "hierarchy": {
                "Z": analyze_hierarchy(code, "Z").to_dict(),
                "X": analyze_hierarchy(code, "X").to_dict(),
            },
        }
    )
    print(name, "done", flush=True)

tmp = HERE / "zoo_data.json.tmp"
tmp.write_text(json.dumps(out, indent=0))
tmp.replace(HERE / "zoo_data.json")
print("saved", HERE / "zoo_data.json")
