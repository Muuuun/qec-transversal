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
import numpy as np
from qec_transversal.monomial import analyze_monomial
from qec_transversal.stabilizer import StabilizerCode, analyze_partition_clifford

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
    first_tau = None
    for (label, tau), a in zip(candidates, analyses):
        if a["is_zx_duality"] and first_tau is None:
            first_tau = np.asarray(tau, dtype=int)
    certified_dualities = sum(1 for a in analyses if a["is_zx_duality"])
    nontrivial = sum(a["nontrivial_generator_count"] for a in analyses if a["is_zx_duality"])
    return {
        "candidates_tested": len(analyses),
        "certified_dualities": certified_dualities,
        "nontrivial_fold_generators": nontrivial,
        "combined_group": logical_group_summary(combined_generators, code.k),
        "all_certified": all(a["certified"] for a in analyses),
        "analyses": analyses,
    }, first_tau

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
        fold, fold_tau = fold_report(name, code, discovered)
    else:
        fold, fold_tau = None, None
    stacked = StabilizerCode(
        np.vstack(
            [
                np.hstack([code.c_x, np.zeros_like(code.c_x)]),
                np.hstack([np.zeros_like(code.c_z), code.c_z]),
            ]
        )
    )
    natural = np.vstack(
        [
            np.hstack([h_x & 1, np.zeros_like(h_x)]),
            np.hstack([np.zeros_like(h_z), h_z & 1]),
        ]
    )
    try:
        mono_start = time.time()
        monomial_summary = analyze_monomial(stacked, natural_rows=natural).to_dict()
        monomial_summary["seconds"] = round(time.time() - mono_start, 2)
    except Exception as error:  # noqa: BLE001 - record, never abort the sweep
        monomial_summary = {"error": str(error)}
    if fold_tau is not None and code.n <= 400:
        try:
            cells = []
            seen_q = set()
            for i, target in enumerate(fold_tau):
                if i in seen_q:
                    continue
                if target == i:
                    cells.append((i,))
                    seen_q.add(i)
                else:
                    cells.append((i, int(target)))
                    seen_q.update({i, int(target)})
            tl_start = time.time()
            two_local_summary = analyze_partition_clifford(
                stacked, cells, dim_cap=20
            ).to_dict()
            two_local_summary.pop("cells", None)
            two_local_summary["seconds"] = round(time.time() - tl_start, 2)
        except Exception as error:  # noqa: BLE001
            two_local_summary = {"error": str(error)}
    elif fold_tau is not None:
        two_local_summary = {"skipped": "n > 400"}
    else:
        two_local_summary = None
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
            "hierarchy4": (
                {
                    "Z": analyze_hierarchy(code, "Z", level=4).to_dict(),
                    "X": analyze_hierarchy(code, "X", level=4).to_dict(),
                }
                if code.n <= 64
                else None
            ),
            "monomial": monomial_summary,
            "two_local": two_local_summary,
        }
    )
    print(name, "done", flush=True)

tmp = HERE / "zoo_data.json.tmp"
tmp.write_text(json.dumps(out, indent=0))
tmp.replace(HERE / "zoo_data.json")
print("saved", HERE / "zoo_data.json")
