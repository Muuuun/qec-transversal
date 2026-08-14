"""Regenerates zoo_data.json by analyzing every registry code.

Run from the repo root with the project installed:
    python docs/zoo/make_data.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.automorphisms import analyze_automorphisms
from qec_transversal.dualities import candidates_for
from qec_transversal.hierarchy import analyze_hierarchy
from qec_transversal.matching import analyze_matching, logical_group_summary
from qec_transversal.monomial import analyze_monomial
from qec_transversal.stabilizer import StabilizerCode, analyze_partition_clifford

try:  # the one-block module may be authored concurrently; degrade gracefully
    from qec_transversal.oneblock import analyze_one_block, single_matching_fullness
except Exception:  # noqa: BLE001 - absent or half-written module: skip cleanly
    analyze_one_block = None
    single_matching_fullness = None

HERE = Path(__file__).resolve().parent

# One-block generated-group sweep budgets: the giant codes are skipped outright,
# everything else gets a per-code wall-clock budget passed to the solver so the
# whole sweep stays bounded.
ONE_BLOCK_MAX_N = 1500
ONE_BLOCK_BUDGET_S = 60.0        # per-code budget for n <= 400
ONE_BLOCK_BUDGET_LARGE_S = 20.0  # per-code budget for 400 < n <= ONE_BLOCK_MAX_N
ONE_BLOCK_WIDE_CAP = 48          # sampled matchings for n <= 300
ONE_BLOCK_CAP = 16               # sampled matchings above that


def one_block_report(name: str, code: CSSCode):
    """row['one_block'] per the data contract: the analyze_one_block to_dict()
    payload, {'error': str} when skipped by cap, or None when the module is
    unavailable.

    The registry ``name`` is load-bearing, not cosmetic: without it
    :func:`analyze_one_block` cannot look up the family's structural duality
    candidates (``dualities.candidates_for``) and silently falls back to the
    BLISS-discovered involution alone, understating the generated group by
    orders of magnitude (gross: 8,640 instead of 11,059,200).
    """

    if analyze_one_block is None:
        return None
    if code.n > ONE_BLOCK_MAX_N:
        return {"error": f"skipped: n = {code.n} > {ONE_BLOCK_MAX_N} sweep cap"}
    budget = ONE_BLOCK_BUDGET_S if code.n <= 400 else ONE_BLOCK_BUDGET_LARGE_S
    # Sampling breadth, not wall clock, is what decides these verdicts: rm64
    # reports a lower bound with 16 sampled matchings and certifies the full
    # Sp(40,2) with 48 — in five seconds either way. Small codes therefore get
    # the wider sample; large ones keep the narrow one, where each matching
    # analysis is what costs.
    cap = ONE_BLOCK_WIDE_CAP if code.n <= 300 else ONE_BLOCK_CAP
    start = time.time()
    result = analyze_one_block(code, name=name, time_budget_s=budget, involution_cap=cap)
    summary = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    summary.setdefault("seconds", round(time.time() - start, 2))
    if summary.get("is_full") is True and single_matching_fullness is not None:
        # Only a full code can be full from one matching, and only that
        # question distinguishes "needs varying matchings" from "one matching
        # plus automorphisms sufficed" — the distinction the zoo renders, so it
        # has to ship as data rather than as prose. Cheap: the full codes are
        # the small ones.
        try:
            single_start = time.time()
            summary.update(
                single_matching_fullness(code, name, time_budget_s=budget, involution_cap=cap)
            )
            summary["single_matching_seconds"] = round(time.time() - single_start, 2)
        except Exception as error:  # noqa: BLE001 - record, never abort the sweep
            summary["single_matching_full"] = None
            summary["single_matching_error"] = str(error)
    return summary


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

# Per-code checkpoint. A full sweep is hours of work and any interruption —
# a stopped background task, a machine sleeping through the nightly cron —
# used to throw all of it away, so completed rows are journalled as they land
# and reused on the next run. Delete the file (or pass --fresh) to force a
# clean sweep; a row is reused only when the registry still defines that code.
CHECKPOINT = HERE / "zoo_data.checkpoint.json"
FRESH = "--fresh" in sys.argv

done: dict[str, dict] = {}
if CHECKPOINT.exists() and not FRESH:
    try:
        done = {row["name"]: row for row in json.loads(CHECKPOINT.read_text())}
        print(f"resuming: {len(done)} code(s) already analyzed in {CHECKPOINT.name}", flush=True)
    except Exception as error:  # noqa: BLE001 - a corrupt journal must not block a sweep
        print(f"checkpoint unreadable ({error}); starting fresh", flush=True)
        done = {}

out = []
for name, entry in REGISTRY.items():
    if name in done:
        out.append(done[name])
        print(name, "cached", flush=True)
        continue
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
    try:
        one_block_summary = one_block_report(name, code)
    except Exception as error:  # noqa: BLE001 - record, never abort the sweep
        one_block_summary = {"error": str(error)}
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
            "one_block": one_block_summary,
        }
    )
    print(name, "done", flush=True)
    CHECKPOINT.write_text(json.dumps(out, indent=0))

tmp = HERE / "zoo_data.json.tmp"
tmp.write_text(json.dumps(out, indent=0))
tmp.replace(HERE / "zoo_data.json")
CHECKPOINT.unlink(missing_ok=True)
print("saved", HERE / "zoo_data.json")
