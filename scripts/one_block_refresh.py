"""Recompute the one_block column for every registry code with the CURRENT engine.

The census column was built across several engine revisions, so generator
counts drifted (cube-832 reported 39 generators then, 31 now).  Anything
derived from G -- notably the depth floor D >= log_G |Sp(2k,2)| -- is only
meaningful when every row comes from one engine version.  This rebuilds that
one column and leaves the others alone.

Budgets and the sampling cap are READ OUT OF ``make_data.py`` rather than
restated here.  That is not fussiness: sampling breadth decides verdicts, and
running with the default cap of 16 instead of the sweep's 48 silently costs
rm64 its FULL certificate.  A copy of the policy would drift; a parse cannot.

    python scripts/one_block_refresh.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.oneblock import analyze_one_block, single_matching_fullness

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "zoo" / "zoo_data.json"
_SOURCE = (ROOT / "docs" / "zoo" / "make_data.py").read_text()


def _constant(name: str) -> float:
    match = re.search(rf"^{name}\s*=\s*([0-9.]+)", _SOURCE, re.MULTILINE)
    if match is None:
        raise SystemExit(f"{name} not found in make_data.py — policy moved; update this script")
    return float(match.group(1))


MAX_N = _constant("ONE_BLOCK_MAX_N")
BUDGET_SMALL = _constant("ONE_BLOCK_BUDGET_S")
BUDGET_LARGE = _constant("ONE_BLOCK_BUDGET_LARGE_S")
BUDGET_GIANT = _constant("ONE_BLOCK_BUDGET_GIANT_S")
WIDE_CAP = int(_constant("ONE_BLOCK_WIDE_CAP"))
CAP = int(_constant("ONE_BLOCK_CAP"))


def policy(n: int) -> tuple[float, int]:
    budget = BUDGET_SMALL if n <= 400 else (BUDGET_LARGE if n <= 1500 else BUDGET_GIANT)
    return budget, (WIDE_CAP if n <= 300 else CAP)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    arguments = parser.parse_args()

    rows = json.loads(DATA.read_text())
    by = {r["name"]: r for r in rows}
    print(f"policy from make_data.py: budgets {BUDGET_SMALL}/{BUDGET_LARGE}/{BUDGET_GIANT}s, "
          f"caps {WIDE_CAP} (n<=300) / {CAP}, max n {MAX_N:.0f}", flush=True)
    for name in arguments.only or [r["name"] for r in rows]:
        entry = REGISTRY[name]
        budget, cap = policy(entry.n)
        start = time.time()
        result = analyze_one_block(
            CSSCode(*entry.build()), name=name, time_budget_s=budget, involution_cap=cap
        )
        payload = result.to_dict()
        payload.setdefault("seconds", round(time.time() - start, 2))
        # Mirror make_data.py: a FULL row also carries the single-matching
        # verdict, which the zoo's §3 sentence is gated on.  Refreshing the
        # column without this silently deletes that sentence.
        if payload.get("is_full") is True:
            single_start = time.time()
            payload.update(
                single_matching_fullness(
                    CSSCode(*entry.build()), name, time_budget_s=budget, involution_cap=cap
                )
            )
            payload["single_matching_seconds"] = round(time.time() - single_start, 2)
        before = (by[name].get("one_block") or {}).get("is_full")
        by[name]["one_block"] = payload
        flag = "" if before == payload["is_full"] else f"  <<< is_full {before} -> {payload['is_full']}"
        print(f"{name:<30} n={entry.n:<5} cap={cap:<3} gens={payload['generator_count']:<4} "
              f"order={payload['logical_order']} full={payload['is_full']}"
              f" [{time.time()-start:.1f}s]{flag}", flush=True)
        DATA.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
