"""Backfills the ``single_matching_full`` fields into ``zoo_data.json``.

``analyze_one_block`` decides whether a code's depth-one layers *together*
generate ``Sp(2k, 2)``; it does not say how few matchings that needed.  That
second question is the one the Chakraborty-Gottesman discussion turns on — a
code reaching full from a single certified matching (plus automorphism gates
that fail to normalize it) is a different phenomenon from one that needs
layers over varying matchings — so the zoo renders a sentence about it, and
that sentence must be backed by published data rather than by a claim in a
commit message.

Run after ``docs/zoo/make_data.py`` (which computes the field itself for new
runs; this script exists to patch a file produced before that wiring, and to
re-run the test cheaply without a full census):

    python scripts/one_block_single_matching.py

Idempotent: rows already carrying the field are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.oneblock import single_matching_fullness

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "zoo" / "zoo_data.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute existing fields")
    parser.add_argument("--budget", type=float, default=300.0, help="per-code seconds")
    arguments = parser.parse_args()

    rows = json.loads(DATA.read_text())
    patched = 0
    for row in rows:
        one_block = row.get("one_block")
        if not isinstance(one_block, dict) or one_block.get("is_full") is not True:
            continue  # only full codes can be full from a single matching
        if "single_matching_full" in one_block and not arguments.force:
            continue
        name = row["name"]
        entry = REGISTRY.get(name)
        if entry is None:
            continue
        code = CSSCode(*entry.build())
        start = time.time()
        verdict = single_matching_fullness(code, name, time_budget_s=arguments.budget)
        one_block.update(verdict)
        one_block["single_matching_seconds"] = round(time.time() - start, 2)
        patched += 1
        print(
            f"{name}: single_matching_full={verdict['single_matching_full']} "
            f"witness={verdict['witness']} tried={verdict['matchings_tried_singly']} "
            f"({one_block['single_matching_seconds']}s)"
        )

    if patched:
        DATA.write_text(json.dumps(rows, indent=1))
    print(f"patched {patched} row(s) in {DATA}")


if __name__ == "__main__":
    main()
