"""Exact worst-case circuit depth over a code's certified one-block layers.

``analyze_one_block`` decides *whether* the layers generate Sp(2k,2); the
counting floor D >= log_G |Sp(2k,2)| then bounds how deep a circuit must be.
That floor is loose, and nothing in the payload says by how much.  This does
the only thing that settles it: breadth-first search over the Cayley graph,
whose last level IS the true worst-case depth (each generator is one
depth-one layer).  Generators are closed under inverse, since the
code-preserving layers on a fixed partition form a group.

Only codes certified FULL qualify -- elsewhere "depth to reach the whole
group" is not a quantity -- and only while the group fits in memory, so this
tops out around k = 3 (|Sp(6,2)| = 1,451,520).  Above that the exact answer
is out of reach and the page says so rather than extrapolating.

    python scripts/one_block_exact_depth.py [--max-order N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.logical.generated import analyze_one_block
from qec_transversal.utils.gf2 import gf2_inverse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from one_block_refresh import policy

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "zoo" / "zoo_data.json"


def _generators_with_inverses(analysis) -> list[np.ndarray]:
    matrices, seen = [], set()
    for generator in analysis.generators:
        for matrix in (generator.matrix, gf2_inverse(generator.matrix)):
            key = matrix.tobytes()
            identity = np.array_equal(matrix, np.eye(matrix.shape[0], dtype=np.uint8))
            if key not in seen and not identity:
                seen.add(key)
                matrices.append(matrix)
    return matrices


def cayley_diameter(generators: list[np.ndarray], dim: int, expected: int) -> tuple[int, int]:
    """(worst-case depth, elements reached) by BFS from the identity."""

    identity = np.eye(dim, dtype=np.uint8)
    depth_of = {identity.tobytes(): 0}
    frontier, depth = [identity], 0
    while frontier:
        depth += 1
        nxt = []
        for element in frontier:
            for generator in generators:
                product = (element @ generator) & 1
                key = product.tobytes()
                if key not in depth_of:
                    depth_of[key] = depth
                    nxt.append(product)
        frontier = nxt
        if len(depth_of) > expected:
            raise RuntimeError("BFS exceeded the expected group order")
    return max(depth_of.values()), len(depth_of)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-order", type=int, default=5_000_000)
    arguments = parser.parse_args()

    rows = json.loads(DATA.read_text())
    for row in rows:
        one_block = row.get("one_block") or {}
        if not one_block.get("is_full") or one_block.get("sp_target", 0) > arguments.max_order:
            continue
        name = row["name"]
        entry = REGISTRY[name]
        budget, cap = policy(entry.n)
        analysis = analyze_one_block(
            CSSCode(*entry.build()), name=name, time_budget_s=budget, involution_cap=cap
        )
        start = time.time()
        depth, reached = cayley_diameter(
            _generators_with_inverses(analysis), 2 * entry.k, one_block["sp_target"]
        )
        assert reached == one_block["sp_target"], f"{name}: BFS reached {reached}"
        assert len(analysis.generators) == one_block["generator_count"], (
            f"{name}: generator count drifted from the census row — refresh it first"
        )
        one_block["exact_depth"] = int(depth)
        one_block["exact_depth_seconds"] = round(time.time() - start, 2)
        print(f"{name:<24} k={row['k']:<3} G={one_block['generator_count']:<4} "
              f"exact depth={depth}  [{time.time()-start:.1f}s]", flush=True)
        DATA.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
