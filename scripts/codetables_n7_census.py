"""External-validation census against codetables.de (Grassl's tables).

Fetches the best-known additive quantum code [[n,k]] for q=4, n=3..7,
k=0..n-1 — 25 pages — from Markus Grassl's bounds tables, caches the raw
HTML under docs/zoo/witnesses/codetables/ (fetch once, reuse forever),
parses the stabilizer matrix each page serves in its PRE block (the
``(X | Z)`` convention StabilizerCode accepts directly), and runs

  * analyze_local_clifford — the complete strict-transversal engine,
  * axis_frame_group at levels 3 and 4 — exhaustive 3^n sweeps at n <= 7,
  * analyze_monomial when python-igraph is available (import guarded).

Emits docs/zoo/codetables_census.json, one row per [[n,k]] pair.

Run from the repo root with the project installed:
    python scripts/codetables_n7_census.py
"""

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from qec_transversal.axes import AxisFrameResult, axis_frame_group
from qec_transversal.stabilizer import StabilizerCode, analyze_local_clifford

HERE = Path(__file__).resolve().parent
ZOO = HERE.parent / "docs" / "zoo"
CACHE = ZOO / "witnesses" / "codetables"
OUT = ZOO / "codetables_census.json"
BASE = "https://codetables.de/QECC.php?q=4&n={n}&k={k}"
RETRIEVED = "2026-08-12"


def cache_path(n: int, k: int) -> Path:
    return CACHE / f"QECC_q4_n{n}_k{k}.html"


def fetch_page(n: int, k: int) -> str:
    """The [[n,k]] page HTML — from the local cache, else one polite fetch."""

    path = cache_path(n, k)
    if path.exists():
        return path.read_text()
    url = BASE.format(n=n, k=k)
    with urllib.request.urlopen(url, timeout=30) as response:
        html = response.read().decode("utf-8", "replace")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    time.sleep(0.4)  # be polite to codetables.de
    return html


def parse_d_bounds(html: str) -> list:
    """``[lower, upper]`` distance bounds from the page's bounds table."""

    lower = re.search(r"lower bound:</TD><TD>(\d+)", html)
    upper = re.search(r"upper bound:</TD><TD>(\d+)", html)
    return [
        int(lower.group(1)) if lower else None,
        int(upper.group(1)) if upper else None,
    ]


def parse_stabilizer_matrix(html: str):
    """The ``(X | Z)`` stabilizer matrix from the PRE block, or None.

    Rows look like ``[1 0 1 0 1|0 0 1 1 0]``; the X and Z halves are
    concatenated into one ``r x 2n`` uint8 matrix.
    """

    match = re.search(
        r"stabilizer matrix:\s*(.*?)(?:last modified|</PRE>)", html, re.DOTALL
    )
    if not match:
        return None
    rows = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]")):
            continue
        body = line[1:-1]
        if "|" not in body:
            return None
        x_part, z_part = body.split("|")
        x = [int(c) for c in x_part.split()]
        z = [int(c) for c in z_part.split()]
        rows.append(x + z)
    return np.asarray(rows, dtype=np.uint8) if rows else None


def frame_summary(result: AxisFrameResult) -> dict:
    """Compact record of one axis-frame sweep.

    ``all_complete`` covers only the nontrivial frames found: a frame
    whose conjugated code is not CSS uses the sound general solver, so an
    *empty* result there is a sound-subgroup statement, not a
    completeness certificate.
    """

    return {
        "frames_tested": int(result.frames_tested),
        "exhaustive": bool(result.exhaustive),
        "nontrivial_count": len(result.nontrivial_frames),
        "all_complete": all(c for _, _, c in result.nontrivial_frames),
    }


def monomial_group_order(code: StabilizerCode):
    """Exact monomial (permutation x local-Clifford) group order, or None
    when python-igraph is unavailable or the analysis fails."""

    try:
        from qec_transversal.monomial import analyze_monomial

        return int(analyze_monomial(code).group_order)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001 - record nothing rather than abort
        return None


def census_row(n: int, k: int) -> dict:
    url = BASE.format(n=n, k=k)
    html = fetch_page(n, k)
    row = {
        "n": n,
        "k": k,
        "d_bounds": parse_d_bounds(html),
        "source_url": url,
        "retrieved": RETRIEVED,
    }
    if "currently not available" in html:
        row["status"] = "unavailable"
        return row
    matrix = parse_stabilizer_matrix(html)
    if matrix is None:
        row["status"] = "no_stabilizer_matrix"
        return row
    code = StabilizerCode(matrix)
    assert code.n == n and code.k == k, f"page [[{n},{k}]] parsed as [[{code.n},{code.k}]]"
    row["css_rows"] = bool(
        all(not (r[:n].any() and r[n:].any()) for r in matrix)
    )
    report = analyze_local_clifford(code).to_dict()
    row.update(
        {
            "status": report["status"],
            "algebra_dim": report["algebra_dimension"],
            "physical_order": report["physical_group_order"],
            "logical_order": report["logical_group"]["order"],
            "certified": report["certified"],
            "frames_l3": frame_summary(axis_frame_group(code, level=3)),
            "frames_l4": frame_summary(axis_frame_group(code, level=4)),
            "monomial_order": monomial_group_order(code),
        }
    )
    return row


def main() -> None:
    out = []
    for n in range(3, 8):
        for k in range(n):
            start = time.time()
            row = census_row(n, k)
            out.append(row)
            print(
                f"[[{n},{k}]] d={row['d_bounds']} status={row.get('status')} "
                f"order={row.get('logical_order')} "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
