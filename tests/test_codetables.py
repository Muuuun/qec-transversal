"""codetables.de census: parser + strict-engine checks on cached pages.

No network: everything reads the HTML cached under
docs/zoo/witnesses/codetables/ by scripts/codetables_n7_census.py.
"""

import importlib.util
from pathlib import Path

from qec_transversal.stabilizer import StabilizerCode, analyze_local_clifford

ROOT = Path(__file__).resolve().parents[1]
CACHED_513 = ROOT / "docs" / "zoo" / "witnesses" / "codetables" / "QECC_q4_n5_k1.html"

_spec = importlib.util.spec_from_file_location(
    "codetables_n7_census", ROOT / "scripts" / "codetables_n7_census.py"
)
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


def test_parser_reads_the_cached_five_qubit_page() -> None:
    html = CACHED_513.read_text()
    assert census.parse_d_bounds(html) == [3, 3]
    matrix = census.parse_stabilizer_matrix(html)
    assert matrix is not None and matrix.shape == (4, 10)
    # first served row: [1 0 1 0 1|0 0 1 1 0]
    assert matrix[0].tolist() == [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
    code = StabilizerCode(matrix)
    assert (code.n, code.k) == (5, 1)


def test_cached_five_one_three_strict_group_has_order_three() -> None:
    # the best-known [[5,1,3]] is the perfect code: strict group C3 = <(SH)^5>
    code = StabilizerCode(census.parse_stabilizer_matrix(CACHED_513.read_text()))
    report = analyze_local_clifford(code).to_dict()
    assert report["status"] == "exact" and report["certified"]
    assert report["physical_group_order"] == 3
    assert report["logical_group"]["order"] == 3
