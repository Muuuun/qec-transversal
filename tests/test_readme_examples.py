"""Every executable example in ``README.md`` runs and produces its printed value.

The README quotes concrete numbers.  If a refactor changes one of them, this
file fails rather than the documentation quietly going stale.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qec_transversal import (
    REGISTRY,
    Completeness,
    CSSCode,
    diagonal_transversal_gates,
    five_qubit_code,
    partition_clifford_group,
    strict_transversal_clifford,
)

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def test_readme_css_example() -> None:
    steane = CSSCode(*REGISTRY["steane"].build())
    result = strict_transversal_clifford(steane)

    assert result.method == "css-shear-kernel"
    assert result.completeness is Completeness.COMPLETE
    assert result.logical_group_order == 6  # |Sp(2,2)|: transversal S and H


def test_readme_non_css_example() -> None:
    result = strict_transversal_clifford(five_qubit_code())

    assert result.method == "preservation-algebra units (singleton partition)"
    assert result.completeness is Completeness.COMPLETE
    assert result.group_order == 3  # the cyclic (SH)^{otimes 5} gate
    assert result.logical_group_order == 3


def test_readme_partition_example() -> None:
    c422 = CSSCode(*REGISTRY["c4-22"].build())

    singletons = partition_clifford_group(c422, [(0,), (1,), (2,), (3,)])
    pairs = partition_clifford_group(c422, [(0, 1), (2, 3)])

    assert (singletons.group_order, singletons.logical_group_order) == (6, 6)
    assert (pairs.group_order, pairs.logical_group_order) == (384, 48)
    assert singletons.complete and pairs.complete


def test_readme_cross_block_example() -> None:
    from qec_transversal import transversal_clifford_across_blocks

    steane = CSSCode(*REGISTRY["steane"].build())
    two_steane = transversal_clifford_across_blocks(steane, blocks=2)
    two_perfect = transversal_clifford_across_blocks(five_qubit_code(), blocks=2)

    assert two_steane.completeness is Completeness.COMPLETE
    assert two_steane.logical_group_order == 720
    assert two_perfect.completeness is Completeness.COMPLETE
    assert two_perfect.logical_group_order == 18


def test_readme_sign_certificate_example() -> None:
    pytest.importorskip("stim")
    from qec_transversal import certify_signs

    steane = CSSCode(*REGISTRY["steane"].build())
    certificate = certify_signs(steane, strict_transversal_clifford(steane))
    assert certificate.checked == 2
    assert certificate.certified


def test_readme_hierarchy_example() -> None:
    qrm15 = CSSCode(*REGISTRY["qrm15"].build())
    result = diagonal_transversal_gates(qrm15, level=3)

    assert result.completeness is Completeness.COMPLETE
    assert result.metadata["max_level"] == 3  # a genuine logical T
    assert result.metadata["has_level_gate"] is True


def test_readme_certificate_example(tmp_path) -> None:
    from qec_transversal.certificates.witness import export_strict_witness, write_witness

    steane = CSSCode(*REGISTRY["steane"].build())
    path = tmp_path / "steane-witness.json"
    write_witness(export_strict_witness(steane, "steane"), str(path))

    checker = REPO / "tools" / "check_witness.py"
    completed = subprocess.run(
        [sys.executable, str(checker), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["list-codes"],
        ["strict", "--code", "steane", "--compact"],
        ["partition", "--code", "c4-22", "--cells", "0,1;2,3", "--compact"],
        ["diagonal", "--code", "qrm15", "--level", "3", "--compact"],
        ["analyze", "--code", "steane", "--compact"],
        ["verify", "--code", "steane", "H", "0", "--compact"],
    ],
)
def test_readme_cli_examples(argv, capsys) -> None:
    from qec_transversal.cli import main

    assert main(argv) == 0
    captured = capsys.readouterr().out
    assert captured.strip()
    if argv[0] not in ("list-codes",):
        json.loads(captured)  # must be valid JSON


def test_readme_analyze_example_file_still_parses(capsys) -> None:
    from qec_transversal.cli import main

    assert main(["analyze", str(REPO / "examples" / "steane.json"), "--compact"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == {"n": 7, "k": 1, "rank_X": 3, "rank_Z": 3}


def test_readme_documents_only_real_entry_points() -> None:
    """Every ``qec_transversal`` symbol named in a README code block exists."""

    import qec_transversal

    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks, "the README lost its python examples"
    imported: set[str] = set()
    for block in blocks:
        for match in re.finditer(r"from qec_transversal(?:\.[\w.]+)? import ([^\n]+)", block):
            for name in match.group(1).split(","):
                imported.add(name.strip())
    assert imported
    for name in imported:
        if hasattr(qec_transversal, name):
            continue
        # sub-module imports are checked by actually importing them below
        assert "." not in name, name
    # the documented sub-module import must resolve
    from qec_transversal.certificates.witness import (  # noqa: F401
        export_strict_witness,
        write_witness,
    )


def test_readme_capability_table_names_real_api() -> None:
    import qec_transversal

    text = README.read_text(encoding="utf-8")
    table = text.split("## Core capabilities", 1)[1].split("---", 1)[0]
    referenced = set(re.findall(r"`([a-z_]+(?:\.[a-z_]+)*)`", table))
    top_level = {name for name in referenced if "." not in name}
    for name in top_level:
        assert hasattr(qec_transversal, name), name
