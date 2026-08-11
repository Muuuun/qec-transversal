"""Witness export + independent checker round-trip and mutation tests."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.witness import export_strict_witness

CHECKER = Path(__file__).resolve().parent.parent / "tools" / "check_witness.py"


def _check(document, tmp_path) -> bool:
    path = tmp_path / "w.json"
    path.write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(path)], capture_output=True, text=True
    )
    return result.returncode == 0


@pytest.mark.parametrize("name", ["steane", "qrm15", "c4-22", "toric-4", "bb72"])
def test_witness_round_trip_passes_independent_checker(name, tmp_path) -> None:
    document = export_strict_witness(CSSCode(*REGISTRY[name].build()), name)
    assert _check(document, tmp_path)


def test_checker_catches_mutations(tmp_path) -> None:
    base = export_strict_witness(CSSCode(*REGISTRY["steane"].build()), "steane")

    mutations = []
    m = copy.deepcopy(base); m["A_Z"]["kernel"] = ["1111110"]; mutations.append(m)
    m = copy.deepcopy(base); m["A_Z"]["kernel"].append("1010101"); mutations.append(m)
    m = copy.deepcopy(base); m["A_Z"]["constraints"] = m["A_Z"]["constraints"][:2]; mutations.append(m)
    m = copy.deepcopy(base); m["A_Z"]["constraints"][0]["dual"] = "1000000"; mutations.append(m)
    m = copy.deepcopy(base); m["generators"][0]["logical"] = ["10", "01"]; mutations.append(m)
    m = copy.deepcopy(base); m["logical_group"]["order"] = 12; mutations.append(m)
    m = copy.deepcopy(base); m["logical_group"]["elements"] = m["logical_group"]["elements"][:3]; mutations.append(m)
    m = copy.deepcopy(base); m["logical_X"] = ["1000000"]; mutations.append(m)

    for index, mutated in enumerate(mutations):
        assert not _check(mutated, tmp_path), f"mutation {index} not caught"
