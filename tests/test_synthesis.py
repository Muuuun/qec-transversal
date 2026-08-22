"""Constructive verify_logical_gate tests + P0 semantics regressions."""

import gzip
import json
from pathlib import Path

import numpy as np

from qec_transversal import REGISTRY, CSSCode
from qec_transversal.ansatz.partition import PartitionCliffordAnalysis
from qec_transversal.ansatz.strict import LocalCliffordAnalysis
from qec_transversal.codes.stabilizer import StabilizerCode
from qec_transversal.logical.synthesis import verify_logical_gate

WITNESSES = Path(__file__).resolve().parent.parent / "docs" / "zoo" / "witnesses"


def test_capped_enumeration_is_never_silently_negative(monkeypatch) -> None:
    css = CSSCode(*REGISTRY["steane"].build())
    h = np.vstack(
        [
            np.hstack([css.c_x, np.zeros_like(css.c_x)]),
            np.hstack([np.zeros_like(css.c_z), css.c_z]),
        ]
    )
    code = StabilizerCode(h)

    # with the structured unit-group route available, a tiny enumeration cap
    # still yields an EXACT result (the correct order, not a phantom one)
    structured = LocalCliffordAnalysis(code, dim_cap=1).to_dict()
    assert structured["status"] == "exact"
    assert structured["physical_group_order"] == 6
    assert structured["logical_group"]["order"] == 6

    # when BOTH routes are unavailable, the report must be honestly unknown -
    # never certified, never an exact trivial group
    monkeypatch.setattr(
        LocalCliffordAnalysis, "_try_structured_route", lambda self: None
    )
    capped = LocalCliffordAnalysis(code, dim_cap=1).to_dict()
    assert capped["status"] == "unknown"
    assert capped["certified"] is False
    assert capped["logical_group"]["computed"] is False
    assert capped["logical_group"]["order"] is None

    partition = PartitionCliffordAnalysis(
        code, [(i,) for i in range(7)], dim_cap=1
    ).to_dict()
    assert partition["status"] == "unknown"
    assert partition["certified"] is False

    complete = LocalCliffordAnalysis(code).to_dict()
    assert complete["status"] == "exact" and complete["certified"] is True


def test_named_targets() -> None:
    code = CSSCode(*REGISTRY["steane"].build())
    for gate in ("S", "H", "SQRT_X"):
        result = verify_logical_gate(code, gate, 0)
        assert result.found and result.verified
    assert not verify_logical_gate(CSSCode(*REGISTRY["toric-4"].build()), "S", 0).found
    assert not verify_logical_gate(CSSCode(*REGISTRY["qrm15"].build()), "H", 0).found
    cz = verify_logical_gate(CSSCode(*REGISTRY["c4-22"].build()), "CZ", 0, 1)
    assert cz.found and cz.verified


def test_synthesis_matches_witness_group_exactly() -> None:
    for name in ("steane", "c4-22", "qrm15"):
        code = CSSCode(*REGISTRY[name].build())
        doc = json.loads(gzip.open(WITNESSES / f"{name}.json.gz").read())
        elements = []
        for rows in doc["logical_group"]["elements"]:
            m = np.array([[int(c) for c in row] for row in rows], dtype=np.uint8)
            elements.append(m)
            result = verify_logical_gate(code, m)
            assert result.found and result.verified
        # a symplectic matrix outside the group must be refused
        k = code.k
        if k == 1:
            outside = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            if not any(np.array_equal(outside, e) for e in elements):
                assert not verify_logical_gate(code, outside).found
