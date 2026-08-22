"""Cross-validate the phi-orbit cut against the enumeration cut."""
import itertools, time
from qec_transversal.codes.registry import REGISTRY
from qec_transversal.codes.css import CSSCode
from qec_transversal.api import _as_stabilizer_code
from qec_transversal.ansatz.partition import partition_units_via_structure
from staircase import partitions_max_cell

bad = 0; n_ok = 0
for name in ["c4-22", "c6-22", "cube-832", "steane"]:
    code = _as_stabilizer_code(CSSCode(*REGISTRY[name].build()))
    for w in (1, 2):
        for p in partitions_max_cell(code.n, w):
            cells = [tuple(c) for c in p]
            a = partition_units_via_structure(code, cells, method="enumeration")
            b = partition_units_via_structure(code, cells, method="phi")
            if a["status"] != "exact":
                continue
            same = (b["status"] == "exact"
                    and a["symplectic_group_order"] == b["symplectic_group_order"]
                    and a["logical_group"].get("order") == b["logical_group"].get("order"))
            if not same:
                bad += 1
                print("MISMATCH", name, cells, a["symplectic_group_order"], a["logical_group"].get("order"),
                      "|", b.get("status"), b.get("symplectic_group_order"),
                      b["logical_group"].get("order"), b.get("detail"))
            else:
                n_ok += 1
    print(f"{name}: {n_ok} agreements so far, {bad} mismatches", flush=True)
print("TOTAL", n_ok, "agree,", bad, "mismatch")
