"""CG's own tightness example: k blocks of Steane, cells = corresponding qubits.

If the bound is tight, the width-k partition of [[7k,k,3]] must reach Sp(2k,2).
"""
import time
import numpy as np
from scipy.linalg import block_diag
from qec_transversal.codes.registry import REGISTRY
from qec_transversal.codes.css import CSSCode
from qec_transversal.api import partition_clifford_group, Completeness
from qec_transversal.utils.symplectic import symplectic_group_order

hx, hz = REGISTRY["steane"].build()
hx = np.asarray(hx, dtype=np.uint8); hz = np.asarray(hz, dtype=np.uint8)
for blocks in (2, 3):
    HX = block_diag(*[hx] * blocks).astype(np.uint8)
    HZ = block_diag(*[hz] * blocks).astype(np.uint8)
    code = CSSCode(HX, HZ)
    cells = [tuple(7 * b + i for b in range(blocks)) for i in range(7)]
    target = symplectic_group_order(code.k)
    t0 = time.time()
    r = partition_clifford_group(code, cells)
    print(f"[[{code.n},{code.k},3]] {blocks} Steane blocks, cells of size {blocks} (= k): "
          f"{r.completeness.value} |G|={r.group_order} logical={r.logical_group_order}"
          f"/{target} {'FULL' if r.logical_group_order == target else 'SHORT'} "
          f"{time.time()-t0:.1f}s", flush=True)
