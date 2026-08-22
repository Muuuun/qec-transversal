"""Backend-neutral computational primitives shared by every solver.

Nothing in this subpackage knows about quantum codes: it is exact linear
algebra over GF(2), the binary symplectic space, permutation groups,
GF(2)[x] polynomial arithmetic, and module elimination over Z_{2^L}.
"""
