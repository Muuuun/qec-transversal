"""Closed-form orders of the classical matrix groups over finite fields.

Used as the *target* against which a randomly found generating set is
certified: a Schreier-Sims chain that reproduces ``|GL(d, q)|`` exactly proves
the generators really do generate the whole block unit group, which is what
turns the Wedderburn split into a certified order for ``A^x``.
"""

from __future__ import annotations


def _gl_order(d: int, q: int) -> int:
    """``|GL(d, q)| = prod_{i=0}^{d-1} (q^d - q^i)``."""

    order = 1
    for i in range(d):
        order *= q**d - q**i
    return order


gl_order = _gl_order
