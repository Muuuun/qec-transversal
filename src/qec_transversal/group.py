"""Small exact matrix-group utilities for logical images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import BinaryMatrix


def symplectic_group_order(logical_qubits: int) -> int:
    """Return ``|Sp(2k, 2)|``."""

    if logical_qubits < 0:
        raise ValueError("logical_qubits must be non-negative")
    order = 2 ** (logical_qubits * logical_qubits)
    for index in range(1, logical_qubits + 1):
        order *= 4**index - 1
    return order


@dataclass(frozen=True)
class GroupOrder:
    exact: bool
    order: int | None
    lower_bound: int


def generated_group_order(generators: list[BinaryMatrix], *, cap: int = 100_000) -> GroupOrder:
    """Compute an exact group order by closure, stopping after ``cap`` elements.

    This is deliberately a small-``k`` fallback.  GAP/MeatAxe integration is a
    later backend; a truncated result is reported honestly as a lower bound.
    """

    if cap < 1:
        raise ValueError("cap must be positive")
    if generators:
        shape = generators[0].shape
        if shape[0] != shape[1] or any(generator.shape != shape for generator in generators):
            raise ValueError("generators must be square matrices of the same size")
        size = shape[0]
    else:
        size = 0

    identity = np.eye(size, dtype=np.uint8)
    seen: set[bytes] = {identity.tobytes()}
    queue = [identity]
    cursor = 0
    while cursor < len(queue):
        element = queue[cursor]
        cursor += 1
        for generator in generators:
            product = (element @ generator) & 1
            key = product.tobytes()
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > cap:
                return GroupOrder(exact=False, order=None, lower_bound=len(seen))
            queue.append(product)
    return GroupOrder(exact=True, order=len(seen), lower_bound=len(seen))

