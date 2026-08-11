"""Two-fold-transversal (matching sweep) tests."""

import numpy as np
import pytest

from qec_transversal import CSSCode, REGISTRY
from qec_transversal.twofold import two_fold_group


def test_steane_two_fold_is_full_immediately() -> None:
    result = two_fold_group(CSSCode(*REGISTRY["steane"].build()), rounds=10)
    assert result.is_full and result.logical_order == 6


def test_c422_reaches_full_sp4() -> None:
    result = two_fold_group(CSSCode(*REGISTRY["c4-22"].build()), rounds=30)
    assert result.is_full and result.logical_order == 720


def test_plateau_reports_lower_bound_not_fullness() -> None:
    # random matchings on the toric code plateau quickly; the result must be
    # reported as a bound, never as certified fullness
    result = two_fold_group(CSSCode(*REGISTRY["toric-4"].build()), rounds=12, plateau=4)
    assert not result.is_full
    assert result.lower_bound >= 1
