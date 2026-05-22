"""Unit tests for src/models/cv.py — rolling-origin folds."""

from __future__ import annotations

import pytest

from src.models.cv import rolling_origin_folds


def _cfg(n_splits: int = 5, test_size: int = 100, gap: int = 4) -> dict:
    return {
        "scheme": "rolling_origin",
        "n_splits": n_splits,
        "test_size_steps": test_size,
        "gap_steps": gap,
    }


def test_basic_shape_and_chronological_order():
    folds = rolling_origin_folds(n_rows=1000, cv_cfg=_cfg())
    assert [f.index for f in folds] == [0, 1, 2, 3, 4]
    # Chronological: each fold's val window starts no earlier than the prior one's.
    for prev, curr in zip(folds[:-1], folds[1:], strict=True):
        assert curr.val_start >= prev.val_end  # disjoint val windows


def test_val_windows_disjoint_and_sized_correctly():
    folds = rolling_origin_folds(n_rows=1000, cv_cfg=_cfg(n_splits=5, test_size=100, gap=4))
    for f in folds:
        assert f.val_end - f.val_start == 100
    # Last fold val_end is exactly n_rows.
    assert folds[-1].val_end == 1000


def test_gap_is_respected():
    folds = rolling_origin_folds(n_rows=1000, cv_cfg=_cfg(gap=4))
    for f in folds:
        assert f.val_start - f.train_end == 4


def test_train_window_is_expanding():
    folds = rolling_origin_folds(n_rows=1000, cv_cfg=_cfg())
    sizes = [f.train_end - f.train_start for f in folds]
    assert sizes == sorted(sizes)  # non-decreasing
    assert all(f.train_start == 0 for f in folds)


def test_raises_if_doesnt_fit():
    # 5 folds × 100 + gap=4 would need at least 504 rows of history.
    with pytest.raises(ValueError, match="does not fit"):
        rolling_origin_folds(n_rows=400, cv_cfg=_cfg())


def test_rejects_bad_scheme():
    with pytest.raises(ValueError, match="unsupported cv scheme"):
        rolling_origin_folds(n_rows=1000, cv_cfg={**_cfg(), "scheme": "kfold"})


def test_realistic_params_fit_real_train_size():
    # Mirrors the params.yaml values against the actual train.parquet row count.
    folds = rolling_origin_folds(
        n_rows=655_379,
        cv_cfg={
            "scheme": "rolling_origin",
            "n_splits": 5,
            "test_size_steps": 2880,
            "gap_steps": 4,
        },
    )
    assert len(folds) == 5
    assert folds[-1].val_end == 655_379
    assert folds[0].train_end > 600_000
