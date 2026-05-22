"""Unit tests for src/ingest/splits.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.schema import TIMESTAMP_COL
from src.ingest.splits import assign_splits, compute_splits
from tests.unit._fixtures import make_canonical_frame

SPLITS_CFG = {
    "promo_months_back_start": 8,
    "promo_months_back_end": 6,
    "replay_months": 6,
}


def test_boundaries_are_strictly_increasing():
    m = compute_splits(
        reference_now=pd.Timestamp("2026-05-10T00:00:00Z"),
        data_first_ts=pd.Timestamp("2007-01-01T00:15:00Z"),
        data_last_ts=pd.Timestamp("2026-05-10T00:00:00Z"),
        splits_cfg=SPLITS_CFG,
    )
    assert m.train.start < m.train.end == m.promo.start < m.promo.end == m.replay.start
    assert m.replay.start < m.replay.end


def test_train_end_is_eight_months_before_reference_now():
    m = compute_splits(
        reference_now=pd.Timestamp("2026-05-10T00:00:00Z"),
        data_first_ts=pd.Timestamp("2007-01-01T00:15:00Z"),
        data_last_ts=pd.Timestamp("2026-05-10T00:00:00Z"),
        splits_cfg=SPLITS_CFG,
    )
    assert m.train.end == pd.Timestamp("2025-09-10T00:00:00Z")
    assert m.promo.end == pd.Timestamp("2025-11-10T00:00:00Z")
    assert m.replay.end == pd.Timestamp("2026-05-10T00:00:00Z")


def test_splits_disjoint_and_cover_input():
    # Build a frame that straddles the promo boundary explicitly.
    df = make_canonical_frame(start="2025-08-01T00:00:00Z", n_rows=96 * 320)  # ~320 days
    m = compute_splits(
        reference_now=pd.Timestamp("2026-05-10T00:00:00Z"),
        data_first_ts=df[TIMESTAMP_COL].iloc[0],
        data_last_ts=df[TIMESTAMP_COL].iloc[-1],
        splits_cfg=SPLITS_CFG,
    )
    assigned = assign_splits(df, m, TIMESTAMP_COL)
    # Every row assigned, no duplicates, all three splits present.
    assert assigned.isna().sum() == 0
    assert set(assigned.unique()) == {"train", "promo", "replay"}


def test_boundary_row_at_promo_end_goes_to_replay_not_promo():
    # Pick reference_now so that promo_end falls exactly on a 15-min grid timestamp.
    ref = pd.Timestamp("2026-05-10T00:00:00Z")
    promo_end = pd.Timestamp("2025-11-10T00:00:00Z")
    # 4 rows: two before, one AT promo_end, one after.
    ts = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-11-09T23:30:00Z"),
            pd.Timestamp("2025-11-09T23:45:00Z"),
            promo_end,
            pd.Timestamp("2025-11-10T00:15:00Z"),
        ]
    )
    df = pd.DataFrame({TIMESTAMP_COL: ts})
    m = compute_splits(
        reference_now=ref,
        data_first_ts=ts[0] - pd.Timedelta(days=400),  # train range exists
        data_last_ts=ref,  # data must reach reference_now for compute_splits to accept
        splits_cfg=SPLITS_CFG,
    )
    assigned = assign_splits(df, m, TIMESTAMP_COL)
    # The row exactly at promo_end belongs to replay (replay is left-inclusive).
    assert assigned.iloc[2] == "replay"
    assert assigned.iloc[1] == "promo"
    assert assigned.iloc[3] == "replay"


def test_reference_now_after_data_last_ts_raises():
    with pytest.raises(ValueError, match="after data_last_ts"):
        compute_splits(
            reference_now=pd.Timestamp("2030-01-01T00:00:00Z"),
            data_first_ts=pd.Timestamp("2007-01-01T00:15:00Z"),
            data_last_ts=pd.Timestamp("2026-05-10T00:00:00Z"),
            splits_cfg=SPLITS_CFG,
        )


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_splits(
            reference_now=pd.Timestamp("2026-05-10T00:00:00"),  # naive
            data_first_ts=pd.Timestamp("2007-01-01T00:15:00Z"),
            data_last_ts=pd.Timestamp("2026-05-10T00:00:00Z"),
            splits_cfg=SPLITS_CFG,
        )
