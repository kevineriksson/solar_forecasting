"""Compose the Stage 2 feature table from validated interim data.

Order matters:

  1. Compute clear-sky (cs_ghi, cs_dni, cs_dhi) from the full UTC index.
  2. Compute k_t from observed GHI and cs_ghi (clipped, night-zeroed).
  3. Compute night mask from zenith.
  4. Add lag and rolling-mean features over the FULL contiguous series (so warmup
     rows at the boundary of the dataset are NaN, but warmup never crosses a
     split boundary inappropriately — splits are assigned AFTER features).
  5. Add calendar sin/cos.
  6. Drop rows with NaN features (the leading warmup window only).
  7. Assign split labels using the same SplitManifest as Stage 1.

The output is a dict {split_name: DataFrame}. Each DataFrame has a fixed,
deterministic column order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.ingest.schema import TIMESTAMP_COL
from src.ingest.splits import SplitManifest, assign_splits

from .clearsky import CLEARSKY_COLUMNS, compute_clearsky
from .transforms import (
    add_lags,
    add_rolling_means,
    calendar_features,
    compute_kt,
    night_mask,
)

OBSERVED_COLS = (
    "ghi",
    "dni",
    "dhi",
    "gti",
    "air_temp",
    "wind_speed_100m",
    "zenith",
    "azimuth",
)


@dataclass(frozen=True)
class FeatureConfig:
    site: Mapping[str, object]
    lags_steps: tuple[int, ...]
    rolling_means_steps: tuple[int, ...]
    lagged_variables: tuple[str, ...]
    kt_clip_min: float
    kt_clip_max: float
    night_zenith_threshold_deg: float

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> FeatureConfig:
        feats: dict = params["features"]  # type: ignore[assignment]
        site: dict = params["site"]  # type: ignore[assignment]
        return cls(
            site=site,
            lags_steps=tuple(int(x) for x in feats["lags_steps"]),
            rolling_means_steps=tuple(int(x) for x in feats["rolling_means_steps"]),
            lagged_variables=tuple(str(x) for x in feats["lagged_variables"]),
            kt_clip_min=float(feats["kt_clip_min"]),
            kt_clip_max=float(feats["kt_clip_max"]),
            night_zenith_threshold_deg=float(feats["night_zenith_threshold_deg"]),
        )


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Build the full feature table from the contiguous validated dataset.

    Input `df` must be sorted by TIMESTAMP_COL ascending, UTC, gap-free at the
    configured resolution (already guaranteed by Stage 1).
    """
    _assert_contiguous(df)

    index = pd.DatetimeIndex(df[TIMESTAMP_COL])
    feats = pd.DataFrame({TIMESTAMP_COL: df[TIMESTAMP_COL].to_numpy()}, index=df.index)

    # Observed columns (we keep them — both for downstream targets and as feature
    # source for lags/rolling). Cast to float64 explicitly for determinism.
    for col in OBSERVED_COLS:
        feats[col] = df[col].astype("float64").to_numpy()

    # 1. clear-sky
    cs = compute_clearsky(index, cfg.site).reset_index(drop=True)
    for col in CLEARSKY_COLUMNS:
        feats[col] = cs[col].to_numpy()

    # 2. k_t
    feats["k_t"] = compute_kt(
        feats["ghi"], feats["cs_ghi"], cfg.kt_clip_min, cfg.kt_clip_max
    ).to_numpy()

    # 3. night mask
    feats["is_night"] = night_mask(feats["zenith"], cfg.night_zenith_threshold_deg).to_numpy()

    # 4. lags + rolling means (over the full contiguous series)
    lags_df = add_lags(feats, cfg.lagged_variables, cfg.lags_steps)
    rolls_df = add_rolling_means(feats, cfg.lagged_variables, cfg.rolling_means_steps)

    # 5. calendar
    cal = calendar_features(index).reset_index(drop=True)

    out = pd.concat([feats, lags_df, rolls_df, cal], axis=1)

    # 6. drop the leading warmup window (rows where any lag/roll is NaN).
    warmup = max((*cfg.lags_steps, *cfg.rolling_means_steps))
    out = out.iloc[warmup:].reset_index(drop=True)
    assert not out.isna().any().any(), "unexpected NaN after warmup drop"

    return out


def split_features(feature_df: pd.DataFrame, manifest: SplitManifest) -> dict[str, pd.DataFrame]:
    """Assign features to splits using the same manifest produced by Stage 1.

    Returns a dict in fixed order: train, promo, replay. The warmup drop in
    `build_features` may remove some leading rows from the TRAIN split; that is
    expected and explicit. promo/replay are never truncated because they sit
    deeper in the timeline than the warmup window.
    """
    labels = assign_splits(feature_df, manifest, TIMESTAMP_COL)
    out: dict[str, pd.DataFrame] = {}
    for split in manifest.as_tuple():
        sub = feature_df.loc[labels == split.name].reset_index(drop=True)
        if sub.empty:
            raise AssertionError(f"split {split.name!r} is empty after feature build")
        out[split.name] = sub
    return out


def _assert_contiguous(df: pd.DataFrame) -> None:
    ts = df[TIMESTAMP_COL]
    if not ts.is_monotonic_increasing:
        raise ValueError("interim data must be sorted ascending by timestamp")
    diffs = ts.diff().dropna().unique()
    if len(diffs) != 1:
        raise ValueError(f"interim data is not at uniform resolution; diffs={diffs}")
