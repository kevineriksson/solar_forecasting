"""Compute train / promotion / replay split boundaries from params.yaml.

Splits are derived from `reference_now` (a fixed UTC timestamp in params.yaml),
never from `datetime.now()`. Boundary inclusivity is explicit:

    train  : [data_first_ts,  train_end )   left-inclusive, right-exclusive
    promo  : [train_end,      promo_end )   left-inclusive, right-exclusive
    replay : [promo_end,      data_last_ts] left-inclusive, right-inclusive

Each boundary timestamp belongs to exactly one split (the later one), except the
final timestamp which belongs to replay. Splits are disjoint and cover the full
input range.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from dateutil.relativedelta import relativedelta

SPLIT_NAMES = ("train", "promo", "replay")


@dataclass(frozen=True)
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    inclusive: str  # "[)" or "[]"

    def contains(self, ts: pd.Series) -> pd.Series:
        if self.inclusive == "[)":
            return (ts >= self.start) & (ts < self.end)
        if self.inclusive == "[]":
            return (ts >= self.start) & (ts <= self.end)
        raise ValueError(f"unsupported inclusivity {self.inclusive!r}")

    def to_dict(self, n_rows: int) -> dict:
        return {
            "start": self.start.isoformat().replace("+00:00", "Z"),
            "end": self.end.isoformat().replace("+00:00", "Z"),
            "inclusive": self.inclusive,
            "n_rows": int(n_rows),
        }


@dataclass(frozen=True)
class SplitManifest:
    reference_now: pd.Timestamp
    data_first_ts: pd.Timestamp
    data_last_ts: pd.Timestamp
    train: Split
    promo: Split
    replay: Split

    def as_tuple(self) -> tuple[Split, Split, Split]:
        return (self.train, self.promo, self.replay)


def compute_splits(
    reference_now: pd.Timestamp,
    data_first_ts: pd.Timestamp,
    data_last_ts: pd.Timestamp,
    splits_cfg: Mapping[str, object],
) -> SplitManifest:
    """Build a SplitManifest from a reference timestamp and the data range."""
    reference_now = _as_utc_ts(reference_now)
    data_first_ts = _as_utc_ts(data_first_ts)
    data_last_ts = _as_utc_ts(data_last_ts)

    if reference_now <= data_first_ts:
        raise ValueError("reference_now must be after data_first_ts")
    if reference_now > data_last_ts:
        # The reference is allowed to be earlier than data_last_ts, but cannot be
        # after the data ends — that would mean we're asking for splits we cannot fill.
        raise ValueError(
            f"reference_now ({reference_now}) is after data_last_ts ({data_last_ts}); "
            "raw data does not reach the requested replay window end"
        )

    promo_back_start = int(splits_cfg["promo_months_back_start"])  # type: ignore[call-overload]
    promo_back_end = int(splits_cfg["promo_months_back_end"])  # type: ignore[call-overload]
    replay_months = int(splits_cfg["replay_months"])  # type: ignore[call-overload]

    if promo_back_start <= promo_back_end:
        raise ValueError("promo_months_back_start must be > promo_months_back_end")
    if replay_months != promo_back_end:
        # Encodes the invariant that the replay window starts where the promo window ends.
        raise ValueError(
            "replay_months must equal promo_months_back_end so promo and replay are adjacent"
        )

    train_end = reference_now - relativedelta(months=promo_back_start)
    promo_end = reference_now - relativedelta(months=promo_back_end)
    replay_end = data_last_ts  # data may end exactly at reference_now or earlier

    if not (data_first_ts < train_end < promo_end <= replay_end):
        raise ValueError(
            f"computed boundaries are not strictly increasing within data range: "
            f"first={data_first_ts}, train_end={train_end}, "
            f"promo_end={promo_end}, last={replay_end}"
        )

    return SplitManifest(
        reference_now=reference_now,
        data_first_ts=data_first_ts,
        data_last_ts=data_last_ts,
        train=Split("train", data_first_ts, train_end, "[)"),
        promo=Split("promo", train_end, promo_end, "[)"),
        replay=Split("replay", promo_end, replay_end, "[]"),
    )


def assign_splits(df: pd.DataFrame, manifest: SplitManifest, ts_col: str) -> pd.Series:
    """Return a Series of split names aligned to df.index. Asserts disjoint coverage."""
    ts = df[ts_col]
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    for split in manifest.as_tuple():
        mask = split.contains(ts)
        if (out[mask].notna()).any():
            raise AssertionError(f"split {split.name!r} overlaps another split")
        out[mask] = split.name
    if out.isna().any():
        n_unassigned = int(out.isna().sum())
        raise AssertionError(
            f"{n_unassigned} row(s) did not fall into any split; coverage incomplete"
        )
    return out


def _as_utc_ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {value!r} must be timezone-aware")
    return ts.tz_convert("UTC")
