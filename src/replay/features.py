"""As-of-t feature access for the replay client (Stage 6 / T11).

The pre-computed `data/features/replay.parquet` already contains, for every
timestamp in the replay window, the exact Stage 2 feature row the model would
have seen at that timestamp — because Stage 2 only uses lags, rolling means and
calendar terms, all of which reference data at or before each row's own
timestamp. So at replay-time we don't need to re-run the pipeline; we just need
to *prove* that the row at `t` depends only on observations at or before `t`,
and refuse any access that would imply otherwise.

This module supplies three pieces:

  * ``LeakageError`` — raised when the caller attempts to build features at
    time ``t`` from a history that contains rows after ``t``.

  * ``build_features_as_of(history_df, t, cfg)`` — the explicit "thin wrapper
    around Stage 2" required by T11: truncates ``history_df`` to rows where
    ``timestamp <= t``, runs the same `src.features.pipeline.build_features`
    code, and returns the final row as a dict. The leakage guard is enforced
    here. The runtime client never calls this (it's O(N) per call); it exists
    for the equivalence + leakage tests.

  * ``ReplaySource`` — wraps the pre-computed replay feature table with
    guarded ``feature_payload(t)`` and ``truth_at(t, horizon_steps)``
    accessors. The replay loop uses this in production.

Persistence-specific clear-sky-at-horizon columns (``cs_<target>_h<label>``)
are computed on demand via :func:`enrich_for_persistence` so the persistence
serving handle has the inputs it expects. We compute those via pvlib using the
horizon timestamp — the horizon timestamp is **time arithmetic**, not a future
observation, so it does not constitute leakage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.features.clearsky import CLEARSKY_COLUMNS, compute_clearsky
from src.features.pipeline import FeatureConfig, build_features
from src.ingest.schema import TIMESTAMP_COL


class LeakageError(RuntimeError):
    """Raised when a feature build would consume data from timestamps > t."""


def build_features_as_of(
    history_df: pd.DataFrame,
    t: pd.Timestamp,
    cfg: FeatureConfig,
) -> dict[str, float | int]:
    """Build the Stage 2 feature row for time ``t`` from raw history.

    ``history_df`` must have the canonical Stage 1 schema (columns from
    ``src.ingest.schema.NUMERIC_COLS`` plus ``period_end``). The function will
    refuse — by raising :class:`LeakageError` — if ``history_df`` contains any
    row with ``period_end > t``: callers are expected to pre-truncate so the
    intent ("look at history up to and including t") is explicit.

    The returned dict represents the feature row at ``t``. Keys match the
    columns of :func:`src.features.pipeline.build_features`.

    This function is intended for tests and one-off equivalence checks, not
    for the per-timestep replay loop (which uses :class:`ReplaySource`).
    """
    t = _as_utc(t)
    ts = history_df[TIMESTAMP_COL]
    if (ts > t).any():
        n_future = int((ts > t).sum())
        raise LeakageError(
            f"history_df contains {n_future} row(s) with timestamp > {t.isoformat()}; "
            "feature build at t may only see data <= t"
        )

    truncated = history_df.loc[ts <= t].reset_index(drop=True)
    if truncated.empty:
        raise ValueError(f"history_df has no rows at or before {t.isoformat()}")

    features = build_features(truncated, cfg)
    # The row whose period_end == t is the most recent; build_features drops
    # the leading warmup window but always retains the trailing row.
    final_ts = features[TIMESTAMP_COL].iloc[-1]
    if final_ts != t:
        raise AssertionError(
            f"feature build dropped the requested timestamp {t.isoformat()} "
            f"(last row {final_ts.isoformat()}); not enough warmup history?"
        )
    return _row_to_dict(features.iloc[-1])


_TARGET_TO_CS_COL: dict[str, str] = {"ghi": "cs_ghi", "dni": "cs_dni", "dhi": "cs_dhi"}


@dataclass(frozen=True)
class GroundTruth:
    """A single ground-truth observation for a (target, horizon) pair."""

    target: str
    horizon_label: str
    horizon_steps: int
    target_timestamp: pd.Timestamp
    value: float
    clearsky_at_horizon: float | None = None


class ReplaySource:
    """Indexed access to a pre-computed replay feature table.

    Reads ``data/features/replay.parquet`` (or any Stage 2 output frame), wraps
    it as a timestamp-keyed table, and exposes:

      * :meth:`timestamps` — the ordered list of in-window timestamps the loop
        should iterate over (excludes the last ``max_horizon`` rows so every
        prediction can be scored against an in-window truth).
      * :meth:`feature_payload` — the dict to POST as the ``features`` field.
      * :meth:`truth_at` — the observed irradiance ``horizon_steps`` ahead of
        ``t`` (used to compute residuals).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        targets: tuple[str, ...],
        horizons_steps: tuple[int, ...],
        horizon_labels: tuple[str, ...],
    ) -> None:
        if TIMESTAMP_COL not in features_df.columns:
            raise ValueError(f"features dataframe missing {TIMESTAMP_COL!r}")
        if not features_df[TIMESTAMP_COL].is_monotonic_increasing:
            raise ValueError("features dataframe must be sorted ascending by timestamp")
        if len(horizons_steps) != len(horizon_labels):
            raise ValueError("horizons_steps and horizon_labels must be the same length")
        for tgt in targets:
            if tgt not in features_df.columns:
                raise ValueError(f"features dataframe missing target column {tgt!r}")

        self._df = features_df.reset_index(drop=True)
        self._targets = tuple(targets)
        self._horizons_steps = tuple(int(s) for s in horizons_steps)
        self._horizon_labels = tuple(horizon_labels)
        self._index = pd.Series(self._df.index.values, index=self._df[TIMESTAMP_COL])
        # Pre-strip the timestamp column from each row dict — JSON-incompatible.
        # We expose copies so callers cannot mutate cached rows.
        self._feature_cols = [c for c in self._df.columns if c != TIMESTAMP_COL]

    # -- iteration ----------------------------------------------------------

    @property
    def max_horizon_steps(self) -> int:
        return max(self._horizons_steps)

    def window_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (
            pd.Timestamp(self._df[TIMESTAMP_COL].iloc[0]),
            pd.Timestamp(self._df[TIMESTAMP_COL].iloc[-1]),
        )

    def timestamps(
        self,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> list[pd.Timestamp]:
        """Return timestamps in [start, end] that have a valid ground-truth lookup.

        The trailing ``max_horizon_steps`` rows of the underlying frame are
        excluded so every iteration can score against an in-window truth.
        ``start`` / ``end`` default to the full replay window.
        """
        last_scoreable_idx = len(self._df) - self.max_horizon_steps - 1
        if last_scoreable_idx < 0:
            return []
        scoreable_ts = self._df[TIMESTAMP_COL].iloc[: last_scoreable_idx + 1]
        if start is not None:
            scoreable_ts = scoreable_ts[scoreable_ts >= _as_utc(start)]
        if end is not None:
            scoreable_ts = scoreable_ts[scoreable_ts <= _as_utc(end)]
        return [pd.Timestamp(t) for t in scoreable_ts.to_numpy()]

    # -- feature access -----------------------------------------------------

    def _row_index(self, t: pd.Timestamp) -> int:
        try:
            idx = int(self._index.loc[_as_utc(t)])
        except KeyError as exc:
            raise KeyError(f"timestamp {t} not present in replay table") from exc
        return idx

    def feature_payload(self, t: pd.Timestamp) -> dict[str, float]:
        """Return the feature dict at ``t`` (Stage 2 row, minus the timestamp).

        Per the as-of-t invariant: only columns from the row at ``t`` are read.
        Lags, rolling means and calendar terms were all computed by Stage 2
        from observations at or before each row's own timestamp, so by
        construction no value in this dict depends on data after ``t``.
        """
        idx = self._row_index(t)
        row = self._df.iloc[idx]
        return {c: _to_jsonable(row[c]) for c in self._feature_cols}

    def feature_sequence(self, t: pd.Timestamp, sequence_length: int) -> list[dict[str, float]]:
        """Return the trailing ``sequence_length`` feature rows ending at ``t``.

        Used for LSTM serving (oldest-first). Raises if the source does not
        contain ``sequence_length`` rows at or before ``t``.
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        end_idx = self._row_index(t)
        start_idx = end_idx - sequence_length + 1
        if start_idx < 0:
            raise ValueError(
                f"need {sequence_length} rows of history ending at {t}; "
                f"only {end_idx + 1} available"
            )
        sub = self._df.iloc[start_idx : end_idx + 1]
        return [{c: _to_jsonable(r[c]) for c in self._feature_cols} for _, r in sub.iterrows()]

    def feature_timestamps(self, t: pd.Timestamp, sequence_length: int) -> list[pd.Timestamp]:
        end_idx = self._row_index(t)
        start_idx = end_idx - sequence_length + 1
        ts = self._df[TIMESTAMP_COL].iloc[start_idx : end_idx + 1]
        return [pd.Timestamp(x) for x in ts.to_numpy()]

    # -- ground truth -------------------------------------------------------

    def truths_at(self, t: pd.Timestamp, horizon_steps: int) -> list[GroundTruth] | None:
        """Return per-target truths at ``t + horizon_steps``, or ``None`` if absent.

        ``horizon_steps`` is the Stage-2 horizon index (1 step = 15 minutes).
        Returns one :class:`GroundTruth` per configured target.
        """
        base_idx = self._row_index(t)
        future_idx = base_idx + int(horizon_steps)
        if future_idx >= len(self._df):
            return None
        future_row = self._df.iloc[future_idx]
        future_ts = pd.Timestamp(future_row[TIMESTAMP_COL])
        label = self._label_for_steps(horizon_steps)
        results = []
        for tgt in self._targets:
            cs_col = _TARGET_TO_CS_COL.get(tgt)
            clearsky = (
                float(future_row[cs_col])
                if cs_col is not None and cs_col in future_row.index
                else None
            )
            results.append(
                GroundTruth(
                    target=tgt,
                    horizon_label=label,
                    horizon_steps=int(horizon_steps),
                    target_timestamp=future_ts,
                    value=float(future_row[tgt]),
                    clearsky_at_horizon=clearsky,
                )
            )
        return results

    def _label_for_steps(self, horizon_steps: int) -> str:
        for steps, label in zip(self._horizons_steps, self._horizon_labels, strict=True):
            if steps == int(horizon_steps):
                return label
        raise KeyError(f"no horizon_label configured for {horizon_steps} steps")


# ---------------------------------------------------------------------------
# Persistence helper: clear-sky at horizon timestamps
# ---------------------------------------------------------------------------


_DEFAULT_RESOLUTION = pd.Timedelta(minutes=15)


def enrich_for_persistence(
    features: dict[str, float],
    t: pd.Timestamp,
    site: Mapping[str, object],
    horizons_steps: tuple[int, ...],
    horizon_labels: tuple[str, ...],
    resolution: pd.Timedelta | None = None,
) -> dict[str, float]:
    """Add ``cs_<target>_h<label>`` columns required by the persistence handle.

    These are clear-sky irradiance forecasts at ``t + horizon_steps * resolution``
    for each target. Clearsky is a deterministic function of timestamp + site;
    computing it for a future timestamp is **not** an observation leak — it's
    the same trick the trainer used at fit time.
    """
    t = _as_utc(t)
    step = resolution if resolution is not None else _DEFAULT_RESOLUTION
    horizon_index = pd.DatetimeIndex([t + step * int(s) for s in horizons_steps], tz="UTC")
    cs = compute_clearsky(horizon_index, site)
    out = dict(features)
    for j, label in enumerate(horizon_labels):
        for cs_col in CLEARSKY_COLUMNS:
            # cs_col is "cs_<target>"; we want "cs_<target>_h<label>".
            out[f"{cs_col}_h{label}"] = float(cs[cs_col].iloc[j])
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_utc(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {value!r} must be UTC-aware")
    return ts.tz_convert("UTC")


def _row_to_dict(row: pd.Series) -> dict[str, float | int]:
    """Strip the timestamp column and return remaining fields as JSON scalars."""
    out: dict[str, float | int] = {}
    for col, val in row.items():
        if col == TIMESTAMP_COL:
            continue
        out[str(col)] = _to_jsonable(val)
    return out


def _to_jsonable(val: object) -> float | int:
    """Coerce a numeric pandas value to a plain Python int/float for JSON."""
    if isinstance(val, bool):
        # bool is a subclass of int; keep as int for the wire format
        return int(val)
    if hasattr(val, "item"):
        try:
            v = val.item()  # type: ignore[attr-defined]
        except (ValueError, TypeError):
            v = val
    else:
        v = val
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    return float(v)  # type: ignore[arg-type]
