"""Smart persistence baseline on the clear-sky index.

Forecast rule, per target y in {ghi, dni, dhi} and horizon h steps:

    k_y(t)     = y(t)   / cs_y(t)      with k_y=0 where cs_y(t)   <= CS_EPSILON
    y_hat(t+h) = k_y(t) * cs_y(t+h)    with   0  where cs_y(t+h)  <= CS_EPSILON

The clip range matches `features.kt_clip_min/max` in params.yaml so the
implicit assumption (clear-sky index in [0, 1.5]) is identical to what the
T3 features use for k_t.

This is "smart persistence on k": we persist the *clear-sky index*, not the
raw irradiance. It is the standard solar-forecasting baseline and the
denominator of every skill-score in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.transforms import CS_EPSILON, compute_kt

TARGET_TO_CS_COL = {
    "ghi": "cs_ghi",
    "dni": "cs_dni",
    "dhi": "cs_dhi",
}


@dataclass(frozen=True)
class PersistenceConfig:
    kt_clip_min: float
    kt_clip_max: float

    @classmethod
    def from_params(cls, params: dict) -> PersistenceConfig:
        feats = params["features"]
        return cls(
            kt_clip_min=float(feats["kt_clip_min"]),
            kt_clip_max=float(feats["kt_clip_max"]),
        )


def persistence_forecast(
    df: pd.DataFrame,
    target: str,
    horizon_steps: int,
    cfg: PersistenceConfig,
) -> pd.Series:
    """Return predictions y_hat valid at the SAME index rows as `df`.

    Interpretation:
        At each row t, the value at position t in the returned Series is the
        forecast made at time `t - h` for time `t`. Equivalently, we predict
        the target value at row t using the clear-sky index observed h rows
        earlier.

    This makes alignment with ground truth trivial: just pair `y_hat[t]`
    with `df[target][t]`.

    Rows where t < h cannot be predicted (no history); those positions are NaN
    in the output. Callers should drop them before scoring.
    """
    if target not in TARGET_TO_CS_COL:
        raise ValueError(f"unknown target {target!r}; expected one of {list(TARGET_TO_CS_COL)}")
    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be > 0; got {horizon_steps}")

    cs_col = TARGET_TO_CS_COL[target]
    y = df[target].astype("float64")
    cs = df[cs_col].astype("float64")

    # k_y(t - h) computed once over the whole frame, then shifted forward by h.
    kt_now = compute_kt(y, cs, cfg.kt_clip_min, cfg.kt_clip_max)
    kt_at_origin = kt_now.shift(horizon_steps)  # value from h rows ago

    pred = kt_at_origin * cs  # cs here is cs_y at row t (the forecast valid time)

    # Where current cs_y is effectively zero (night at valid time), force pred to 0
    # rather than carrying tiny floating-point noise.
    night_now = cs <= CS_EPSILON
    pred = pred.where(~night_now, 0.0)

    # Cold-start rows (t < h) remain NaN intentionally.
    pred[:horizon_steps] = np.nan

    pred.name = f"{target}_persistence_h{horizon_steps}"
    return pred.astype("float64")


def score_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """MAE + RMSE on the overlap of non-NaN rows."""
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        raise ValueError("no overlapping non-NaN rows between y_true and y_pred")
    diff = (y_pred[mask] - y_true[mask]).to_numpy(dtype="float64")
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}
