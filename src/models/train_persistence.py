"""Stage 3a: smart persistence baseline.

Loads `data/features/train.parquet`, runs rolling-origin CV using the SAME
fold generator T5 and T6 will use, logs MAE/RMSE per fold/target/horizon to
MLflow, runs sanity assertions, and exits non-zero if any assertion fails.

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000   # MLflow port-forwarded
    python -m src.models.train_persistence --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.ingest.schema import TIMESTAMP_COL

from .cv import Fold, rolling_origin_folds
from .mlflow_utils import reproducibility_tags, resolve_tracking_uri
from .persistence import (
    PersistenceConfig,
    persistence_forecast,
    score_predictions,
)

LOG = logging.getLogger("persistence")

EXPERIMENT_NAME = "solar_forecaster"
RUN_NAME = "persistence_baseline"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params).resolve().parent
    params = yaml.safe_load(Path(args.params).read_text())

    targets: list[str] = list(params["forecast"]["targets"])
    horizons: list[int] = [int(h) for h in params["forecast"]["horizons_steps"]]
    horizon_labels: list[str] = list(params["forecast"]["horizon_labels"])
    assert len(horizons) == len(horizon_labels), "horizons_steps / horizon_labels mismatch"

    cv_cfg = params["training"]["cv"]
    pcfg = PersistenceConfig.from_params(params)

    train_path = repo_root / params["paths"]["features"] / "train.parquet"
    LOG.info("loading train features: %s", train_path)
    df = pd.read_parquet(train_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    LOG.info("train rows=%d cols=%d", len(df), df.shape[1])

    folds = rolling_origin_folds(len(df), cv_cfg)
    LOG.info(
        "CV folds: n=%d test_size=%s gap=%s",
        len(folds),
        cv_cfg["test_size_steps"],
        cv_cfg["gap_steps"],
    )
    for f in folds:
        LOG.info(
            "  fold %d: train[0:%d] (%d) val[%d:%d] (%d)",
            f.index,
            f.train_end,
            f.train_end,
            f.val_start,
            f.val_end,
            f.val_end - f.val_start,
        )

    # Compute predictions over the WHOLE train frame once per (target, horizon),
    # then slice by fold for scoring. Cheap; avoids redundant shifts.
    preds: dict[tuple[str, int], pd.Series] = {}
    for t in targets:
        for h in horizons:
            preds[(t, h)] = persistence_forecast(df, t, h, pcfg)

    # MLflow
    tracking_uri = resolve_tracking_uri()
    LOG.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    tags = reproducibility_tags(repo_root)
    tags["model_type"] = "persistence"
    tags["cv_scheme"] = str(cv_cfg["scheme"])
    LOG.info("repro tags: git_commit=%s dvc_hash=%s", tags["git_commit"], tags["dvc_hash"])

    fold_metrics: dict[int, dict[str, float]] = {}
    per_cell: dict[str, list[float]] = {}  # "mae.{target}.{label}" -> list across folds

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tags(tags)

        # Params: cv config, persistence config, what we forecast.
        mlflow.log_params(
            {
                "cv_scheme": cv_cfg["scheme"],
                "cv_n_splits": int(cv_cfg["n_splits"]),
                "cv_test_size_steps": int(cv_cfg["test_size_steps"]),
                "cv_gap_steps": int(cv_cfg["gap_steps"]),
                "kt_clip_min": pcfg.kt_clip_min,
                "kt_clip_max": pcfg.kt_clip_max,
                "targets": ",".join(targets),
                "horizons_steps": ",".join(str(h) for h in horizons),
                "horizon_labels": ",".join(horizon_labels),
                "persistence_rule": "smart_persistence_on_clearsky_index",
            }
        )

        for fold in folds:
            fold_metrics[fold.index] = {}
            sl = fold.val_slice
            y_block = df.iloc[sl]

            for t in targets:
                y_true = y_block[t]
                for h, label in zip(horizons, horizon_labels, strict=True):
                    y_pred = preds[(t, h)].iloc[sl]
                    scores = score_predictions(y_true, y_pred)

                    mae_key = f"fold{fold.index}.mae.{t}.{label}"
                    rmse_key = f"fold{fold.index}.rmse.{t}.{label}"
                    mlflow.log_metric(mae_key, scores["mae"])
                    mlflow.log_metric(rmse_key, scores["rmse"])
                    fold_metrics[fold.index][mae_key] = scores["mae"]
                    fold_metrics[fold.index][rmse_key] = scores["rmse"]

                    per_cell.setdefault(f"mae.{t}.{label}", []).append(scores["mae"])
                    per_cell.setdefault(f"rmse.{t}.{label}", []).append(scores["rmse"])

        # Aggregates across folds (mean).
        agg: dict[str, float] = {}
        for cell, vals in per_cell.items():
            mean_val = float(np.mean(vals))
            mlflow.log_metric(f"mean.{cell}", mean_val)
            agg[f"mean.{cell}"] = mean_val

        # Skill-vs-self (== 0 by definition; sanity check).
        for t in targets:
            for label in horizon_labels:
                mlflow.log_metric(f"mean.skill_vs_self.{t}.{label}", 0.0)

        # --- T9: score on promotion validation window. ---
        # Persistence's own skill on promo is 0 by construction (it IS the
        # baseline). We still log the raw RMSE/MAE so xgb/lstm runs can read
        # them as the denominator for their own promo skill scores.
        promo_path = repo_root / params["paths"]["features"] / "promo.parquet"
        LOG.info("loading promo features for promotion-window scoring: %s", promo_path)
        promo_df = pd.read_parquet(promo_path)
        promo_df[TIMESTAMP_COL] = pd.to_datetime(promo_df[TIMESTAMP_COL], utc=True)
        promo_df = promo_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        # Stitch the last max_h rows of train onto promo so persistence has
        # history at the boundary. Slice them back off after computing.
        max_h = max(horizons)
        stitched = pd.concat([df.tail(max_h), promo_df], axis=0, ignore_index=True)
        promo_y_pred: dict[tuple[str, str], pd.Series] = {}
        for t in targets:
            for h, lbl in zip(horizons, horizon_labels, strict=True):
                pred_full = persistence_forecast(stitched, t, h, pcfg)
                pred_slice = pred_full.iloc[max_h:].reset_index(drop=True)
                scores = score_predictions(promo_df[t], pred_slice)
                mlflow.log_metric(f"promo.rmse.{t}.{lbl}", scores["rmse"])
                mlflow.log_metric(f"promo.mae.{t}.{lbl}", scores["mae"])
                mlflow.log_metric(f"promo.skill.{t}.{lbl}", 0.0)
                promo_y_pred[(t, lbl)] = pred_slice
                LOG.info(
                    "  promo  %s %s  RMSE=%.3f  MAE=%.3f  skill=+0.0000",
                    t,
                    lbl,
                    scores["rmse"],
                    scores["mae"],
                )
        mlflow.log_metric("promo.mean_skill", 0.0)

        # Train window — used by promote.py's leakage guardrail.
        mlflow.log_param("train_window_start", _iso(df[TIMESTAMP_COL].iloc[0]))
        mlflow.log_param("train_window_end", _iso(df[TIMESTAMP_COL].iloc[-1]))
        mlflow.log_param("promo_window_start", _iso(promo_df[TIMESTAMP_COL].iloc[0]))
        mlflow.log_param("promo_window_end", _iso(promo_df[TIMESTAMP_COL].iloc[-1]))

        # Artifacts: fold boundaries + last-fold predictions.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            folds_path = tmpdir / "cv_folds.json"
            folds_path.write_text(
                json.dumps(
                    {
                        "n_rows_train_split": len(df),
                        "folds": [
                            {
                                **f.to_dict(),
                                "train_end_ts": _iso(df[TIMESTAMP_COL].iloc[f.train_end - 1]),
                                "val_start_ts": _iso(df[TIMESTAMP_COL].iloc[f.val_start]),
                                "val_end_ts": _iso(df[TIMESTAMP_COL].iloc[f.val_end - 1]),
                            }
                            for f in folds
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            mlflow.log_artifact(str(folds_path), artifact_path="cv")

            last_fold = folds[-1]
            last_pred_path = tmpdir / "last_fold_predictions.csv"
            _dump_last_fold_predictions(df, preds, last_fold, targets, horizons, last_pred_path)
            mlflow.log_artifact(str(last_pred_path), artifact_path="predictions")

        LOG.info("MLflow run_id=%s", run.info.run_id)

    # ----- Sanity assertions (post-run) -----
    failures: list[str] = []

    # 1. Monotone with horizon: 1h RMSE > 15min RMSE for every target.
    for t in targets:
        rmse_15 = agg[f"mean.rmse.{t}.15min"]
        rmse_1h = agg[f"mean.rmse.{t}.1h"]
        LOG.info("  %s: 15min RMSE=%.3f  1h RMSE=%.3f", t, rmse_15, rmse_1h)
        if not (rmse_1h > rmse_15):
            failures.append(
                f"expected 1h RMSE > 15min RMSE for {t}, got 15min={rmse_15:.3f} 1h={rmse_1h:.3f}"
            )

    # 2. No NaN predictions in val windows (after the cold-start drop).
    for fold in folds:
        for t in targets:
            for h, label in zip(horizons, horizon_labels, strict=True):
                pred_slice = preds[(t, h)].iloc[fold.val_slice]
                if pred_slice.isna().any():
                    failures.append(f"NaN predictions in fold {fold.index} {t} {label} val window")

    # 3. Stable-high-sun window check: on a 2-hour clear-sky midday window,
    #    15-min GHI persistence should have very low relative error.
    sanity = _clear_day_sanity(df, preds[("ghi", 1)])
    if sanity is None:
        failures.append("no stable clear-sky midday window found in train split")
    else:
        rmse, mean_ghi, label = sanity
        rel = rmse / mean_ghi if mean_ghi > 0 else float("inf")
        LOG.info(
            "  clear-sky window (%s) GHI 15min RMSE=%.2f  mean ghi=%.2f  rel=%.4f",
            label,
            rmse,
            mean_ghi,
            rel,
        )
        if rel > 0.05:
            failures.append(
                f"clear-sky-window 15min GHI RMSE/mean={rel:.4f} > 0.05 — persistence broken"
            )

    # Print final summary.
    print()
    print("=" * 72)
    print(f"persistence_baseline  (MLflow run_id={run.info.run_id})")
    print("=" * 72)
    print(f"{'target':<6} {'horizon':<8} {'mean MAE':>12} {'mean RMSE':>12}")
    for t in targets:
        for label in horizon_labels:
            print(
                f"{t:<6} {label:<8} "
                f"{agg[f'mean.mae.{t}.{label}']:>12.3f} "
                f"{agg[f'mean.rmse.{t}.{label}']:>12.3f}"
            )
    print("=" * 72)

    if failures:
        print()
        for msg in failures:
            print(f"FAIL: {msg}")
        return 1

    print("all sanity checks passed.")
    return 0


def _clear_day_sanity(
    df: pd.DataFrame, ghi_15min_pred: pd.Series
) -> tuple[float, float, str] | None:
    """Find a stable high-sun clear-sky window and return (rmse, mean_ghi, label).

    The classical "near-perfect on a clear winter day" property of smart
    persistence really means: when nothing is changing, the next-step forecast
    is the current value. We test exactly that by finding a 2-hour midday
    window where the sun is high (zenith < 60°) and GHI is both high and
    nearly constant, then asserting 15-min persistence RMSE / mean GHI < 5%.

    Avoiding the full-day filter is intentional: the Ineichen clear-sky model
    underestimates real GHI near the horizon at this latitude, so morning /
    evening k_t systematically clips to 1.5 on truly clear days. Restricting
    to zenith < 60° sidesteps that artifact without inventing new tolerances.
    """
    window_steps = 8  # 2 hours at 15-min resolution
    ts = df[TIMESTAMP_COL]

    ghi = df["ghi"].to_numpy()
    zenith = df["zenith"].to_numpy()
    pred = ghi_15min_pred.to_numpy()

    # Rolling stats over the window. .rolling(window=W) at position t covers
    # rows [t-W+1, t]; we treat that window as the candidate.
    s_ghi = pd.Series(ghi)
    roll_mean = s_ghi.rolling(window_steps).mean().to_numpy()
    roll_std = s_ghi.rolling(window_steps).std(ddof=0).to_numpy()
    roll_zmax = pd.Series(zenith).rolling(window_steps).max().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_std = np.where(roll_mean > 0, roll_std / roll_mean, np.inf)

    # High-sun + high-GHI + low variability
    eligible = (roll_zmax < 60.0) & (roll_mean > 500.0) & (rel_std < 0.03)
    # Predictions must be non-NaN across the whole window.
    pred_window_has_nan = pd.Series(pred).isna().rolling(window_steps).sum().to_numpy() > 0
    eligible &= ~pred_window_has_nan

    idx_end = int(np.argmax(eligible)) if eligible.any() else -1
    if idx_end < 0:
        return None

    sl = slice(idx_end - window_steps + 1, idx_end + 1)
    diff = pred[sl] - ghi[sl]
    rmse = float(np.sqrt(np.mean(diff**2)))
    mean_ghi = float(np.mean(ghi[sl]))
    label = f"midday {pd.Timestamp(ts.iloc[sl.start]).isoformat()}"
    return rmse, mean_ghi, label


def _dump_last_fold_predictions(
    df: pd.DataFrame,
    preds: dict[tuple[str, int], pd.Series],
    fold: Fold,
    targets: list[str],
    horizons: list[int],
    out_path: Path,
) -> None:
    sl = fold.val_slice
    rows = []
    ts = df[TIMESTAMP_COL].iloc[sl].to_numpy()
    for t in targets:
        y_true = df[t].iloc[sl].to_numpy()
        for h in horizons:
            y_pred = preds[(t, h)].iloc[sl].to_numpy()
            for i in range(len(ts)):
                rows.append(
                    {
                        "timestamp": _iso(pd.Timestamp(ts[i])),
                        "target": t,
                        "horizon_steps": h,
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]) if not np.isnan(y_pred[i]) else None,
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train smart persistence baseline (T4)")
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
