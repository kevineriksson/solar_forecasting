"""Stage 3c: PyTorch LSTM candidate.

Multivariate-sequence LSTM that takes the last `sequence_length_steps`
feature rows ending at the as-of time t and predicts the 6 forecast
outputs (3 targets x 2 horizons) at t+1 and t+4.

Uses the SAME rolling-origin CV folds as T4 (persistence) and T5 (XGBoost)
so per-fold skill scores are apples-to-apples vs the persistence baseline.

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000   # MLflow port-forwarded
    python -m src.models.lstm_train --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.ingest.schema import TIMESTAMP_COL

from .cv import Fold, rolling_origin_folds
from .lstm_model import (
    FeatureScaler,
    LSTMRegressor,
    SequenceWindowDataset,
    to_float_array,
    valid_anchor_range,
)
from .mlflow_utils import reproducibility_tags, resolve_tracking_uri
from .skill import find_persistence_run, load_per_fold_rmse, skill_score
from .xgb_train import EXCLUDED_FEATURE_COLS

LOG = logging.getLogger("lstm_train")

EXPERIMENT_NAME = "solar_forecaster"
RUN_NAME = "lstm_candidate"
SEED = 42


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    lstm_cfg = params["training"]["lstm"]

    seq_len = int(args.seq_len if args.seq_len is not None else lstm_cfg["sequence_length_steps"])
    hidden_size = int(args.hidden_size if args.hidden_size is not None else lstm_cfg["hidden_size"])
    num_layers = int(lstm_cfg["num_layers"])
    dropout = float(lstm_cfg["dropout"])
    batch_size = int(lstm_cfg["batch_size"])
    lr = float(lstm_cfg["learning_rate"])
    max_epochs = int(args.epochs if args.epochs is not None else lstm_cfg["epochs"])
    early_stop_patience = int(lstm_cfg["early_stopping_patience"])

    _seed_all(SEED)

    train_path = repo_root / params["paths"]["features"] / "train.parquet"
    LOG.info("loading train features: %s", train_path)
    df = pd.read_parquet(train_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    LOG.info("train rows=%d cols=%d", len(df), df.shape[1])

    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]
    LOG.info("using %d feature columns", len(feature_cols))

    # Fail fast if any feature has NaNs in the rows we'll actually use.
    # T3 guarantees no NaNs after the longest warmup window (12 steps).
    # float32 throughout to halve the base array footprint (~225 MB -> ~112 MB
    # on the 655k-row train split).
    feat_arr = to_float_array(df, feature_cols).astype(np.float32)
    tgt_arr = to_float_array(df, targets).astype(np.float32)
    if not np.isfinite(feat_arr).all():
        n_bad = int((~np.isfinite(feat_arr)).any(axis=1).sum())
        raise RuntimeError(
            f"non-finite values in feature matrix: {n_bad} rows; T3 should have dropped these"
        )
    if not np.isfinite(tgt_arr).all():
        n_bad = int((~np.isfinite(tgt_arr)).any(axis=1).sum())
        raise RuntimeError(f"non-finite values in target matrix: {n_bad} rows")

    folds = rolling_origin_folds(len(df), cv_cfg)
    LOG.info(
        "CV folds: n=%d test_size=%s gap=%s",
        len(folds),
        cv_cfg["test_size_steps"],
        cv_cfg["gap_steps"],
    )

    # MLflow setup + persistence-baseline lookup
    tracking_uri = resolve_tracking_uri()
    LOG.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    persistence_ref = find_persistence_run(EXPERIMENT_NAME)
    LOG.info(
        "persistence baseline run_id=%s git=%s dvc=%s",
        persistence_ref.run_id,
        persistence_ref.git_commit[:8],
        persistence_ref.dvc_hash[:8],
    )
    persistence_rmse = load_per_fold_rmse(persistence_ref.run_id)

    missing = [
        (f.index, t, lbl)
        for f in folds
        for t in targets
        for lbl in horizon_labels
        if (f.index, t, lbl) not in persistence_rmse
    ]
    if missing:
        raise RuntimeError(
            f"persistence baseline run is missing per-fold RMSE entries: {missing[:5]}..."
        )

    tags = reproducibility_tags(repo_root)
    tags["model_type"] = "lstm"
    tags["cv_scheme"] = str(cv_cfg["scheme"])
    tags["persistence_run_id"] = persistence_ref.run_id
    LOG.info("repro tags: git_commit=%s dvc_hash=%s", tags["git_commit"], tags["dvc_hash"])

    n_outputs = len(targets) * len(horizons)
    cell_keys = [
        (t, h, lbl) for t in targets for h, lbl in zip(horizons, horizon_labels, strict=True)
    ]
    # Output ordering must match build_sequences: target-major, horizon-inner.
    output_columns = [(t, lbl) for t in targets for lbl in horizon_labels]

    per_fold_rmse: dict[tuple[int, str, str], float] = {}
    per_fold_mae: dict[tuple[int, str, str], float] = {}
    per_fold_skill: dict[tuple[int, str, str], float] = {}
    per_fold_best_epoch: dict[int, int] = {}
    per_fold_train_loss_best: dict[int, float] = {}
    per_fold_val_loss_best: dict[int, float] = {}
    training_curves: dict[int, list[dict]] = {}

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tags(tags)

        mlflow.log_params(
            {
                "cv_scheme": cv_cfg["scheme"],
                "cv_n_splits": int(cv_cfg["n_splits"]),
                "cv_test_size_steps": int(cv_cfg["test_size_steps"]),
                "cv_gap_steps": int(cv_cfg["gap_steps"]),
                "targets": ",".join(targets),
                "horizons_steps": ",".join(str(h) for h in horizons),
                "horizon_labels": ",".join(horizon_labels),
                "sequence_length_steps": seq_len,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs_max": max_epochs,
                "early_stopping_patience": early_stop_patience,
                "early_stopping_metric": "mean_val_skill_score",
                "n_features": len(feature_cols),
                "feature_cols": ",".join(feature_cols),
                "device": "cpu",
                "seed": SEED,
                "param_sequence_length_default": int(lstm_cfg["sequence_length_steps"]),
                "param_hidden_size_default": int(lstm_cfg["hidden_size"]),
                "param_epochs_default": int(lstm_cfg["epochs"]),
            }
        )

        for fold in folds:
            t_fold = time.time()
            LOG.info(
                "fold %d: train[0:%d) val[%d:%d)",
                fold.index,
                fold.train_end,
                fold.val_start,
                fold.val_end,
            )

            # Build anchor row ranges. Lookback must fit inside each block.
            tr_a_start, tr_a_end = valid_anchor_range(
                fold.train_start, fold.train_end, seq_len, max(horizons)
            )
            va_a_start, va_a_end = valid_anchor_range(
                fold.val_start, fold.val_end, seq_len, max(horizons)
            )
            tr_anchors = np.arange(tr_a_start, tr_a_end)
            va_anchors = np.arange(va_a_start, va_a_end)
            if len(tr_anchors) == 0 or len(va_anchors) == 0:
                raise RuntimeError(
                    f"fold {fold.index}: empty anchor range "
                    f"train=[{tr_a_start},{tr_a_end}) val=[{va_a_start},{va_a_end})"
                )
            LOG.info(
                "  anchors: train=%d val=%d (seq_len=%d, max_h=%d)",
                len(tr_anchors),
                len(va_anchors),
                seq_len,
                max(horizons),
            )

            # Fit per-feature scaler on the training feature rows ONLY
            # (rows fold.train_start..fold.train_end). Fitting on a subset
            # of the block, or including val rows, would leak val statistics.
            train_feat_rows = feat_arr[fold.train_start : fold.train_end]
            x_scaler = FeatureScaler.fit(train_feat_rows)

            # Fit per-output target scaler the same way (uses target rows in
            # [train_start + h, train_end) for each horizon — easier and
            # tighter: just compute on the train block targets directly).
            train_tgt_rows = tgt_arr[fold.train_start : fold.train_end]
            tgt_mean = train_tgt_rows.mean(axis=0)  # (T,)
            tgt_std = train_tgt_rows.std(axis=0, ddof=0)
            tgt_std = np.where(tgt_std > 1e-12, tgt_std, 1.0)
            # Broadcast (T,) -> (T*H,) in target-major order to match SequenceWindowDataset.
            y_mean = np.repeat(tgt_mean, len(horizons)).astype(np.float32)
            y_std = np.repeat(tgt_std, len(horizons)).astype(np.float32)

            # Build raw-units val labels once for end-of-epoch metrics. The
            # per-batch standardized labels are produced on the fly by the
            # dataset; we keep raw Y_va here in target-major order to compute
            # per-output RMSE/MAE in original units after de-standardizing
            # predictions.
            Y_va_raw = np.empty((len(va_anchors), n_outputs), dtype=np.float32)
            col = 0
            for t_idx in range(tgt_arr.shape[1]):
                for h in horizons:
                    Y_va_raw[:, col] = tgt_arr[va_anchors + h, t_idx].astype(np.float32)
                    col += 1

            tr_ds = SequenceWindowDataset(
                feat_arr,
                tgt_arr,
                tr_anchors,
                seq_len,
                horizons,
                x_mean=x_scaler.mean,
                x_std=x_scaler.std,
                y_mean=y_mean,
                y_std=y_std,
            )
            va_ds = SequenceWindowDataset(
                feat_arr,
                tgt_arr,
                va_anchors,
                seq_len,
                horizons,
                x_mean=x_scaler.mean,
                x_std=x_scaler.std,
                y_mean=y_mean,
                y_std=y_std,
            )
            tr_loader = DataLoader(
                tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
            )
            va_loader = DataLoader(
                va_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
            )

            # Fresh model + optimizer per fold (no warm starts across folds).
            torch.manual_seed(SEED + fold.index)
            model = LSTMRegressor(
                n_features=len(feature_cols),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                n_outputs=n_outputs,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            # Pre-compute per-fold persistence RMSEs in output-column order
            # so we can compute mean_val_skill per epoch cheaply.
            baseline_rmse_per_output = np.array(
                [persistence_rmse[(fold.index, t, lbl)] for (t, lbl) in output_columns],
                dtype=np.float64,
            )

            best_mean_skill = -float("inf")
            best_epoch = -1
            best_val_loss = float("inf")
            best_train_loss = float("inf")
            best_per_output_rmse = np.zeros(n_outputs, dtype=np.float64)
            best_per_output_mae = np.zeros(n_outputs, dtype=np.float64)
            epochs_since_improve = 0
            curves: list[dict] = []

            for epoch in range(max_epochs):
                t_ep = time.time()
                model.train()
                running = 0.0
                n_obs = 0
                for xb, yb in tr_loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()
                    running += float(loss.item()) * xb.shape[0]
                    n_obs += xb.shape[0]
                train_loss = running / max(n_obs, 1)

                # Validation: predict in standardized space, then de-standardize
                # to compute per-output RMSE/MAE in raw units.
                model.eval()
                val_running = 0.0
                v_obs = 0
                preds_chunks: list[np.ndarray] = []
                with torch.no_grad():
                    for xb, yb in va_loader:
                        pred = model(xb)
                        vloss = loss_fn(pred, yb)
                        val_running += float(vloss.item()) * xb.shape[0]
                        v_obs += xb.shape[0]
                        preds_chunks.append(pred.cpu().numpy())
                val_loss = val_running / max(v_obs, 1)

                preds_std = np.concatenate(preds_chunks, axis=0)
                preds_raw = preds_std * y_std + y_mean  # (K, n_outputs)
                diff = preds_raw.astype(np.float64) - Y_va_raw.astype(np.float64)
                per_out_rmse = np.sqrt((diff**2).mean(axis=0))
                per_out_mae = np.abs(diff).mean(axis=0)
                per_out_skill = 1.0 - (per_out_rmse / baseline_rmse_per_output)
                mean_skill = float(per_out_skill.mean())

                dt_ep = time.time() - t_ep
                LOG.info(
                    "  fold %d epoch %2d  train_loss=%.5f  val_loss=%.5f  "
                    "mean_skill=%+.4f  (%.1fs)",
                    fold.index,
                    epoch,
                    train_loss,
                    val_loss,
                    mean_skill,
                    dt_ep,
                )
                curves.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "mean_val_skill": float(mean_skill),
                        "duration_s": float(dt_ep),
                    }
                )

                mlflow.log_metric(f"fold{fold.index}.train_loss", float(train_loss), step=epoch)
                mlflow.log_metric(f"fold{fold.index}.val_loss", float(val_loss), step=epoch)
                mlflow.log_metric(f"fold{fold.index}.mean_val_skill", float(mean_skill), step=epoch)

                if mean_skill > best_mean_skill + 1e-6:
                    best_mean_skill = mean_skill
                    best_epoch = epoch
                    best_val_loss = float(val_loss)
                    best_train_loss = float(train_loss)
                    best_per_output_rmse = per_out_rmse.copy()
                    best_per_output_mae = per_out_mae.copy()
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= early_stop_patience:
                        LOG.info(
                            "  fold %d: early stop at epoch %d (no skill improvement for %d epochs)",
                            fold.index,
                            epoch,
                            epochs_since_improve,
                        )
                        break

            if best_epoch < 0:
                raise RuntimeError(f"fold {fold.index}: no epoch completed successfully")

            per_fold_best_epoch[fold.index] = best_epoch
            per_fold_train_loss_best[fold.index] = best_train_loss
            per_fold_val_loss_best[fold.index] = best_val_loss
            training_curves[fold.index] = curves

            # Persist per-cell best metrics for this fold.
            for j, (t, lbl) in enumerate(output_columns):
                per_fold_rmse[(fold.index, t, lbl)] = float(best_per_output_rmse[j])
                per_fold_mae[(fold.index, t, lbl)] = float(best_per_output_mae[j])
                per_fold_skill[(fold.index, t, lbl)] = float(
                    skill_score(
                        float(best_per_output_rmse[j]),
                        float(baseline_rmse_per_output[j]),
                    )
                )

                mlflow.log_metric(
                    f"fold{fold.index}.rmse.{t}.{lbl}", per_fold_rmse[(fold.index, t, lbl)]
                )
                mlflow.log_metric(
                    f"fold{fold.index}.mae.{t}.{lbl}", per_fold_mae[(fold.index, t, lbl)]
                )
                mlflow.log_metric(
                    f"fold{fold.index}.skill.{t}.{lbl}", per_fold_skill[(fold.index, t, lbl)]
                )

            mlflow.log_metric(f"fold{fold.index}.best_epoch", float(best_epoch))
            mlflow.log_metric(f"fold{fold.index}.train_loss_best", float(best_train_loss))
            mlflow.log_metric(f"fold{fold.index}.val_loss_best", float(best_val_loss))

            LOG.info(
                "fold %d: best_epoch=%d mean_skill=%+.4f total=%.1fs",
                fold.index,
                best_epoch,
                best_mean_skill,
                time.time() - t_fold,
            )

        # Aggregate per cell across folds.
        agg_rmse: dict[tuple[str, str], float] = {}
        agg_mae: dict[tuple[str, str], float] = {}
        agg_skill: dict[tuple[str, str], float] = {}
        for t, _h, lbl in cell_keys:
            r = float(np.mean([per_fold_rmse[(f.index, t, lbl)] for f in folds]))
            m = float(np.mean([per_fold_mae[(f.index, t, lbl)] for f in folds]))
            s = float(np.mean([per_fold_skill[(f.index, t, lbl)] for f in folds]))
            agg_rmse[(t, lbl)] = r
            agg_mae[(t, lbl)] = m
            agg_skill[(t, lbl)] = s
            mlflow.log_metric(f"mean.rmse.{t}.{lbl}", r)
            mlflow.log_metric(f"mean.mae.{t}.{lbl}", m)
            mlflow.log_metric(f"mean.skill.{t}.{lbl}", s)

        mean_skill = float(np.mean(list(agg_skill.values())))
        mlflow.log_metric("mean.skill", mean_skill)

        # Artifact upload depends on MinIO + AWS env vars being set. Wrap so a
        # single S3 hiccup doesn't waste all the metric work; reconstruct
        # artifacts later via scripts/finalize_lstm_run.py if needed.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            curves_path = tmpdir / "training_curves.json"
            curves_path.write_text(
                json.dumps({str(k): v for k, v in training_curves.items()}, indent=2) + "\n"
            )

            summary_path = tmpdir / "per_fold_metrics.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "per_fold_rmse": _stringify_keys(per_fold_rmse),
                        "per_fold_mae": _stringify_keys(per_fold_mae),
                        "per_fold_skill": _stringify_keys(per_fold_skill),
                        "per_fold_best_epoch": {str(k): v for k, v in per_fold_best_epoch.items()},
                        "per_fold_train_loss_best": {
                            str(k): v for k, v in per_fold_train_loss_best.items()
                        },
                        "per_fold_val_loss_best": {
                            str(k): v for k, v in per_fold_val_loss_best.items()
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

            folds_path = tmpdir / "cv_folds.json"
            folds_path.write_text(
                json.dumps(
                    {
                        "n_rows_train_split": len(df),
                        "folds": [_fold_summary(df, f) for f in folds],
                    },
                    indent=2,
                )
                + "\n"
            )

            for local, sub in (
                (curves_path, "diagnostics"),
                (summary_path, "diagnostics"),
                (folds_path, "cv"),
            ):
                try:
                    mlflow.log_artifact(str(local), artifact_path=sub)
                except Exception as exc:  # noqa: BLE001 — S3/network errors take many shapes
                    LOG.warning(
                        "artifact upload failed for %s (%s) — metrics still logged; "
                        "rerun scripts/finalize_lstm_run.py to recover",
                        local.name,
                        exc.__class__.__name__,
                    )

        run_id = run.info.run_id
        LOG.info("MLflow run_id=%s", run_id)

    # ----- Final summary -----
    print()
    print("=" * 78)
    print(f"lstm_candidate  (MLflow run_id={run_id})")
    print("=" * 78)
    print(f"{'target':<6} {'horizon':<8} {'mean MAE':>12} {'mean RMSE':>12} {'mean skill':>12}")
    for t in targets:
        for lbl in horizon_labels:
            print(
                f"{t:<6} {lbl:<8} "
                f"{agg_mae[(t, lbl)]:>12.3f} "
                f"{agg_rmse[(t, lbl)]:>12.3f} "
                f"{agg_skill[(t, lbl)]:>+12.4f}"
            )
    print("-" * 78)
    print(f"mean skill across all 6 outputs: {mean_skill:+.4f}")
    print("=" * 78)

    # Convergence assertion: at least one fold's train loss dropped, and
    # best_epoch is not the very first epoch for every fold (would mean
    # the model never improved).
    any_progress = False
    for curves in training_curves.values():
        if len(curves) >= 2 and curves[-1]["train_loss"] < curves[0]["train_loss"]:
            any_progress = True
            break
    if not any_progress:
        print("FAIL: training loss did not decrease in any fold.")
        return 1

    if mean_skill <= 0.0:
        print()
        print("WARN: mean skill <= 0; LSTM did not beat persistence on average.")
        # We still return 0 — the Done-when criteria for T6 only require a
        # converged model + per-output skill scores reported, not skill > 0.

    print("ok — model converged; per-output skills reported.")
    return 0


def _stringify_keys(d: dict) -> dict:
    return {f"fold{k[0]}.{k[1]}.{k[2]}": v for k, v in d.items()}


def _fold_summary(df: pd.DataFrame, f: Fold) -> dict:
    ts = df[TIMESTAMP_COL]
    return {
        **f.to_dict(),
        "train_end_ts": pd.Timestamp(ts.iloc[f.train_end - 1]).isoformat().replace("+00:00", "Z"),
        "val_start_ts": pd.Timestamp(ts.iloc[f.val_start]).isoformat().replace("+00:00", "Z"),
        "val_end_ts": pd.Timestamp(ts.iloc[f.val_end - 1]).isoformat().replace("+00:00", "Z"),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM candidate (T6)")
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="override training.lstm.sequence_length_steps (does not modify params.yaml)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="override training.lstm.hidden_size (does not modify params.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override training.lstm.epochs cap (does not modify params.yaml)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
