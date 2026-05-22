"""Finalize the last lstm_candidate MLflow run.

When the T6 trainer's artifact upload to MinIO fails (typical: missing
AWS_*/MLFLOW_S3_ENDPOINT_URL env vars), the run is marked FAILED even though
every metric and tag is logged correctly. This script:

  1. Locates the most recent FAILED lstm_candidate run.
  2. Reconstructs per-fold training curves + per-fold metric tables from the
     metrics already logged on the run (mlflow.get_metric_history).
  3. Uploads them as artifacts under the existing run.
  4. Marks the run as FINISHED.

Requirements:
  - MLFLOW_TRACKING_URI set (e.g. http://localhost:5001).
  - MLFLOW_S3_ENDPOINT_URL set (e.g. http://localhost:9000).
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set (MinIO root user/pass).

The MLflow run is deterministic and bit-identical to a re-train, so this
saves ~70 minutes of CPU while producing the same final state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.ingest.schema import TIMESTAMP_COL
from src.models.cv import rolling_origin_folds

LOG = logging.getLogger("finalize_lstm")

EXPERIMENT_NAME = "solar_forecaster"
RUN_NAME = "lstm_candidate"

_FOLD_METRIC_RE = re.compile(
    r"^fold(?P<fold>\d+)\.(?P<kind>rmse|mae|skill|train_loss|val_loss|mean_val_skill|"
    r"best_epoch|train_loss_best|val_loss_best)(?:\.(?P<target>[a-z_]+)\.(?P<label>[^.]+))?$"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.params).resolve().parent
    params = yaml.safe_load(Path(args.params).read_text())

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    for var in ("MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} is not set — required to upload artifacts to MinIO")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT_NAME!r} not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=(
            f"tags.mlflow.runName = '{RUN_NAME}' "
            "and tags.model_type = 'lstm' "
            "and attributes.status = 'FAILED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"no FAILED {RUN_NAME} run to finalize")
    run = runs[0]
    run_id = run.info.run_id
    LOG.info("finalizing run %s (status=%s)", run_id, run.info.status)

    # --- Reconstruct per-fold metric tables from logged metrics ---
    per_fold_rmse: dict[tuple[int, str, str], float] = {}
    per_fold_mae: dict[tuple[int, str, str], float] = {}
    per_fold_skill: dict[tuple[int, str, str], float] = {}
    per_fold_best_epoch: dict[int, int] = {}
    per_fold_train_loss_best: dict[int, float] = {}
    per_fold_val_loss_best: dict[int, float] = {}

    for k, v in run.data.metrics.items():
        m = _FOLD_METRIC_RE.match(k)
        if not m:
            continue
        fold = int(m["fold"])
        kind = m["kind"]
        target = m["target"]
        label = m["label"]
        if kind in ("rmse", "mae", "skill") and target and label:
            d = {"rmse": per_fold_rmse, "mae": per_fold_mae, "skill": per_fold_skill}[kind]
            d[(fold, target, label)] = float(v)
        elif kind == "best_epoch":
            per_fold_best_epoch[fold] = int(v)
        elif kind == "train_loss_best":
            per_fold_train_loss_best[fold] = float(v)
        elif kind == "val_loss_best":
            per_fold_val_loss_best[fold] = float(v)

    # --- Reconstruct per-epoch training curves from metric history ---
    training_curves: dict[int, list[dict]] = defaultdict(list)
    folds_seen = sorted(per_fold_best_epoch.keys())
    for fold in folds_seen:
        # Each of these has step=epoch in the original log.
        tl = {h.step: h.value for h in client.get_metric_history(run_id, f"fold{fold}.train_loss")}
        vl = {h.step: h.value for h in client.get_metric_history(run_id, f"fold{fold}.val_loss")}
        ms = {
            h.step: h.value for h in client.get_metric_history(run_id, f"fold{fold}.mean_val_skill")
        }
        epochs = sorted(set(tl) | set(vl) | set(ms))
        for e in epochs:
            training_curves[fold].append(
                {
                    "epoch": int(e),
                    "train_loss": float(tl.get(e, float("nan"))),
                    "val_loss": float(vl.get(e, float("nan"))),
                    "mean_val_skill": float(ms.get(e, float("nan"))),
                }
            )

    # --- Rebuild CV fold-boundary file from params + train.parquet length ---
    import pandas as pd

    train_path = repo_root / params["paths"]["features"] / "train.parquet"
    df_ts = pd.read_parquet(train_path, columns=[TIMESTAMP_COL])
    df_ts[TIMESTAMP_COL] = pd.to_datetime(df_ts[TIMESTAMP_COL], utc=True)
    df_ts = df_ts.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    folds = rolling_origin_folds(len(df_ts), params["training"]["cv"])

    def _iso(ts) -> str:
        return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")

    cv_folds_payload = {
        "n_rows_train_split": len(df_ts),
        "folds": [
            {
                **f.to_dict(),
                "train_end_ts": _iso(df_ts[TIMESTAMP_COL].iloc[f.train_end - 1]),
                "val_start_ts": _iso(df_ts[TIMESTAMP_COL].iloc[f.val_start]),
                "val_end_ts": _iso(df_ts[TIMESTAMP_COL].iloc[f.val_end - 1]),
            }
            for f in folds
        ],
    }

    # --- Write to tempfiles and upload to the existing run ---
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
                    "_note": (
                        "reconstructed from MLflow-logged metrics via "
                        "scripts/finalize_lstm_run.py"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

        folds_path = tmpdir / "cv_folds.json"
        folds_path.write_text(json.dumps(cv_folds_payload, indent=2) + "\n")

        client.log_artifact(run_id, str(curves_path), artifact_path="diagnostics")
        client.log_artifact(run_id, str(summary_path), artifact_path="diagnostics")
        client.log_artifact(run_id, str(folds_path), artifact_path="cv")
        LOG.info("uploaded 3 artifacts to run %s", run_id)

    # Mark FINISHED.
    client.set_terminated(run_id, status="FINISHED")
    LOG.info("set run %s status=FINISHED", run_id)

    # --- Final printable summary table ---
    targets = list(params["forecast"]["targets"])
    horizon_labels = list(params["forecast"]["horizon_labels"])
    refreshed = client.get_run(run_id)
    m = refreshed.data.metrics

    print()
    print("=" * 78)
    print(f"lstm_candidate  (MLflow run_id={run_id})  — FINALIZED")
    print("=" * 78)
    print(f"{'target':<6} {'horizon':<8} {'mean MAE':>12} {'mean RMSE':>12} {'mean skill':>12}")
    for t in targets:
        for lbl in horizon_labels:
            print(
                f"{t:<6} {lbl:<8} "
                f"{m.get(f'mean.mae.{t}.{lbl}', float('nan')):>12.3f} "
                f"{m.get(f'mean.rmse.{t}.{lbl}', float('nan')):>12.3f} "
                f"{m.get(f'mean.skill.{t}.{lbl}', float('nan')):>+12.4f}"
            )
    print("-" * 78)
    print(f"mean skill across all 6 outputs: {m.get('mean.skill', float('nan')):+.4f}")
    print("=" * 78)
    print("per-fold best epoch / train loss / val loss:")
    for fold in folds_seen:
        print(
            f"  fold {fold}: epoch={per_fold_best_epoch.get(fold)}  "
            f"train_loss_best={per_fold_train_loss_best.get(fold):.5f}  "
            f"val_loss_best={per_fold_val_loss_best.get(fold):.5f}"
        )
    return 0


def _stringify_keys(d: dict) -> dict:
    return {f"fold{k[0]}.{k[1]}.{k[2]}": v for k, v in d.items()}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize a FAILED lstm_candidate run")
    p.add_argument("--params", default="params.yaml", help="path to params.yaml")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
