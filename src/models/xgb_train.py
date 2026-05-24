"""Stage 3b: XGBoost candidate.

Trains 6 independent XGBoost regressors (3 targets x 2 horizons) on the
training-split feature table from T3, using the same rolling-origin CV folds
as T4. Computes per-fold skill score vs the persistence baseline run in
MLflow, logs everything under a new "xgboost_candidate" run, and exits
non-zero if mean skill score across the 6 outputs is <= 0.

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000   # MLflow port-forwarded
    python -m src.models.xgb_train --params params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from src.ingest.schema import TIMESTAMP_COL

from .cv import rolling_origin_folds
from .mlflow_utils import reproducibility_tags, resolve_tracking_uri
from .skill import find_persistence_run, load_per_fold_rmse, skill_score

LOG = logging.getLogger("xgb_train")

EXPERIMENT_NAME = "solar_forecaster"
RUN_NAME = "xgboost_candidate"

# Columns excluded from the feature matrix:
# - the timestamp index
# - raw target values at time t (we predict t+h; including current targets at
#   t blurs the persistence-style "what's the value h ahead" framing — k_t
#   already encodes current normalised GHI)
# - gti (raw irradiance observation tightly correlated with targets)
EXCLUDED_FEATURE_COLS = {TIMESTAMP_COL, "ghi", "dni", "dhi", "gti"}


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
    xgb_cfg = params["training"]["xgboost"]
    strategy = str(xgb_cfg.get("strategy", "per_output"))
    if strategy != "per_output":
        raise ValueError(
            f"only training.xgboost.strategy='per_output' is supported in T5; got {strategy!r}"
        )

    train_path = repo_root / params["paths"]["features"] / "train.parquet"
    LOG.info("loading train features: %s", train_path)
    df = pd.read_parquet(train_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    LOG.info("train rows=%d cols=%d", len(df), df.shape[1])

    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]
    LOG.info("using %d feature columns: %s", len(feature_cols), ", ".join(feature_cols))

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
        "persistence baseline run_id=%s  git=%s  dvc=%s",
        persistence_ref.run_id,
        persistence_ref.git_commit[:8],
        persistence_ref.dvc_hash[:8],
    )
    persistence_rmse = load_per_fold_rmse(persistence_ref.run_id)

    # Verify the baseline has metrics for every (fold, target, label) we need.
    missing = [
        (f.index, t, lbl)
        for f in folds
        for t in targets
        for lbl in horizon_labels
        if (f.index, t, lbl) not in persistence_rmse
    ]
    if missing:
        raise RuntimeError(
            f"persistence baseline run is missing required per-fold RMSE entries: {missing[:5]}..."
        )

    tags = reproducibility_tags(repo_root)
    tags["model_type"] = "xgboost"
    tags["cv_scheme"] = str(cv_cfg["scheme"])
    tags["persistence_run_id"] = persistence_ref.run_id
    LOG.info("repro tags: git_commit=%s dvc_hash=%s", tags["git_commit"], tags["dvc_hash"])

    # Per-cell containers.
    cell_keys = [
        (t, h, lbl) for t in targets for h, lbl in zip(horizons, horizon_labels, strict=True)
    ]
    per_fold_rmse: dict[tuple[int, str, str], float] = {}
    per_fold_mae: dict[tuple[int, str, str], float] = {}
    per_fold_skill: dict[tuple[int, str, str], float] = {}
    per_fold_train_rmse: dict[tuple[int, str, str], float] = {}
    per_fold_best_iter: dict[tuple[int, str, str], int] = {}
    feature_importances_last: dict[tuple[str, str], dict[str, float]] = {}
    # Last-fold models per (target, horizon_label) — used for T9 promo scoring.
    last_fold_models: dict[tuple[str, str], XGBRegressor] = {}

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
                "strategy": strategy,
                "n_estimators": int(xgb_cfg["n_estimators"]),
                "max_depth": int(xgb_cfg["max_depth"]),
                "learning_rate": float(xgb_cfg["learning_rate"]),
                "subsample": float(xgb_cfg["subsample"]),
                "colsample_bytree": float(xgb_cfg["colsample_bytree"]),
                "early_stopping_rounds": int(xgb_cfg["early_stopping_rounds"]),
                "n_features": len(feature_cols),
                "feature_cols": ",".join(feature_cols),
            }
        )

        # Pre-shift features once per horizon: features_at_t_minus_h aligned to row t.
        # This matches T4's alignment so skill scores compare apples-to-apples.
        X_full = df[feature_cols]
        X_shifted: dict[int, pd.DataFrame] = {
            h: X_full.shift(h).reset_index(drop=True) for h in horizons
        }

        for fold in folds:
            LOG.info(
                "fold %d: train[0:%d) (%d rows) val[%d:%d) (%d rows)",
                fold.index,
                fold.train_end,
                fold.train_end,
                fold.val_start,
                fold.val_end,
                fold.val_end - fold.val_start,
            )

            for t in targets:
                y_full = df[t].astype("float64")
                for h, lbl in zip(horizons, horizon_labels, strict=True):
                    Xh = X_shifted[h]

                    # Build train arrays: first h rows have NaN features (shift),
                    # drop them from training to keep XGBoost focused.
                    tr_start = max(fold.train_start, h)
                    X_tr = Xh.iloc[tr_start : fold.train_end]
                    y_tr = y_full.iloc[tr_start : fold.train_end]

                    X_val = Xh.iloc[fold.val_start : fold.val_end]
                    y_val = y_full.iloc[fold.val_start : fold.val_end]

                    model = XGBRegressor(
                        n_estimators=int(xgb_cfg["n_estimators"]),
                        max_depth=int(xgb_cfg["max_depth"]),
                        learning_rate=float(xgb_cfg["learning_rate"]),
                        subsample=float(xgb_cfg["subsample"]),
                        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
                        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
                        tree_method="hist",
                        n_jobs=-1,
                        objective="reg:squarederror",
                        random_state=42,
                    )
                    t0 = time.time()
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    dt = time.time() - t0

                    y_pred = model.predict(X_val)
                    y_pred_tr = model.predict(X_tr)
                    val_rmse = float(np.sqrt(np.mean((y_pred - y_val.to_numpy()) ** 2)))
                    val_mae = float(np.mean(np.abs(y_pred - y_val.to_numpy())))
                    tr_rmse = float(np.sqrt(np.mean((y_pred_tr - y_tr.to_numpy()) ** 2)))

                    baseline = persistence_rmse[(fold.index, t, lbl)]
                    sk = skill_score(val_rmse, baseline)

                    per_fold_rmse[(fold.index, t, lbl)] = val_rmse
                    per_fold_mae[(fold.index, t, lbl)] = val_mae
                    per_fold_skill[(fold.index, t, lbl)] = sk
                    per_fold_train_rmse[(fold.index, t, lbl)] = tr_rmse
                    per_fold_best_iter[(fold.index, t, lbl)] = int(
                        getattr(model, "best_iteration", -1)
                    )

                    mlflow.log_metric(f"fold{fold.index}.rmse.{t}.{lbl}", val_rmse)
                    mlflow.log_metric(f"fold{fold.index}.mae.{t}.{lbl}", val_mae)
                    mlflow.log_metric(f"fold{fold.index}.skill.{t}.{lbl}", sk)
                    mlflow.log_metric(f"fold{fold.index}.train_rmse.{t}.{lbl}", tr_rmse)
                    if getattr(model, "best_iteration", None) is not None:
                        mlflow.log_metric(
                            f"fold{fold.index}.best_iter.{t}.{lbl}", float(model.best_iteration)
                        )

                    LOG.info(
                        "  cell %s %s  val_rmse=%.3f  baseline=%.3f  skill=%+.4f  "
                        "train_rmse=%.3f  best_iter=%s  fit=%.1fs",
                        t,
                        lbl,
                        val_rmse,
                        baseline,
                        sk,
                        tr_rmse,
                        getattr(model, "best_iteration", "n/a"),
                        dt,
                    )

                    # Capture feature importances from the LAST fold for diagnostics.
                    if fold.index == folds[-1].index:
                        imp = model.feature_importances_
                        feature_importances_last[(t, lbl)] = {
                            feature_cols[i]: float(imp[i]) for i in range(len(feature_cols))
                        }
                        last_fold_models[(t, lbl)] = model

        # Aggregate per-cell across folds (mean).
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

        # --- T9: score the last-fold models on the promotion validation window. ---
        # Skill denominator: persistence's `promo.rmse.*` from its own run.
        persistence_promo_rmse = _load_persistence_promo_rmse(
            persistence_ref.run_id, targets, horizon_labels
        )
        promo_path = repo_root / params["paths"]["features"] / "promo.parquet"
        LOG.info("loading promo features for promotion-window scoring: %s", promo_path)
        promo_df = pd.read_parquet(promo_path)
        promo_df[TIMESTAMP_COL] = pd.to_datetime(promo_df[TIMESTAMP_COL], utc=True)
        promo_df = promo_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        # Stitch train tail (max_h rows) onto promo so X.shift(h) has valid
        # history for the first h rows of promo. Slice off after predict.
        max_h = max(horizons)
        promo_with_history = pd.concat([df.tail(max_h), promo_df], axis=0, ignore_index=True)
        promo_X_full = promo_with_history[feature_cols]
        promo_X_shifted: dict[int, pd.DataFrame] = {
            h: promo_X_full.shift(h).reset_index(drop=True) for h in horizons
        }

        promo_skills: list[float] = []
        for t in targets:
            y_true = promo_df[t].astype("float64").to_numpy()
            for h, lbl in zip(horizons, horizon_labels, strict=True):
                model = last_fold_models[(t, lbl)]
                X_pred = promo_X_shifted[h].iloc[max_h:].reset_index(drop=True)
                y_pred = model.predict(X_pred)
                diff = y_pred - y_true
                p_rmse = float(np.sqrt(np.mean(diff**2)))
                p_mae = float(np.mean(np.abs(diff)))
                baseline = persistence_promo_rmse[(t, lbl)]
                p_skill = skill_score(p_rmse, baseline)
                mlflow.log_metric(f"promo.rmse.{t}.{lbl}", p_rmse)
                mlflow.log_metric(f"promo.mae.{t}.{lbl}", p_mae)
                mlflow.log_metric(f"promo.skill.{t}.{lbl}", p_skill)
                promo_skills.append(p_skill)
                LOG.info(
                    "  promo  %s %s  RMSE=%.3f  baseline=%.3f  skill=%+.4f",
                    t,
                    lbl,
                    p_rmse,
                    baseline,
                    p_skill,
                )
        promo_mean_skill = float(np.mean(promo_skills))
        mlflow.log_metric("promo.mean_skill", promo_mean_skill)
        LOG.info("promo mean_skill across 6 outputs: %+.4f", promo_mean_skill)

        mlflow.log_param("train_window_start", _iso_ts(df[TIMESTAMP_COL].iloc[0]))
        mlflow.log_param("train_window_end", _iso_ts(df[TIMESTAMP_COL].iloc[-1]))
        mlflow.log_param("promo_window_start", _iso_ts(promo_df[TIMESTAMP_COL].iloc[0]))
        mlflow.log_param("promo_window_end", _iso_ts(promo_df[TIMESTAMP_COL].iloc[-1]))

        # Artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # T10: persist the last-fold models so serving can load them.
            # The promo-scored model IS the model we serve — promote.py reads
            # this run's artifacts directly, so the booster files here are
            # what becomes Production.
            model_dir = tmpdir / "model"
            model_dir.mkdir()
            model_files: dict[str, str] = {}
            for (t, lbl), mdl in last_fold_models.items():
                fname = f"xgb_{t}_{lbl}.json"
                mdl.get_booster().save_model(str(model_dir / fname))
                model_files[f"{t}.{lbl}"] = fname
            manifest = {
                "model_type": "xgboost",
                "strategy": strategy,
                "feature_columns": feature_cols,
                "targets": targets,
                "horizons_steps": horizons,
                "horizon_labels": horizon_labels,
                "output_columns": [[t, lbl] for t in targets for lbl in horizon_labels],
                "n_outputs": len(targets) * len(horizons),
                "model_files": model_files,
            }
            (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

            fi_path = tmpdir / "feature_importances.json"
            fi_path.write_text(
                json.dumps(
                    {
                        f"{t}.{lbl}": dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
                        for (t, lbl), imp in feature_importances_last.items()
                    },
                    indent=2,
                )
                + "\n"
            )
            mlflow.log_artifact(str(fi_path), artifact_path="diagnostics")

            summary_path = tmpdir / "per_fold_metrics.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "per_fold_rmse": _stringify_keys(per_fold_rmse),
                        "per_fold_mae": _stringify_keys(per_fold_mae),
                        "per_fold_skill": _stringify_keys(per_fold_skill),
                        "per_fold_train_rmse": _stringify_keys(per_fold_train_rmse),
                        "per_fold_best_iter": _stringify_keys(per_fold_best_iter),
                    },
                    indent=2,
                )
                + "\n"
            )
            mlflow.log_artifact(str(summary_path), artifact_path="diagnostics")

        run_id = run.info.run_id
        LOG.info("MLflow run_id=%s", run_id)

    # ----- Final summary + decision -----
    print()
    print("=" * 78)
    print(f"xgboost_candidate  (MLflow run_id={run_id})")
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

    if mean_skill <= 0.0:
        print()
        print("FAIL: mean skill score across outputs <= 0. Diagnostic dump:")
        worst = sorted(agg_skill.items(), key=lambda kv: kv[1])[:3]
        print("  worst (target, horizon) cells by mean skill:")
        for (t, lbl), s in worst:
            tr = float(np.mean([per_fold_train_rmse[(f.index, t, lbl)] for f in folds]))
            va = agg_rmse[(t, lbl)]
            ratio = va / tr if tr > 0 else float("inf")
            print(
                f"    {t} {lbl}: skill={s:+.4f}  train_rmse={tr:.3f}  val_rmse={va:.3f}  "
                f"val/train={ratio:.2f}  (>>1 = overfit)"
            )
        print("  see MLflow diagnostics/feature_importances.json for top features per cell.")
        return 1

    print("ok — mean skill > 0.")
    return 0


def _stringify_keys(d: dict) -> dict:
    return {f"fold{k[0]}.{k[1]}.{k[2]}": v for k, v in d.items()}


def _iso_ts(ts) -> str:
    return pd.Timestamp(ts).isoformat().replace("+00:00", "Z")


def _load_persistence_promo_rmse(
    run_id: str, targets: list[str], horizon_labels: list[str]
) -> dict[tuple[str, str], float]:
    """Return {(target, horizon_label): rmse} from a persistence run's promo metrics.

    Mirrors `load_per_fold_rmse` but for the promo-window metrics added in T9.
    Raises if any required cell is missing — promote.py depends on it.
    """
    run = mlflow.get_run(run_id)
    out: dict[tuple[str, str], float] = {}
    for t in targets:
        for lbl in horizon_labels:
            key = f"promo.rmse.{t}.{lbl}"
            if key not in run.data.metrics:
                raise RuntimeError(
                    f"persistence run {run_id} is missing {key!r} — "
                    "re-run T4 with T9 changes applied"
                )
            out[(t, lbl)] = float(run.data.metrics[key])
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost candidate (T5)")
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
