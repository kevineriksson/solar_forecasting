"""Query MLflow for the latest run of each model_type and emit a markdown table
of MAE / RMSE / skill at both horizons for all three targets.

Used by `make report` (and indirectly by docs/report.md) to keep the results
section in sync with whatever's currently in MLflow.

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000   # port-forward MLflow first
    python -m scripts.build_report               # prints markdown to stdout
    python -m scripts.build_report -o results.md # writes to file

Assumes:
  * Experiment "solar_forecaster" exists (created by Stage 3 training).
  * Each run has a `model_type` tag in {persistence, xgb, lstm}.
  * Metrics follow the schema  mean.{mae|rmse|skill}.{target}.{horizon_label}.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

LOG = logging.getLogger("build_report")

EXPERIMENT_NAME = "solar_forecaster"
# Tag values used at training time, in column-display order. xgb_train.py tags
# runs as "xgboost", not "xgb" — keep this list in sync with the tag setters
# in src/models/{train_persistence,xgb_train,lstm_train}.py.
MODEL_TYPES = ["persistence", "xgboost", "lstm"]


def _load_axes(params_path: Path) -> tuple[list[str], list[str]]:
    params = yaml.safe_load(params_path.read_text())
    return params["forecast"]["targets"], params["forecast"]["horizon_labels"]


def _latest_run_per_type(client: MlflowClient) -> dict[str, object]:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"MLflow experiment {EXPERIMENT_NAME!r} not found. Has the pipeline run?")
    out: dict[str, object] = {}
    for mt in MODEL_TYPES:
        # Only consider FINISHED runs — RUNNING ones may be OOM/crash victims
        # that never logged aggregate metrics, and FAILED ones we don't want
        # to surface as "results."
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_type = '{mt}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            LOG.warning("no runs for model_type=%s — leaving cells blank", mt)
            out[mt] = None
        else:
            out[mt] = runs[0]
    return out


def _cell(run, metric_key: str) -> str:
    if run is None:
        return "—"
    metrics = run.data.metrics
    if metric_key not in metrics:
        return "—"
    v = metrics[metric_key]
    # Skill scores live in [-∞, 1] but practical range is roughly [-1, 1].
    # MAE/RMSE are in W/m^2 and easily reach hundreds. Pick a precision per metric.
    if metric_key.startswith("mean.skill."):
        return f"{v:+.3f}"
    return f"{v:.1f}"


def build_table(client: MlflowClient, targets: list[str], horizons: list[str]) -> str:
    """Return a single markdown table comparing the three models.

    Layout — one block per (target, horizon), columns = model_type, rows =
    {MAE, RMSE, skill}. Compact: 6 small tables in a 3×2 grid would be hard
    to scan, so we use one wide table indexed by (target, horizon, metric).
    """
    runs = _latest_run_per_type(client)

    # Column ordering: persistence is the baseline, then xgb, then lstm.
    cols = [(mt, runs[mt]) for mt in MODEL_TYPES]
    header_row = "| Target | Horizon | Metric | " + " | ".join(mt for mt, _ in cols) + " |"
    sep_row = "|" + "|".join("---" for _ in range(3 + len(cols))) + "|"
    lines = [header_row, sep_row]

    for target in targets:
        for horizon in horizons:
            for metric in ("mae", "rmse", "skill"):
                key = f"mean.{metric}.{target}.{horizon}"
                cells = [_cell(r, key) for _, r in cols]
                lines.append(
                    f"| {target.upper()} | {horizon} | {metric.upper():<5} | "
                    + " | ".join(cells)
                    + " |"
                )
    return "\n".join(lines)


def build_tags_table(client: MlflowClient) -> str:
    """Per-model provenance: git_commit + dvc_hash from the latest run of each type.

    Demonstrates the every-model-traceable-to-commit-plus-hash invariant in
    the report itself.
    """
    runs = _latest_run_per_type(client)
    lines = ["| Model | run_id | git_commit | dvc_hash |", "|---|---|---|---|"]
    for mt in MODEL_TYPES:
        run = runs[mt]
        if run is None:
            lines.append(f"| {mt} | — | — | — |")
            continue
        tags = run.data.tags
        git_sha = tags.get("git_commit", "?")[:8]
        dvc = tags.get("dvc_hash", "?")[:12]
        lines.append(f"| {mt} | `{run.info.run_id[:8]}` | `{git_sha}` | `{dvc}` |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    targets, horizons = _load_axes(Path(args.params))
    client = MlflowClient()

    metrics_md = build_table(client, targets, horizons)
    tags_md = build_tags_table(client)

    body = (
        "## Results — latest MLflow run per model_type\n\n"
        f"{metrics_md}\n\n"
        "### Reproducibility tags\n\n"
        f"{tags_md}\n"
    )

    if args.output:
        Path(args.output).write_text(body)
        LOG.info("wrote %s (%d bytes)", args.output, len(body))
    else:
        sys.stdout.write(body)
        sys.stdout.write("\n")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the T14 MLflow results table.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "-o", "--output", default=None, help="write markdown to file (default: stdout)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
