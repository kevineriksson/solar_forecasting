"""Load the Production model from MLflow and return a uniform predictor.

The trainers in src/models/* persist a `model/` subtree under their MLflow
run artifacts (see T10 prereq commit). Promotion (Stage 4) registers the
winning run's artifact root as a new version of `solar_forecaster` and
transitions it to `Production`. At serve-time we resolve that version,
download `model/`, read `manifest.json`, and dispatch to a per-flavor
`ModelHandle` that exposes a uniform `.predict(payload)` signature.

This module has no FastAPI dependency — it can be exercised by tests with
no HTTP layer involved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mlflow.tracking import MlflowClient

LOG = logging.getLogger("serving.loader")

SUPPORTED_MODEL_TYPES = frozenset({"persistence", "xgboost", "lstm"})


class LoaderError(RuntimeError):
    """Raised when no usable Production model can be loaded."""


@dataclass(frozen=True)
class ModelInfo:
    """Provenance metadata surfaced via /healthz and prometheus model_info."""

    model_type: str
    version: str
    run_id: str
    git_commit: str
    dvc_hash: str
    feature_columns: tuple[str, ...]
    output_columns: tuple[tuple[str, str], ...]
    targets: tuple[str, ...]
    horizon_labels: tuple[str, ...]
    # For sequence models: how many timesteps the client must send.
    # 1 for tabular models (xgboost / persistence).
    sequence_length: int


class ModelHandle:
    """Uniform predict interface across model flavors.

    Implementations live in this file as small classes per `model_type`.
    All implementations accept a single feature dict (tabular) or a list of
    feature dicts (sequence) and return a flat dict keyed by
    ``"{target}_{horizon_label}"`` (six floats).
    """

    info: ModelInfo

    def predict(self, features: list[dict[str, float]]) -> dict[str, float]:
        """Run a single-request forecast.

        `features` is a list of per-timestep feature dicts ordered oldest-to-newest.
        Tabular models require len(features) == 1; sequence models require
        len(features) == self.info.sequence_length.
        """
        raise NotImplementedError


def load_production_model(
    model_name: str, stage: str = "Production", client: MlflowClient | None = None
) -> ModelHandle:
    """Resolve `solar_forecaster@Production`, download `model/`, build a handle."""
    if client is None:
        client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise LoaderError(f"no {stage} version registered for model {model_name!r}")
    if len(versions) > 1:
        # MLflow guarantees at most one per stage when archive_existing_versions=True
        # is used (promote.py does this), but be defensive.
        LOG.warning(
            "found %d %s versions for %s; using highest version number",
            len(versions),
            stage,
            model_name,
        )
    mv = max(versions, key=lambda v: int(v.version))
    run_id = mv.run_id
    if not run_id:
        raise LoaderError(f"{model_name} version {mv.version} has no source run_id")

    # MLflow's ModelVersion.tags is already a dict[str, str] (string -> string).
    mv_tags_raw = client.get_model_version(model_name, mv.version).tags
    tags: dict[str, str] = dict(mv_tags_raw or {})
    # Older registry entries may not carry tags on the version; fall back to the
    # underlying training run, which always has the reproducibility tags.
    if "git_commit" not in tags or "dvc_hash" not in tags:
        run = client.get_run(run_id)
        for k in ("git_commit", "dvc_hash", "model_type"):
            tags.setdefault(k, run.data.tags.get(k, ""))

    LOG.info(
        "resolving %s v%s  run_id=%s  model_type=%s",
        model_name,
        mv.version,
        run_id,
        tags.get("model_type", "?"),
    )

    local_model_dir = Path(client.download_artifacts(run_id, "model"))
    manifest_path = local_model_dir / "manifest.json"
    if not manifest_path.exists():
        raise LoaderError(
            f"{model_name} v{mv.version} run {run_id} has no model/manifest.json — "
            "trainer did not save the model artifact subtree"
        )
    manifest = json.loads(manifest_path.read_text())
    model_type = manifest["model_type"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise LoaderError(f"unsupported model_type {model_type!r} in manifest")

    info = ModelInfo(
        model_type=model_type,
        version=str(mv.version),
        run_id=run_id,
        git_commit=tags.get("git_commit", ""),
        dvc_hash=tags.get("dvc_hash", ""),
        feature_columns=tuple(manifest["feature_columns"]),
        output_columns=tuple((t, lbl) for t, lbl in manifest["output_columns"]),
        targets=tuple(manifest["targets"]),
        horizon_labels=tuple(manifest["horizon_labels"]),
        sequence_length=int(manifest.get("sequence_length_steps", 1)),
    )

    if model_type == "xgboost":
        return _XGBHandle.from_dir(local_model_dir, manifest, info)
    if model_type == "lstm":
        return _LSTMHandle.from_dir(local_model_dir, manifest, info)
    if model_type == "persistence":
        return _PersistenceHandle.from_dir(local_model_dir, manifest, info)
    raise LoaderError(f"no handle implementation for model_type {model_type!r}")  # unreachable


def _features_to_row(features: list[dict[str, float]], info: ModelInfo) -> np.ndarray:
    """Validate a tabular request and return a (1, F) array in feature order."""
    if len(features) != 1:
        raise ValueError(
            f"tabular model expects exactly 1 timestep of features, got {len(features)}"
        )
    row = features[0]
    missing = [c for c in info.feature_columns if c not in row]
    if missing:
        raise ValueError(f"missing features: {missing}")
    return np.asarray([[float(row[c]) for c in info.feature_columns]], dtype=np.float64)


# ---------- XGBoost ----------


class _XGBHandle(ModelHandle):
    def __init__(
        self,
        info: ModelInfo,
        boosters: dict[tuple[str, str], Any],
    ) -> None:
        self.info = info
        self._boosters = boosters

    @classmethod
    def from_dir(cls, model_dir: Path, manifest: dict, info: ModelInfo) -> _XGBHandle:
        import xgboost as xgb

        boosters: dict[tuple[str, str], Any] = {}
        for key, fname in manifest["model_files"].items():
            target, lbl = key.split(".", 1)
            b = xgb.Booster()
            b.load_model(str(model_dir / fname))
            boosters[(target, lbl)] = b
        return cls(info, boosters)

    def predict(self, features: list[dict[str, float]]) -> dict[str, float]:
        import xgboost as xgb

        row = _features_to_row(features, self.info)
        dmat = xgb.DMatrix(row, feature_names=list(self.info.feature_columns))
        out: dict[str, float] = {}
        for (t, lbl), b in self._boosters.items():
            y = float(b.predict(dmat)[0])
            out[f"{t}_{lbl}"] = y
        return out


# ---------- LSTM ----------


class _LSTMHandle(ModelHandle):
    def __init__(
        self,
        info: ModelInfo,
        model: Any,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.info = info
        self._model = model
        self._x_mean = x_mean.astype(np.float32)
        self._x_std = x_std.astype(np.float32)
        self._y_mean = y_mean.astype(np.float32)
        self._y_std = y_std.astype(np.float32)

    @classmethod
    def from_dir(cls, model_dir: Path, manifest: dict, info: ModelInfo) -> _LSTMHandle:
        import torch

        from src.models.lstm_model import LSTMRegressor

        arch = json.loads((model_dir / "arch.json").read_text())
        model = LSTMRegressor(
            n_features=int(arch["n_features"]),
            hidden_size=int(arch["hidden_size"]),
            num_layers=int(arch["num_layers"]),
            dropout=float(arch["dropout"]),
            n_outputs=int(arch["n_outputs"]),
        )
        state = torch.load(str(model_dir / "state_dict.pt"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        xs = json.loads((model_dir / "x_scaler.json").read_text())
        ys = json.loads((model_dir / "y_scaler.json").read_text())
        return cls(
            info,
            model,
            x_mean=np.asarray(xs["mean"], dtype=np.float32),
            x_std=np.asarray(xs["std"], dtype=np.float32),
            y_mean=np.asarray(ys["mean"], dtype=np.float32),
            y_std=np.asarray(ys["std"], dtype=np.float32),
        )

    def predict(self, features: list[dict[str, float]]) -> dict[str, float]:
        import torch

        L = self.info.sequence_length
        if len(features) != L:
            raise ValueError(f"lstm model expects {L} timesteps of features, got {len(features)}")
        # Validate columns on the first row; assume the rest match (cheaper).
        missing = [c for c in self.info.feature_columns if c not in features[0]]
        if missing:
            raise ValueError(f"missing features: {missing}")

        X = np.asarray(
            [[float(row[c]) for c in self.info.feature_columns] for row in features],
            dtype=np.float32,
        )
        X = (X - self._x_mean) / self._x_std  # (L, F)
        X = X[np.newaxis, :, :]  # (1, L, F)
        with torch.no_grad():
            y_std_pred = self._model(torch.from_numpy(X)).cpu().numpy()[0]  # (n_outputs,)
        y_raw = y_std_pred * self._y_std + self._y_mean

        out: dict[str, float] = {}
        for j, (t, lbl) in enumerate(self.info.output_columns):
            out[f"{t}_{lbl}"] = float(y_raw[j])
        return out


# ---------- Persistence ----------


class _PersistenceHandle(ModelHandle):
    def __init__(
        self,
        info: ModelInfo,
        kt_clip_min: float,
        kt_clip_max: float,
        target_to_cs_at_t: dict[str, str],
    ) -> None:
        self.info = info
        self._kt_min = float(kt_clip_min)
        self._kt_max = float(kt_clip_max)
        self._cs_at_t = dict(target_to_cs_at_t)

    @classmethod
    def from_dir(cls, model_dir: Path, manifest: dict, info: ModelInfo) -> _PersistenceHandle:
        return cls(
            info,
            kt_clip_min=manifest["kt_clip_min"],
            kt_clip_max=manifest["kt_clip_max"],
            target_to_cs_at_t=manifest["target_to_cs_at_t"],
        )

    def predict(self, features: list[dict[str, float]]) -> dict[str, float]:
        if len(features) != 1:
            raise ValueError(
                f"persistence model expects 1 timestep of features, got {len(features)}"
            )
        row = features[0]
        missing = [c for c in self.info.feature_columns if c not in row]
        if missing:
            raise ValueError(f"missing features: {missing}")

        out: dict[str, float] = {}
        for t in self.info.targets:
            cs_now_col = self._cs_at_t[t]
            cs_now = float(row[cs_now_col])
            y_now = float(row[t])
            if cs_now > 1e-6:
                k = y_now / cs_now
                k = max(self._kt_min, min(self._kt_max, k))
            else:
                k = 0.0
            for lbl in self.info.horizon_labels:
                cs_h = float(row[f"cs_{t}_h{lbl}"])
                out[f"{t}_{lbl}"] = k * cs_h
        return out
