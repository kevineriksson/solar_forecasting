"""Rolling statistics for the replay client (T12).

Two small primitives used by the replay loop to drive Grafana panels and the
two PrometheusRule alerts (drift_high, skill_score_low):

  * :class:`RollingResidualWindow` — fixed-size deques of model and naive-
    persistence residuals per ``(target, horizon)``; computes rolling MAE,
    RMSE, and skill score on demand.

  * :class:`FeaturePSI` — snapshots a reference distribution per feature from
    the training split, then maintains a rolling sample of recent values and
    computes the Population Stability Index against the reference.

Both are deliberately stdlib-only. The serving and replay pods scrape at 15 s
intervals, and the window defaults to 96 steps (24 h of 15-min simulated
ticks), so recomputing per step costs O(window) which is trivially cheap.

PSI bin edges are frozen at construction time so subsequent samples are
binned against the same fixed grid the reference was binned on.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

# Smoothing constant: PSI explodes when an empty bin appears on either side
# of the comparison. Replacing zero proportions with this floor caps the
# log term and matches the convention used in most production drift suites.
_PSI_EPSILON = 1e-4


@dataclass
class _PairBuffers:
    model: deque[float]
    persistence: deque[float]


class RollingResidualWindow:
    """Trailing-window MAE / RMSE / skill score per (target, horizon).

    Keeps two parallel deques per cell:
      * ``model`` — residual = prediction - truth from the served model
      * ``persistence`` — residual from smart persistence (kₜ × cs_y(t+h))

    Skill score is ``1 - RMSE_model / RMSE_persistence``. Returns ``NaN`` for
    a given cell until both deques are at least ``min_samples`` full; we
    refuse to publish a noisy skill estimate over 2 points.
    """

    def __init__(
        self,
        window_steps: int,
        targets: Iterable[str],
        horizon_labels: Iterable[str],
        min_samples: int = 16,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self._window = int(window_steps)
        self._min_samples = int(min_samples)
        self._cells: dict[tuple[str, str], _PairBuffers] = {
            (t, h): _PairBuffers(
                model=deque(maxlen=self._window),
                persistence=deque(maxlen=self._window),
            )
            for t in targets
            for h in horizon_labels
        }

    @property
    def window_steps(self) -> int:
        return self._window

    def observe(
        self,
        target: str,
        horizon_label: str,
        model_residual: float,
        persistence_residual: float,
    ) -> None:
        cell = self._cells.get((target, horizon_label))
        if cell is None:
            return
        cell.model.append(float(model_residual))
        cell.persistence.append(float(persistence_residual))

    def fill_ratio(self) -> float:
        """Average window-fill ratio across cells, in [0, 1]."""
        if not self._cells:
            return 0.0
        total = 0.0
        for cell in self._cells.values():
            total += min(len(cell.model), len(cell.persistence)) / self._window
        return total / len(self._cells)

    def mae(self, target: str, horizon_label: str) -> float:
        cell = self._cells.get((target, horizon_label))
        if cell is None or len(cell.model) < self._min_samples:
            return float("nan")
        arr = np.fromiter((abs(v) for v in cell.model), dtype=np.float64)
        return float(arr.mean())

    def rmse(self, target: str, horizon_label: str) -> float:
        cell = self._cells.get((target, horizon_label))
        if cell is None or len(cell.model) < self._min_samples:
            return float("nan")
        arr = np.fromiter((v * v for v in cell.model), dtype=np.float64)
        return float(math.sqrt(arr.mean()))

    def skill(self, target: str, horizon_label: str) -> float:
        cell = self._cells.get((target, horizon_label))
        if cell is None:
            return float("nan")
        if len(cell.model) < self._min_samples or len(cell.persistence) < self._min_samples:
            return float("nan")
        m = np.fromiter((v * v for v in cell.model), dtype=np.float64)
        p = np.fromiter((v * v for v in cell.persistence), dtype=np.float64)
        rmse_m = math.sqrt(m.mean())
        rmse_p = math.sqrt(p.mean())
        if rmse_p <= 0.0:
            # Persistence is perfect on the window (e.g. all-night). Treat the
            # cell as uninformative rather than emit a divide-by-zero spike.
            return float("nan")
        return float(1.0 - rmse_m / rmse_p)


def _bin_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges with degenerate-distribution fallback."""
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return np.array([-1.0, 0.0, 1.0])
    # Quantile edges produce roughly-equal-mass bins, which is what PSI wants.
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        # Constant feature or near-degenerate — widen synthetically so binning works.
        center = float(edges[0]) if edges.size else 0.0
        edges = np.array([center - 1.0, center, center + 1.0])
    # Open the outer edges so future samples that exceed the training range
    # still land in the first / last bin instead of being dropped.
    edges = edges.astype(np.float64).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _histogram_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin ``values`` into ``edges`` and return per-bin proportions (sum = 1)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        # All-NaN — return uniform proportions so PSI vs a real reference is finite.
        n_bins = max(1, edges.size - 1)
        return np.full(n_bins, 1.0 / n_bins)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        n_bins = counts.size
        return np.full(n_bins, 1.0 / n_bins)
    return counts.astype(np.float64) / float(total)


def _psi(ref_p: np.ndarray, curr_p: np.ndarray, epsilon: float = _PSI_EPSILON) -> float:
    ref_safe = np.where(ref_p <= 0, epsilon, ref_p)
    curr_safe = np.where(curr_p <= 0, epsilon, curr_p)
    return float(((curr_safe - ref_safe) * np.log(curr_safe / ref_safe)).sum())


class FeaturePSI:
    """PSI of a rolling feature window vs a frozen reference distribution.

    Construct once per replay run from the training-split DataFrame; the bin
    edges are derived from the reference and then frozen, so subsequent
    samples are always compared against the same partitioning of feature space.

    ``observe(feat, value)`` appends one sample. ``psi(feat)`` recomputes the
    statistic against the current buffer; cheap enough to call every step.
    """

    def __init__(
        self,
        reference_values: Mapping[str, Iterable[float]],
        window_steps: int,
        n_bins: int = 10,
        min_samples: int | None = None,
    ) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self._window = int(window_steps)
        self._n_bins = int(n_bins)
        self._min_samples = (
            int(min_samples) if min_samples is not None else max(2, window_steps // 4)
        )
        self._edges: dict[str, np.ndarray] = {}
        self._ref_p: dict[str, np.ndarray] = {}
        self._buffer: dict[str, deque[float]] = {}
        for feat, values in reference_values.items():
            arr = np.asarray(list(values), dtype=np.float64)
            edges = _bin_edges(arr, self._n_bins)
            self._edges[feat] = edges
            self._ref_p[feat] = _histogram_proportions(arr, edges)
            self._buffer[feat] = deque(maxlen=self._window)

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self._buffer.keys())

    def observe(self, feature: str, value: float) -> None:
        buf = self._buffer.get(feature)
        if buf is None:
            return
        if value is None or not math.isfinite(float(value)):
            return
        buf.append(float(value))

    def psi(self, feature: str) -> float:
        buf = self._buffer.get(feature)
        if buf is None or len(buf) < self._min_samples:
            return float("nan")
        values = np.fromiter(buf, dtype=np.float64)
        curr_p = _histogram_proportions(values, self._edges[feature])
        return _psi(self._ref_p[feature], curr_p)
