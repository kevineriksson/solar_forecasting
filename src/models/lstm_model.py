"""LSTM model + sequence/window helpers for T6.

Sequence convention (must match the unit tests in test_models_lstm.py):

    For "as-of time" row t in a chronologically sorted feature frame,
    the LSTM input is the feature matrix over rows [t - L + 1 .. t]
    inclusive (L = sequence_length_steps). The labels are the target
    values at t + h for each horizon h.

    No future leakage: features end strictly at t; labels live strictly
    at t + h > t.

The training-time valid row range per contiguous block is
    t in [block_start + L - 1, block_end - max_horizon)
because the earliest t needing L history rows is L - 1, and the latest t
must leave room for the longest label horizon.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSpec:
    """Slice bookkeeping for sequence/label tensors."""

    sequence_length: int
    horizons: tuple[int, ...]
    n_features: int
    n_targets: int

    @property
    def n_outputs(self) -> int:
        return self.n_targets * len(self.horizons)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


def valid_anchor_range(
    block_start: int, block_end: int, sequence_length: int, max_horizon: int
) -> tuple[int, int]:
    """Inclusive-exclusive [a_start, a_end) range of valid anchor rows t.

    `block_*` are half-open row indices into the full frame; the lookback
    must fit fully inside the block (so we don't peek before the split)
    and the longest label must also live inside the block.
    """
    a_start = block_start + sequence_length - 1
    a_end = block_end - max_horizon
    if a_end <= a_start:
        return (a_start, a_start)
    return (a_start, a_end)


def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    anchors: np.ndarray,
    sequence_length: int,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize (X_seq, Y) for the given anchor rows.

    Parameters
    ----------
    features : (N, F) float array of feature values, chronologically sorted.
    targets  : (N, T) float array of target values, aligned to `features`.
    anchors  : (K,)   int  array of valid anchor rows t (each satisfies
               t - sequence_length + 1 >= 0 and t + max(horizons) < N).
    sequence_length : L.
    horizons : tuple of integer steps ahead.

    Returns
    -------
    X_seq : (K, L, F) float32 — feature window ending at t inclusive.
    Y     : (K, T * H) float32 — flattened labels, target-major:
            [tgt0_h0, tgt0_h1, ..., tgt1_h0, ...].
    """
    L = int(sequence_length)
    H = tuple(int(h) for h in horizons)
    K = anchors.shape[0]
    F = features.shape[1]
    T = targets.shape[1]

    X_seq = np.empty((K, L, F), dtype=np.float32)
    # Build via fancy indexing on the time axis: (K, L) -> features (K, L, F).
    offsets = np.arange(-L + 1, 1)  # length L, values [-L+1 .. 0]
    rows = anchors[:, None] + offsets[None, :]  # (K, L)
    X_seq = features[rows].astype(np.float32, copy=False)

    Y = np.empty((K, T * len(H)), dtype=np.float32)
    col = 0
    for t_idx in range(T):
        for h in H:
            Y[:, col] = targets[anchors + h, t_idx].astype(np.float32, copy=False)
            col += 1
    return X_seq, Y


@dataclass
class FeatureScaler:
    """Per-column z-score scaler fit on a training block only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> FeatureScaler:
        m = x.mean(axis=0)
        s = x.std(axis=0, ddof=0)
        # Avoid divide-by-zero on constant columns.
        s = np.where(s > 1e-12, s, 1.0)
        return cls(mean=m.astype(np.float64), std=s.astype(np.float64))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def standardize_features(arr_seq: np.ndarray, scaler: FeatureScaler) -> np.ndarray:
    """Apply a per-feature z-score to a (K, L, F) sequence tensor in float32."""
    out = (arr_seq.astype(np.float32) - scaler.mean.astype(np.float32)) / scaler.std.astype(
        np.float32
    )
    return out


def column_indices(columns: Sequence[str], names: Sequence[str]) -> list[int]:
    name_to_idx = {c: i for i, c in enumerate(columns)}
    return [name_to_idx[n] for n in names]


def to_float_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Return df[cols] as a contiguous (N, len(cols)) float64 NumPy array."""
    return np.ascontiguousarray(df[list(cols)].to_numpy(dtype=np.float64))


class SequenceWindowDataset(Dataset):
    """Builds (X_seq, Y_std) windows on the fly from shared arrays.

    Memory: O(N * F + N * T) for the underlying features and targets — a
    sequence window is materialized per item (size L * F floats). For our
    setup that's a 1.05 MB tensor per batch of 256, versus 2.65 GB if all
    640k windows were materialized up front.

    Standardization is folded in here: features get z-scored by the train
    block's scaler, labels get z-scored by the train block's target mean/std.
    Both produce float32 output ready for the model.

    Anchors must already be filtered to valid rows (each anchor t satisfies
    t - sequence_length + 1 >= 0 and t + max(horizons) < N).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        anchors: np.ndarray,
        sequence_length: int,
        horizons: Sequence[int],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.features = features
        self.targets = targets
        self.anchors = np.ascontiguousarray(anchors, dtype=np.int64)
        self.L = int(sequence_length)
        self.H = tuple(int(h) for h in horizons)
        # Cast scaling stats once.
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)

    def __len__(self) -> int:
        return int(self.anchors.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.anchors[idx])
        # Window: rows [t - L + 1 .. t] inclusive.
        x = self.features[t - self.L + 1 : t + 1].astype(np.float32, copy=False)
        x = (x - self.x_mean) / self.x_std  # (L, F)

        # Labels in target-major, horizon-inner order.
        out = np.empty(self.targets.shape[1] * len(self.H), dtype=np.float32)
        col = 0
        for t_idx in range(self.targets.shape[1]):
            for h in self.H:
                out[col] = self.targets[t + h, t_idx]
                col += 1
        out = (out - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(out)


class LSTMRegressor(nn.Module):
    """Standard multivariate LSTM regressor with a single Linear head.

    Input  : (batch, seq_len, n_features)
    Output : (batch, n_outputs)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        n_outputs: int,
    ) -> None:
        super().__init__()
        # PyTorch warns when dropout is set on a 1-layer LSTM; suppress that
        # by zeroing it in the constructor (still applied via self.head_dropout).
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(n_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head_dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden_size), int(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.head_dropout(last))
