"""Unit tests for src/models/lstm_model.py — window/alignment + smoke test.

The full LSTM training entrypoint is exercised by the T6 run; these tests pin
down the cheap invariants that would silently break the skill-score
comparison or introduce future-leakage if violated.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.lstm_model import (
    FeatureScaler,
    LSTMRegressor,
    SequenceWindowDataset,
    build_sequences,
    standardize_features,
    valid_anchor_range,
)


def test_valid_anchor_range_inclusive_exclusive():
    # block_start=0, block_end=20, seq_len=4, max_h=2
    # earliest anchor: 0 + 4 - 1 = 3
    # latest anchor (exclusive): 20 - 2 = 18
    a, b = valid_anchor_range(0, 20, sequence_length=4, max_horizon=2)
    assert (a, b) == (3, 18)


def test_valid_anchor_range_empty_when_block_too_small():
    a, b = valid_anchor_range(0, 4, sequence_length=8, max_horizon=2)
    assert a == b  # empty


def test_build_sequences_alignment():
    # 10 rows, 2 features, 1 target, seq_len=3, horizons=(1, 4).
    n = 10
    features = np.arange(n * 2, dtype=np.float64).reshape(n, 2)  # rows: 0/1, 2/3, ...
    targets = (np.arange(n, dtype=np.float64) * 100.0).reshape(n, 1)
    seq_len = 3
    horizons = (1, 4)

    a_start, a_end = valid_anchor_range(0, n, seq_len, max(horizons))
    # earliest anchor: 0 + 3 - 1 = 2; latest exclusive: 10 - 4 = 6
    assert (a_start, a_end) == (2, 6)
    anchors = np.arange(a_start, a_end)  # [2, 3, 4, 5]

    X, Y = build_sequences(features, targets, anchors, seq_len, horizons)
    assert X.shape == (len(anchors), seq_len, 2)
    assert Y.shape == (len(anchors), 1 * len(horizons))

    # X[0] is the window ending at anchor 2 inclusive -> rows 0,1,2.
    assert np.allclose(X[0], features[0:3])
    # X[3] is the window ending at anchor 5 inclusive -> rows 3,4,5.
    assert np.allclose(X[3], features[3:6])

    # Y for anchor 2: target[2+1]=300, target[2+4]=600.
    assert Y[0, 0] == 300.0
    assert Y[0, 1] == 600.0
    # Y for anchor 5: target[5+1]=600, target[5+4]=900.
    assert Y[3, 0] == 600.0
    assert Y[3, 1] == 900.0


def test_build_sequences_no_future_leakage_in_inputs():
    # The last timestep of the window must be at the anchor row itself,
    # never further into the future. If anyone "fixes" this to include
    # row t+1 in the window, this test must scream.
    n = 12
    features = np.arange(n, dtype=np.float64).reshape(n, 1)
    targets = np.zeros((n, 1), dtype=np.float64)
    seq_len = 4
    anchors = np.array([5, 7], dtype=np.int64)
    X, _ = build_sequences(features, targets, anchors, seq_len, (1,))
    # Last timestep of X[0] is feature[5]; X[1] is feature[7].
    assert X[0, -1, 0] == 5.0
    assert X[1, -1, 0] == 7.0
    # First timestep covers t-seq_len+1.
    assert X[0, 0, 0] == 2.0  # 5 - 4 + 1
    assert X[1, 0, 0] == 4.0  # 7 - 4 + 1


def test_build_sequences_multitarget_output_order_is_target_major():
    # 2 targets, 2 horizons -> 4 outputs in order:
    # (t0, h0), (t0, h1), (t1, h0), (t1, h1)
    n = 8
    features = np.zeros((n, 1), dtype=np.float64)
    targets = np.column_stack(
        [
            np.arange(n, dtype=np.float64),  # target 0: 0, 1, 2, ..., 7
            np.arange(n, dtype=np.float64) + 1000.0,  # target 1: 1000, 1001, ...
        ]
    )
    seq_len = 1
    horizons = (1, 2)
    anchors = np.array([2, 3], dtype=np.int64)
    _, Y = build_sequences(features, targets, anchors, seq_len, horizons)
    # anchor 2: target0[3]=3, target0[4]=4, target1[3]=1003, target1[4]=1004
    assert Y[0].tolist() == [3.0, 4.0, 1003.0, 1004.0]
    # anchor 3: target0[4]=4, target0[5]=5, target1[4]=1004, target1[5]=1005
    assert Y[1].tolist() == [4.0, 5.0, 1004.0, 1005.0]


def test_feature_scaler_fits_on_train_only_then_transforms_val():
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
    val = rng.normal(loc=12.0, scale=0.5, size=(200, 3))  # very different distribution

    scaler = FeatureScaler.fit(train)
    train_z = (train - scaler.mean) / scaler.std
    val_z = (val - scaler.mean) / scaler.std

    # Train scaled has mean~0, std~1.
    assert np.allclose(train_z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(train_z.std(axis=0, ddof=0), 1.0, atol=1e-6)
    # Val scaled does NOT have mean 0 — proving train stats were used, not val.
    assert np.linalg.norm(val_z.mean(axis=0)) > 1.0


def test_feature_scaler_handles_constant_column():
    # Constant column would normally divide by zero — scaler must guard it.
    arr = np.column_stack([np.zeros(50), np.arange(50, dtype=np.float64)])
    scaler = FeatureScaler.fit(arr)
    # No NaN/inf after scaling, even for the constant column.
    z = (arr - scaler.mean) / scaler.std
    assert np.isfinite(z).all()


def test_standardize_features_shape_preserved():
    arr_seq = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    flat = arr_seq.reshape(-1, 4)
    scaler = FeatureScaler.fit(flat)
    z = standardize_features(arr_seq, scaler)
    assert z.shape == arr_seq.shape
    assert z.dtype == np.float32


def test_sequence_window_dataset_matches_build_sequences():
    # On-the-fly windowing must produce items identical to batched
    # build_sequences output (after standardization), or the trainer would
    # silently use a different alignment than the unit tests check.
    n = 20
    rng = np.random.default_rng(7)
    features = rng.normal(size=(n, 3)).astype(np.float64)
    targets = rng.normal(size=(n, 2)).astype(np.float64)
    seq_len = 5
    horizons = (1, 3)

    a_start, a_end = valid_anchor_range(0, n, seq_len, max(horizons))
    anchors = np.arange(a_start, a_end, dtype=np.int64)

    # Reference: materialize then standardize.
    X_ref, Y_ref = build_sequences(features, targets, anchors, seq_len, horizons)
    train_block = features[:a_end]  # any consistent training block
    scaler = FeatureScaler.fit(train_block)
    X_ref_z = standardize_features(X_ref, scaler)
    y_mean = np.zeros(Y_ref.shape[1], dtype=np.float32)
    y_std = np.ones(Y_ref.shape[1], dtype=np.float32)

    ds = SequenceWindowDataset(
        features,
        targets,
        anchors,
        seq_len,
        horizons,
        x_mean=scaler.mean,
        x_std=scaler.std,
        y_mean=y_mean,
        y_std=y_std,
    )
    assert len(ds) == len(anchors)
    for i in range(len(ds)):
        x, y = ds[i]
        assert np.allclose(x.numpy(), X_ref_z[i], atol=1e-6)
        # y_mean/y_std are identity here, so y should equal Y_ref[i] in float32.
        assert np.allclose(y.numpy(), Y_ref[i].astype(np.float32), atol=1e-6)


def test_sequence_window_dataset_label_standardization():
    n = 10
    features = np.zeros((n, 1), dtype=np.float64)
    targets = np.arange(n, dtype=np.float64).reshape(n, 1)  # 0..9
    seq_len = 1
    horizons = (1,)
    anchors = np.array([3, 5], dtype=np.int64)
    y_mean = np.array([2.0], dtype=np.float32)
    y_std = np.array([4.0], dtype=np.float32)
    ds = SequenceWindowDataset(
        features,
        targets,
        anchors,
        seq_len,
        horizons,
        x_mean=np.zeros(1),
        x_std=np.ones(1),
        y_mean=y_mean,
        y_std=y_std,
    )
    _, y0 = ds[0]  # anchor=3, target[4]=4 -> (4 - 2) / 4 = 0.5
    _, y1 = ds[1]  # anchor=5, target[6]=6 -> (6 - 2) / 4 = 1.0
    assert float(y0[0]) == pytest.approx(0.5)
    assert float(y1[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_lstm_regressor_forward_shape(num_layers: int):
    torch.manual_seed(0)
    n_features = 5
    n_outputs = 6
    seq_len = 4
    batch = 7
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=8,
        num_layers=num_layers,
        dropout=0.1,
        n_outputs=n_outputs,
    )
    x = torch.randn(batch, seq_len, n_features)
    y = model(x)
    assert y.shape == (batch, n_outputs)
    assert torch.isfinite(y).all()
