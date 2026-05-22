"""Rolling-origin cross-validation folds.

Shared by T4 (persistence), T5 (XGBoost), T6 (LSTM) so skill-score comparisons
are apples-to-apples — every model is scored on the exact same validation
windows over the train split.

Scheme (expanding window, walking backward from the end of the train split):

    For fold i in [0..n_splits-1]:
        val_end_i   = N - i * test_size_steps
        val_start_i = val_end_i - test_size_steps
        train_end_i = val_start_i - gap_steps
        train_start = 0                                   # expanding

Folds are returned in chronological order (oldest validation window first), so
fold 0 has the LEAST training data and the EARLIEST validation window.

All index pairs are half-open [start, end) Python slices over a contiguous,
chronologically sorted DataFrame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "n_train": self.train_end - self.train_start,
            "n_val": self.val_end - self.val_start,
        }


def rolling_origin_folds(n_rows: int, cv_cfg: Mapping[str, object]) -> list[Fold]:
    """Generate expanding-window folds from `training.cv` params.

    Raises if the requested folds don't fit in `n_rows`.
    """
    scheme = str(cv_cfg["scheme"])
    if scheme != "rolling_origin":
        raise ValueError(f"unsupported cv scheme {scheme!r}; expected 'rolling_origin'")

    n_splits = int(cv_cfg["n_splits"])  # type: ignore[call-overload]
    test_size = int(cv_cfg["test_size_steps"])  # type: ignore[call-overload]
    gap = int(cv_cfg["gap_steps"])  # type: ignore[call-overload]

    if n_splits <= 0 or test_size <= 0 or gap < 0:
        raise ValueError("n_splits and test_size_steps must be > 0; gap_steps >= 0")

    # Walking backward: the oldest fold must still have a positive-size train window.
    earliest_val_start = n_rows - n_splits * test_size
    earliest_train_end = earliest_val_start - gap
    if earliest_train_end <= 0:
        raise ValueError(
            f"rolling-origin CV does not fit: n_rows={n_rows}, n_splits={n_splits}, "
            f"test_size={test_size}, gap={gap}; "
            f"earliest train_end would be {earliest_train_end}"
        )

    folds: list[Fold] = []
    # Build backward then reverse so we return chronological order.
    for i in range(n_splits):
        val_end = n_rows - i * test_size
        val_start = val_end - test_size
        train_end = val_start - gap
        folds.append(
            Fold(
                index=0,  # reassigned below
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
        )

    folds.reverse()
    return [
        Fold(
            index=i,
            train_start=f.train_start,
            train_end=f.train_end,
            val_start=f.val_start,
            val_end=f.val_end,
        )
        for i, f in enumerate(folds)
    ]
