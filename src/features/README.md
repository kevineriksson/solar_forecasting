# src/features — Stage 2

Build the feature table from validated interim data and emit per-split Parquet.

Pipeline order (`pipeline.build_features`):

1. **Clear-sky** (`cs_ghi`, `cs_dni`, `cs_dhi`) via pvlib Ineichen for the fixed
   Wolf Point site (lat 48.30783, lon -105.1017, alt 640 m).
2. **`k_t`** = `ghi / cs_ghi`, set to 0 at night (`cs_ghi ≤ 1e-3`),
   clipped to `[features.kt_clip_min, features.kt_clip_max]`.
3. **`is_night`** mask from `zenith ≥ features.night_zenith_threshold_deg`.
4. **Lags** for `features.lagged_variables` at `features.lags_steps`.
   `_lagK` at row *t* is the value at *t-K* (strictly backward).
5. **Rolling means** for the same variables at `features.rolling_means_steps`.
   Window at row *t* covers `[t-W, t-1]` (excludes *t*; `min_periods=W`).
6. **Calendar**: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` from the UTC index.
7. Drop the leading warmup window
   (`max(lags ∪ rolling)` rows; 12 rows with current params).
8. Assign splits using the same `SplitManifest` as Stage 1 → write
   `data/features/{train,promo,replay}.parquet`.

### Outputs

| Path | Description |
|---|---|
| `data/features/train.parquet`    | Train split features (~655 k rows) |
| `data/features/promo.parquet`    | Promotion-validation features (~5.8 k rows) |
| `data/features/replay.parquet`   | Replay features (~17 k rows) |
| `data/features/features_manifest.json` | Column list, row counts, warmup-drop count |

### Run locally

```bash
python -m src.features.main --params params.yaml
# or via DVC (cached, hash-checked, deterministic):
dvc repro features
```

### Determinism

The stage produces byte-identical Parquet across re-runs (snappy, fixed
`row_group_size=64K`, fixed column order, no random ops). Verified by running
`dvc repro -f features` twice and comparing output md5s.
