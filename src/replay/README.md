# src/replay — Stage 6 replay client

Walks the 6-month replay window forward in simulated time, builds the
feature vector exactly as Stage 2 would have at each timestamp, posts it
to the serving `/predict` endpoint, and emits residual = prediction - truth
metrics that Prometheus scrapes from the replay client's own `/metrics`.

## Modules

| File | Purpose |
|---|---|
| `features.py` | `LeakageError`, `build_features_as_of`, `ReplaySource`, `enrich_for_persistence`. The as-of-t guard and the runtime feature accessor for the loop. |
| `metrics.py` | Prometheus instruments: residual / |residual| / prediction / truth histograms, request latency, progress + simulated-clock gauges, an info gauge with model provenance. |
| `client.py` | Argparse + `/healthz` probe + main loop. Throttled by `--speedup`. |

## How the no-future-leakage guarantee works

The pre-computed `data/features/replay.parquet` already contains, for every
timestamp `t`, the exact Stage 2 feature row the trainer would have seen at
`t`: lags reference `t - k*Δt`, rolling means reference `(t - n*Δt, t]`, and
calendar features depend only on `t` itself. So the runtime loop just reads
the pre-computed row at `t` — no future data can possibly appear in it.

`build_features_as_of(history_df, t, cfg)` makes this explicit. It refuses
(via `LeakageError`) any call where `history_df` contains rows with
`timestamp > t`. The leakage test in `tests/unit/test_replay_features.py`
exercises both the guard and the equivalence with the full Stage 2 pipeline.

## Running locally

```bash
# In one terminal: forward the serving Service.
kubectl port-forward -n solar svc/solar-serve 18000:80

# In another: run the client against the local features file.
python -m src.replay.client \
    --endpoint http://localhost:18000 \
    --features data/features/replay.parquet \
    --speedup 200 --max-requests 500 --metrics-port 19090

# Scrape the client's metrics.
curl -s http://localhost:19090/metrics | grep ^solar_replay_
```

For in-cluster deployment see [`k8s/replay/`](../../k8s/replay/).
