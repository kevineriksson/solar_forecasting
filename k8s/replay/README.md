# k8s/replay — Stage 6 simulated traffic

The replay client walks the 6-month replay window, posts each step to the
serving Service, looks up the in-window ground truth at `t + horizon`, and
emits per-target/per-horizon residual histograms. Prometheus scrapes those
from the replay pod's own `/metrics` endpoint (push-style is an anti-pattern;
we expose, Prometheus pulls).

## Manifests

| File | What it does |
|---|---|
| `service.yaml` | Headless `Service` (port 9090) targeting any pod with `app: solar-replay`. |
| `servicemonitor.yaml` | Tells the platform Prometheus to scrape every replay pod. |
| `job-demo.yaml` | One-shot `Job` — walks the full replay window once at 500 rps, then exits. The canonical "show it working" run. |
| `cronjob.yaml` | `CronJob` for ongoing simulated traffic. `suspend: true` by default — flip it on once T12/T13 are wired up. |

Both Job and CronJob pods carry `app: solar-replay`, so a single Service +
ServiceMonitor pair covers both workloads.

## Running the demo

```bash
# Build + load image, then submit the Job.
docker build -f docker/replay.Dockerfile -t solar-replay:$(git rev-parse --short HEAD) .
minikube image load solar-replay:$(git rev-parse --short HEAD)

kubectl apply -f k8s/replay/service.yaml
kubectl apply -f k8s/replay/servicemonitor.yaml
kubectl apply -f k8s/replay/job-demo.yaml

# Watch progress
kubectl logs -n solar -l job-name=solar-replay-demo -f

# After it finishes, the pod lingers for ~45s so Prometheus gets a final scrape.
```

## Why an initContainer

The replay client reads `data/features/replay.parquet` — that's DVC-tracked,
not baked into the image. The `fetch-features` initContainer downloads it
from MinIO using `src.common.fetch_features`, keyed by `GIT_COMMIT_OVERRIDE`
so the replay always reads the features the currently-Production model
trained on (not the replay-client commit). Update `GIT_COMMIT_OVERRIDE` in
both `job-demo.yaml` and `cronjob.yaml` after a fresh promotion.

## Scaling the serving pod

At 500 rps a single uvicorn worker is the bottleneck on a laptop minikube.
If `solar_predict_latency_seconds_bucket` shows sustained p95 > 50 ms or
`solar_replay_request_failures_total{reason="timeout"}` increments, bump
`k8s/serving/deployment.yaml` to `replicas: 2` and re-apply.
