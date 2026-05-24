# k8s/serving

Deployment, Service, and ServiceMonitor for the FastAPI predictor (T10 / Stage 5).
HPA is intentionally omitted — `params.yaml:serving.replicas = 1`, and a single
replica is the right size for the demo (T10 done-when only requires
`/predict` returning valid forecasts on the cluster).

## Manifests

| File                  | Purpose                                                                        |
|-----------------------|--------------------------------------------------------------------------------|
| `secret.yaml`         | Mirrors `solar-mlops-creds` from `kubeflow` ns into `solar` ns.                |
| `deployment.yaml`     | 1 replica, in-cluster image, readiness/liveness on `/healthz`.                 |
| `service.yaml`        | ClusterIP on port 80 -> container 8000.                                        |
| `servicemonitor.yaml` | Tells kube-prometheus-stack to scrape `/metrics` every 15s.                    |

## Deploy

```bash
# Build + load (commit hash drives the tag)
docker build -f docker/serve.Dockerfile -t solar-serve:$(git rev-parse --short HEAD) .
minikube image load solar-serve:$(git rev-parse --short HEAD)
# Update deployment.yaml's image tag if the SHA changed, then:
kubectl apply -f k8s/serving/
kubectl rollout status -n solar deploy/solar-serve --timeout=180s
```

## Verify (T10 done-when)

```bash
# 1) /predict returns 6 numbers.
kubectl port-forward -n solar svc/solar-serve 8000:80 &
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @../../tests/fixtures/predict_request.json | jq

# 2) /metrics exposes expected series.
curl -s http://localhost:8000/metrics | grep -E "^solar_(predict|prediction|input_feature|model_info)"

# 3) Prometheus shows the target Up.
kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus 9090:9090 &
# Open http://localhost:9090/targets and look for "serviceMonitor/solar/solar-serve/0".

# 4) Refuses to start when no Production model is registered.
# Archive the current Production version, then bounce the pod:
curl -s -X POST http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage \
  -H "Content-Type: application/json" \
  -d '{"name":"solar_forecaster","version":"<v>","stage":"Archived"}'
kubectl delete pod -n solar -l app=solar-serve
# The new pod's lifespan hook raises LoaderError -> uvicorn exits -> CrashLoopBackOff.
# Restore the version and the next pod comes back to Ready.
```

## When the Production model is an LSTM

`requirements-serve.txt` ships without PyTorch (CPU `torch` adds ~700 MB to
the image). The loader imports `torch` lazily inside `_LSTMHandle.from_dir`,
so importing the app works without it — but loading an LSTM Production model
raises `ImportError` at lifespan. If the next promotion winner is LSTM,
uncomment the torch line in `docker/requirements-serve.txt` and rebuild.
