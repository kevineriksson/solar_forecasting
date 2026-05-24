# Demo Script — Solar Forecasting MLOps Pipeline

A ~10-minute walkthrough for a live presentation. Assumes the cluster is already
warm (run `make e2e` *before* the demo, not during).

## 0. Pre-flight (run before the audience joins)

```bash
# Verify everything is green.
make verify
```

Then start the five port-forwards in their own terminal:

```bash
make ports     # MLflow, KFP, MinIO, Grafana, Prometheus
```

Open four browser tabs:

- MLflow:     http://localhost:5000
- KFP:        http://localhost:8080
- Grafana:    http://localhost:3000   (admin / admin)
- Prometheus: http://localhost:9090

---

## 1. The story (30 s)

> *"We're forecasting GHI/DNI/DHI 15 minutes and 1 hour ahead for a fixed site
> in Wolf Point, Montana. Solcast historicals go back to 2007. The whole pipeline
> runs on a laptop minikube — and when the model degrades in production, it
> retrains itself."*

---

## 2. MLflow — what got promoted (1 min)

**Open:** http://localhost:5000 → Models → `solar_forecaster`

Show:
- The `Production` version.
- The version tags: `git_commit`, `model_type`, `mean_skill`.

> *"Every promoted model is traceable to an exact Git commit. That's one of
> the project's core invariants — and it's what makes the rebuild-from-history demo work."*

**Click into the source run** → show:
- Tag `dvc_hash` (the feature artifact fingerprint).
- The per-fold metrics: `fold0.skill.ghi.1h`, etc.

---

## 3. KFP — the DAG that produced it (1 min)

**Open:** http://localhost:8080 → Experiments → `solar-mlops`

Show the most recent successful run. Click into it. Walk through the DAG:

```
ingest -> features -> { persistence, xgb, lstm } (parallel) -> promotion
```

> *"Three trainers compete; persistence is the baseline; the winner gets
> promoted if it beats Production by ≥ 0.02 in mean skill score."*

---

## 4. Serving — live `/predict` (1 min)

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp_utc": "2026-05-10T12:00:00Z",
    "features": {
      "k_t": 0.92, "ghi_clearsky": 850.0,
      "k_t_lag_1": 0.91, "k_t_lag_4": 0.88, "k_t_lag_12": 0.82,
      "k_t_roll_4": 0.90, "k_t_roll_12": 0.85,
      "sin_hour": 0.5, "cos_hour": 0.866,
      "sin_doy": 0.1, "cos_doy": 0.995,
      "is_night": 0.0
    }
  }' | jq
```

Six floats — `{ghi,dni,dhi}_{15min,1h}` — plus the serving model's version
and git commit.

---

## 5. Grafana — live dashboard (1 min)

**Open:** http://localhost:3000 → Dashboards → `Solar Forecasting — Overview`

Walk the panels:
- Predict latency p50/p95
- Rolling MAE / RMSE / skill per (target, horizon) — replay is live, so these
  populate in real time
- Feature drift (PSI) per tracked feature
- Prediction distribution histograms

> *"Everything we see is from the replay client walking the last 6 months of
> Solcast history through the serving pod at 500 requests/sec — about 2 min
> to replay the entire window."*

---

## 6. Trigger drift → retrain (3 min — the wow moment)

In a second terminal, lower the alert threshold so PSI > 0.05 trips it:

```bash
make trigger-drift
```

This:
1. Patches the PrometheusRule to `> 0.05` and `for: 1m`.
2. Polls the receiver's `runs_submitted_total` every 10 s.
3. Restores the rule when a submission is observed.

**Switch to the Alerts tab** in Grafana (or http://localhost:9090/alerts) and
narrate:
- `SolarDriftHigh` → Pending → Firing.
- Alertmanager picks it up, routes to the webhook (`http://solar-retrain.monitoring.svc.cluster.local:8000/alert`).
- Receiver POSTs to KFP.

**Switch to KFP UI** → a new run appears within ~2 min. Click into it; show the
DAG running. Promotion is the final step.

**Receiver logs** (third terminal):

```bash
kubectl logs -n monitoring deploy/solar-retrain -f
# Look for: submitted run_id=… alertname=SolarDriftHigh
```

---

## 7. Reproducibility demo (2 min)

> *"All right — if I told you this model was bad, could you rebuild the exact
> training run that produced it?"*

Pick the current Production model from MLflow and read its tags:

```bash
SHA=$(kubectl exec -n mlflow deploy/mlflow -- \
  python -c "import mlflow; \
    v=mlflow.tracking.MlflowClient().get_latest_versions('solar_forecaster', stages=['Production'])[0]; \
    print(v.tags['git_commit'])")
DVC_HASH=$(kubectl exec -n mlflow deploy/mlflow -- \
  python -c "import mlflow; \
    v=mlflow.tracking.MlflowClient().get_latest_versions('solar_forecaster', stages=['Production'])[0]; \
    print(v.tags.get('dvc_hash','?'))")
echo "Production = git $SHA / dvc $DVC_HASH"

git checkout $SHA
# All you need to rebuild is in this commit + the feature artifacts under
# s3://solar-features/$SHA/ in MinIO. Both are versioned.
```

Show that the Dockerfile, params.yaml, and feature schemas at that commit are
exactly what trained the live model. Don't rerun — the audience trusts the
demo more if you don't try to fit a 6-minute pipeline run into a 30-second
window.

---

## 8. Wrap (30 s)

> *"Three invariants made this work end-to-end: code in Git, data in DVC,
> models in MLflow tagged with both. Drift detection closes the loop:
> Prometheus to Alertmanager to webhook to KFP — fully automated retraining
> with a 30-minute debounce so a sustained alert doesn't spawn a storm of
> training runs."*

---

## Reset between demos

```bash
kubectl delete -f k8s/replay/job-demo.yaml --ignore-not-found
kubectl rollout restart -n monitoring deploy/solar-retrain   # clears in-memory cooldown
kubectl apply -f k8s/monitoring/alerts.yaml                  # ensures thresholds are baseline
```

If anything looks broken, run `make verify` — it pinpoints which stage is off.
