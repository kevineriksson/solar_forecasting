# pipelines/kubeflow — T8 Kubeflow Pipelines DAG (Stages 1–4)

Wires the four Stage 1–4 trainers + a minimal promotion step into a single
KFP v2 pipeline that runs end-to-end on minikube.

```
                          ┌─► train_persistence ─┐
ingest ─► features ──────►├─► train_xgb         ─┼─► promotion (→ Staging)
                          └─► train_lstm        ─┘
```

## Design (per PROMPTS.md T8)

- **DVC stays outside KFP.** Features are materialised locally via
  `dvc repro features`, then pushed to MinIO under a flat per-sha prefix
  (`s3://solar-features/<git_sha>/`). The pipeline pods fetch from there at
  startup via `src.common.fetch_features` — no `dvc` CLI inside KFP.
- **One image, parameterised at submit time.** All six components run
  `solar-train:<short-sha>` (built by T7). The submitter compiles a fresh
  pipeline YAML with that image baked in; `imagePullPolicy=Never` keeps the
  pods on the locally-loaded image.
- **Provenance via env vars.** `GIT_COMMIT_OVERRIDE` and `DVC_HASH_OVERRIDE`
  are injected into every pod so `src.models.mlflow_utils` tags MLflow runs
  with the same values the submitter saw — without a working `.git` or
  `dvc.lock` in-pod.
- **Promotion is minimal (T8 scope).** `src.promotion.register_staging`
  picks the candidate with the highest `mean.skill` metric among the three
  training runs sharing the submitted `git_commit`, registers it as
  `solar_forecaster`, and transitions the new version to `Staging`. T9
  replaces this with full Production-vs-candidate logic on the promotion
  window.

## Files

| File              | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `components.py`   | Factory returning the 6 `@dsl.container_component` ops for an image. |
| `pipeline.py`     | `build_pipeline(image)` returning the `@dsl.pipeline` function.      |
| `submit.py`       | CLI that compiles + submits the pipeline to a KFP endpoint.          |

## Run it

End-to-end, from a clean checkout with the platform already up
(`bash scripts/00_cluster_up.sh && bash scripts/01_install_platform.sh`):

```bash
# 1. Local DVC artifacts must exist on disk.
dvc pull data/features data/interim/splits.json

# 2. Build + load the training image at the current sha.
SHA_SHORT=$(git rev-parse --short HEAD)
docker build -f docker/train.Dockerfile -t solar-train:$SHA_SHORT .
minikube image load solar-train:$SHA_SHORT  -p solar-mlops

# 3. Apply cluster prerequisites (Secret + params ConfigMap).
kubectl apply -f k8s/pipelines/secret.yaml
kubectl -n kubeflow create configmap solar-params \
  --from-file=params.yaml=./params.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Port-forward MinIO and publish the feature parquets for this sha.
kubectl port-forward -n minio svc/minio 9000:9000 &
bash scripts/02_publish_features.sh

# 5. Port-forward KFP and submit.
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
python -m pipelines.kubeflow.submit
```

The submitter prints a KFP UI link and blocks until the run finishes
(use `--no-wait` to detach).

## Verifying done-when

After a successful run:

1. **KFP UI** at <http://localhost:8080> shows the run as `SUCCEEDED` with
   six green tasks.
2. **MLflow UI** at <http://localhost:5000> shows three FINISHED runs in the
   `solar_forecaster` experiment all sharing the same `git_commit` tag —
   one each of `model_type ∈ {persistence, xgboost, lstm}`.
3. **MLflow Model Registry** lists `solar_forecaster` with the latest
   version transitioned to `Staging`. Its version tags include
   `model_type`, `source_run_id`, and `mean_skill`.

```bash
# Quick CLI check:
mlflow runs list --experiment-name solar_forecaster | head
mlflow models get-latest-versions -n solar_forecaster --stages Staging
```

## Re-submitting

`submit.py` always recompiles from the current code; the run shows up under
a new run name (`solar-mlops-<short-sha>`). To rerun an exact git_sha after
swapping branches, override:

```bash
python -m pipelines.kubeflow.submit \
  --git-sha <full-sha> --image-tag <short-sha>
```

## When the pipeline fails

- **`ImagePullBackOff`** on a pod — image wasn't loaded into the cluster.
  Re-run `minikube image load solar-train:<tag> -p solar-mlops`.
- **`fetch_features` exits 1 with `404 NoSuchKey`** — you skipped step 4
  above. Re-run `scripts/02_publish_features.sh`.
- **`persistence baseline not found`** in the XGB / LSTM pod — the
  persistence task either failed or its tags don't match. Confirm `train_xgb`
  / `train_lstm` ran *after* `train_persistence` (the DAG enforces this) and
  that the `git_commit` tag matches.
- **`promotion` fails with missing model types** — at least one of the three
  training pods didn't finish. Check the KFP UI per-task logs; promotion is
  hard-gated on having all three.
