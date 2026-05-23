# k8s/pipelines — T8 KFP runtime config

Manifests the Kubeflow pipeline pods need at runtime:

| File          | What it provides                                                    |
|---------------|---------------------------------------------------------------------|
| `secret.yaml` | `solar-mlops-creds` — MinIO + MLflow endpoints and credentials.     |

The `params.yaml` consumed by the trainers is baked into the `solar-train`
image at `/app/params.yaml` (see `docker/train.Dockerfile`). The image is
rebuilt at each git_sha, so the baked file always matches the running
pipeline — no cluster-side override needed.

## Apply

```bash
kubectl apply -f k8s/pipelines/secret.yaml
```

Namespace-scoped to `kubeflow` (where KFP launches run pods).
