# k8s/pipelines — T8 KFP runtime config

Manifests the Kubeflow pipeline pods need at runtime:

| File          | What it provides                                                    |
|---------------|---------------------------------------------------------------------|
| `secret.yaml` | `solar-mlops-creds` — MinIO + MLflow endpoints and credentials.     |

The `solar-params` ConfigMap (holding `params.yaml`) is generated at submit
time from the live file on disk:

```bash
kubectl -n kubeflow create configmap solar-params \
  --from-file=params.yaml=./params.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

This is intentionally not a checked-in manifest — `params.yaml` changes more
often than the image, and baking it into the cluster from the working tree
keeps the two in sync without an extra commit step.

## Apply order

```bash
kubectl apply -f k8s/pipelines/secret.yaml
kubectl -n kubeflow create configmap solar-params \
  --from-file=params.yaml=./params.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

Both are namespace-scoped to `kubeflow` (where KFP launches run pods).
