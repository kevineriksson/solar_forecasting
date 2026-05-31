# Makefile — solar forecasting MLOps pipeline (T14)
#
# Top-level entry points:
#
#   make e2e         End-to-end demo from current cluster state (does NOT delete
#                    minikube; use `make teardown` first for a strict cold start).
#   make teardown    Wipe minikube.
#   make report      Regenerate the MLflow-driven results section of docs/report.md.
#
# Run `make help` for the full target list.
#
# Conventions:
#   - All image tags derive from $(SHA) = short git sha of HEAD. The retrain
#     image additionally bakes in $(GIT_SHA) (full) + $(DVC_HASH) at build time
#     so its receiver submits at the same provenance it was built for.
#   - k8s manifests with hard-pinned image tags are rendered through `sed` into
#     a build/k8s/ tree before `kubectl apply`. Source manifests are not mutated.
#   - Port-forwards are opened on a per-target basis and torn down at end. Run
#     `pkill -f 'kubectl port-forward'` if you ever end up with zombie tunnels.

SHELL          := /usr/bin/env bash
.SHELLFLAGS    := -eu -o pipefail -c
.DEFAULT_GOAL  := help

# --- versions / addresses -------------------------------------------------
PROFILE        ?= solar-mlops
GIT_SHA        := $(shell git rev-parse HEAD)
SHA            := $(shell git rev-parse --short HEAD)
PARAMS         ?= params.yaml

# Image names — tags applied at build time via $(SHA).
TRAIN_IMAGE    := solar-train
SERVE_IMAGE    := solar-serve
REPLAY_IMAGE   := solar-replay
RETRAIN_IMAGE  := solar-retrain

# In-cluster service endpoints. The Makefile port-forwards these on demand.
MLFLOW_PORT    ?= 5001
KFP_PORT       ?= 8080
MINIO_PORT     ?= 9000
SERVE_PORT     ?= 8000
RETRAIN_PORT   ?= 8001
PROM_PORT      ?= 9090
GRAFANA_PORT   ?= 3000

# Where rendered manifests land.
BUILD_DIR      := build
RENDERED_K8S   := $(BUILD_DIR)/k8s

# --- helpers --------------------------------------------------------------

# Resolve the DVC features hash for use as a build-time argument and for
# annotating MLflow runs. Mirrors src.models.mlflow_utils.get_dvc_features_hash.
DVC_HASH = $(shell python3 -c "from pathlib import Path; \
from src.models.mlflow_utils import get_dvc_features_hash; \
print(get_dvc_features_hash(Path('.')))")

define pf_start
	@if ! lsof -nP -iTCP:$(1) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "  port-forward -> $(1)"; \
	  kubectl port-forward -n $(2) svc/$(3) $(1):$(4) >/dev/null 2>&1 & \
	  echo $$! > $(BUILD_DIR)/pf-$(1).pid; \
	  for i in 1 2 3 4 5 6; do \
	    lsof -nP -iTCP:$(1) -sTCP:LISTEN >/dev/null 2>&1 && break; \
	    sleep 0.5; \
	  done; \
	fi
endef

define pf_stop
	@if [ -f $(BUILD_DIR)/pf-$(1).pid ]; then \
	  kill $$(cat $(BUILD_DIR)/pf-$(1).pid) 2>/dev/null || true; \
	  rm -f $(BUILD_DIR)/pf-$(1).pid; \
	fi
endef

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR) $(RENDERED_K8S)

# --- help -----------------------------------------------------------------

help:  ## Show this help.
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | sort

# --- platform setup -------------------------------------------------------

cluster-up:  ## Bring minikube up (idempotent).
	bash scripts/00_cluster_up.sh

platform:  ## Install MinIO, MLflow, KFP, kube-prometheus-stack.
	bash scripts/01_install_platform.sh

teardown:  ## minikube delete + kill any leftover port-forwards.
	bash scripts/99_teardown.sh

# --- build ----------------------------------------------------------------

build: build-train build-serve build-replay build-retrain  ## Build + load all four images into minikube.

build-train:  ## Build solar-train:$(SHA), load into minikube.
	DOCKER_BUILDKIT=1 docker build -f docker/train.Dockerfile -t $(TRAIN_IMAGE):$(SHA) .
	minikube image load -p $(PROFILE) $(TRAIN_IMAGE):$(SHA)

build-serve:  ## Build solar-serve:$(SHA), load into minikube.
	DOCKER_BUILDKIT=1 docker build -f docker/serve.Dockerfile -t $(SERVE_IMAGE):$(SHA) .
	minikube image load -p $(PROFILE) $(SERVE_IMAGE):$(SHA)

build-replay:  ## Build solar-replay:$(SHA), load into minikube.
	DOCKER_BUILDKIT=1 docker build -f docker/replay.Dockerfile -t $(REPLAY_IMAGE):$(SHA) .
	minikube image load -p $(PROFILE) $(REPLAY_IMAGE):$(SHA)

build-retrain: $(BUILD_DIR)  ## Build solar-retrain:$(SHA) baking GIT_SHA + DVC_HASH, load into minikube.
	@echo "  GIT_SHA  = $(GIT_SHA)"
	@echo "  DVC_HASH = $(DVC_HASH)"
	DOCKER_BUILDKIT=1 docker build \
	  -f docker/retrain.Dockerfile \
	  --build-arg GIT_SHA=$(GIT_SHA) \
	  --build-arg DVC_HASH=$(DVC_HASH) \
	  -t $(RETRAIN_IMAGE):$(SHA) .
	minikube image load -p $(PROFILE) $(RETRAIN_IMAGE):$(SHA)

# --- features (Stage 1+2 outside KFP, by design) ----------------------

features: $(BUILD_DIR)  ## Run dvc repro then publish features to MinIO under $(GIT_SHA).
	dvc repro
	$(call pf_start,$(MINIO_PORT),minio,minio,9000)
	bash scripts/02_publish_features.sh $(GIT_SHA)
	$(call pf_stop,$(MINIO_PORT))

# --- pipeline (Stage 3+4) -------------------------------------------------

pipeline: $(BUILD_DIR)  ## Submit the KFP pipeline at HEAD and wait for completion.
	$(call pf_start,$(KFP_PORT),kubeflow,ml-pipeline-ui,80)
	$(call pf_start,$(MLFLOW_PORT),mlflow,mlflow,80)
	MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT) \
	python -m pipelines.kubeflow.submit \
	  --git-sha $(GIT_SHA) \
	  --image-tag $(SHA) \
	  --kfp-endpoint http://localhost:$(KFP_PORT)
	$(call pf_stop,$(KFP_PORT))
	$(call pf_stop,$(MLFLOW_PORT))

# --- deploy serving (Stage 5) ---------------------------------------------

serve: $(BUILD_DIR)  ## Render k8s/serving with $(SHA) and apply.
	@mkdir -p $(RENDERED_K8S)/serving
	@for f in k8s/serving/*.yaml; do \
	  sed "s|$(SERVE_IMAGE):[A-Za-z0-9_.-]*|$(SERVE_IMAGE):$(SHA)|g" "$$f" > $(RENDERED_K8S)/serving/$$(basename $$f); \
	done
	kubectl apply -f $(RENDERED_K8S)/serving/
	kubectl rollout status -n solar deploy/solar-serve --timeout=300s

# --- deploy monitoring (Stage 6) ------------------------------------------

monitoring:  ## Apply the PrometheusRule + Grafana dashboard configmap.
	kubectl apply -f k8s/monitoring/alerts.yaml
	bash scripts/build_dashboard_configmap.sh
	kubectl apply -f k8s/monitoring/dashboard-configmap.yaml

# --- deploy replay (Stage 6) ----------------------------------------------

replay: $(BUILD_DIR)  ## Apply k8s/replay with $(SHA); fire the one-shot demo Job.
	@mkdir -p $(RENDERED_K8S)/replay
	@for f in k8s/replay/*.yaml; do \
	  sed -E "s|$(REPLAY_IMAGE):[A-Za-z0-9_.-]+|$(REPLAY_IMAGE):$(SHA)|g; s|(GIT_COMMIT_OVERRIDE\"?[[:space:]]*\n?[[:space:]]*value: \")[a-f0-9]+|\1$(GIT_SHA)|" "$$f" > $(RENDERED_K8S)/replay/$$(basename $$f); \
	done
	# Apply service + servicemonitor + cronjob (suspended) + demo Job.
	kubectl apply -f $(RENDERED_K8S)/replay/service.yaml
	kubectl apply -f $(RENDERED_K8S)/replay/servicemonitor.yaml
	kubectl apply -f $(RENDERED_K8S)/replay/cronjob.yaml
	# Delete any prior demo Job so this is a fresh run.
	kubectl delete -f $(RENDERED_K8S)/replay/job-demo.yaml --ignore-not-found
	kubectl apply -f $(RENDERED_K8S)/replay/job-demo.yaml

# --- deploy retrain (T13) -------------------------------------------------

retrain: $(BUILD_DIR)  ## Apply k8s/retrain with $(SHA); rotates WEBHOOK_TOKEN.
	@mkdir -p $(RENDERED_K8S)/retrain
	@for f in k8s/retrain/*.yaml; do \
	  sed "s|$(RETRAIN_IMAGE):[A-Za-z0-9_.-]*|$(RETRAIN_IMAGE):$(SHA)|g" "$$f" > $(RENDERED_K8S)/retrain/$$(basename $$f); \
	done
	# Rotate the shared bearer secret. Idempotent: replaces in place.
	@TOKEN=$$(python3 -c "import secrets; print(secrets.token_hex(32))"); \
	  kubectl create secret generic solar-retrain-webhook -n monitoring \
	    --from-literal=WEBHOOK_TOKEN=$$TOKEN --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f $(RENDERED_K8S)/retrain/service.yaml
	kubectl apply -f $(RENDERED_K8S)/retrain/servicemonitor.yaml
	kubectl apply -f $(RENDERED_K8S)/retrain/alertmanager-config.yaml
	kubectl apply -f $(RENDERED_K8S)/retrain/deployment.yaml
	kubectl rollout status -n monitoring deploy/solar-retrain --timeout=180s

# --- e2e demo with deliberate drift trigger -------------------------------

trigger-drift:  ## Lower PSI threshold to force a SolarDriftHigh firing, wait, restore.
	@echo "==> Patching SolarDriftHigh to fire on PSI > 0.05 (was 3.0)"
	@kubectl get prometheusrule -n monitoring solar-forecasting-alerts -o yaml \
	  | sed 's|max(solar_replay_feature_psi) > 3.0|max(solar_replay_feature_psi) > 0.05|; s|for: 10m|for: 1m|' \
	  | kubectl apply -f -
	@echo "==> Waiting up to 4 min for retrain to be triggered…"
	@for i in $$(seq 1 24); do \
	  sleep 10; \
	  c=$$(kubectl exec -n monitoring deploy/solar-retrain -- \
	         python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())" 2>/dev/null \
	       | awk '/^solar_retrain_runs_submitted_total\{/ {sub(/^[^ ]+ +/, \"\", $$0); print int($$1); exit}'); \
	  c=$${c:-0}; \
	  echo "  t+$$((i*10))s  runs_submitted=$$c"; \
	  if [ "$$c" -gt 0 ]; then echo "  drift triggered retrain — break"; break; fi; \
	done
	@echo "==> Restoring PrometheusRule"
	kubectl apply -f k8s/monitoring/alerts.yaml

# --- top-level orchestration ---------------------------------------------

verify:  ## Smoke-test serve + replay + retrain end-to-end.
	bash scripts/verify_e2e.sh

# Note: e2e assumes cluster-up + platform have already run. Use `make teardown
# cluster-up platform e2e` if you really want a cold start (~45-60 min).
# Replay comes AFTER pipeline so its predictions hit the just-promoted model;
# retrain comes AFTER replay so the trigger has metrics to fire on.
e2e: build features pipeline serve monitoring retrain replay trigger-drift verify  ## Full demo: build -> pipeline -> deploy -> trigger drift -> verify.

# --- report ---------------------------------------------------------------

report: $(BUILD_DIR)  ## Regenerate docs/report.md results section from live MLflow.
	$(call pf_start,$(MLFLOW_PORT),mlflow,mlflow,80)
	MLFLOW_TRACKING_URI=http://localhost:$(MLFLOW_PORT) \
	  python -m scripts.build_report -o $(BUILD_DIR)/results.md
	$(call pf_stop,$(MLFLOW_PORT))
	@echo "Generated $(BUILD_DIR)/results.md — paste under '## Results' in docs/report.md."

# --- ports / dev convenience ---------------------------------------------

ports:  ## Start all the UI port-forwards (MLflow, KFP, MinIO, Grafana). Foreground; ctrl-C to stop.
	@echo "Forwarding: MLflow=$(MLFLOW_PORT)  KFP=$(KFP_PORT)  MinIO=$(MINIO_PORT)  Grafana=$(GRAFANA_PORT)  Prom=$(PROM_PORT)"
	@trap "kill 0" EXIT; \
	 kubectl port-forward -n mlflow svc/mlflow $(MLFLOW_PORT):80 & \
	 kubectl port-forward -n kubeflow svc/ml-pipeline-ui $(KFP_PORT):80 & \
	 kubectl port-forward -n minio svc/minio-console 9001:9001 & \
	 kubectl port-forward -n monitoring svc/kps-grafana $(GRAFANA_PORT):80 & \
	 kubectl port-forward -n monitoring svc/kps-kube-prometheus-stack-prometheus $(PROM_PORT):9090 & \
	 wait

clean:  ## Drop the build/ scratch tree.
	rm -rf $(BUILD_DIR)

.PHONY: help cluster-up platform teardown \
        build build-train build-serve build-replay build-retrain \
        features pipeline serve monitoring replay retrain \
        trigger-drift verify e2e report ports clean
