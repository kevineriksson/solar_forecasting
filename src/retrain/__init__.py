"""Retraining trigger (T13).

Tiny FastAPI receiver that listens for Alertmanager webhooks for the
``SolarDriftHigh`` / ``SolarSkillScoreLow`` rules and submits a fresh
Kubeflow Pipelines run against the latest ``solar-train`` image loaded
into the cluster.
"""
