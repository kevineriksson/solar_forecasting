"""Smoke test: params.yaml parses and contains required top-level keys.

Runnable standalone (`python tests/test_params.py`) so we don't require pytest yet.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = REPO_ROOT / "params.yaml"

REQUIRED_TOP_LEVEL = [
    "site",
    "data",
    "forecast",
    "splits",
    "features",
    "training",
    "promotion",
    "serving",
    "replay",
    "monitoring",
    "paths",
]


def test_params_loads() -> None:
    with PARAMS_PATH.open() as f:
        params = yaml.safe_load(f)

    assert isinstance(params, dict), "params.yaml did not parse to a mapping"

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in params]
    assert not missing, f"params.yaml missing required top-level keys: {missing}"

    ref_now = params["splits"]["reference_now"]
    parsed = datetime.fromisoformat(ref_now.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, "reference_now must be timezone-aware"


if __name__ == "__main__":
    test_params_loads()
    print("OK: params.yaml parses and has all required keys.")
    sys.exit(0)
