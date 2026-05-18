#!/usr/bin/env bash
set -euo pipefail

# Linux/macOS smoke test for the portfolio repository.
#
# The Windows workflow uses smoke-test.ps1. This script gives reviewers on
# Ubuntu, WSL, or macOS the same one-command confidence check without needing
# to translate PowerShell commands. It intentionally exercises the lightweight
# paths only: no hardware, no external datasets, and no ONNX/Torch download.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

run_project() {
  local project="$1"
  local install_spec="$2"
  shift 2

  echo
  echo "== ${project}: install, tests, demo =="
  pushd "${ROOT}/${project}" >/dev/null
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install -e "${install_spec}" pytest
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONIOENCODING=utf-8 "${PYTHON}" -m pytest -q
  "$@"
  popd >/dev/null
}

run_project "edge-audio-anomaly-service" ".[api]" \
  "${PYTHON}" scripts/run_audio_demo.py

pushd "${ROOT}/edge-audio-anomaly-service" >/dev/null
"${PYTHON}" scripts/evaluate_public_audio_dataset.py
"${PYTHON}" scripts/benchmark_model_backends.py
popd >/dev/null

run_project "edge-ai-maintenance-gateway" ".[api]" \
  "${PYTHON}" scripts/run_gateway_demo.py

run_project "edge-vision-inspection" ".[api]" \
  "${PYTHON}" scripts/run_vision_demo.py

run_project "edgebench" "." \
  "${PYTHON}" -m edgebench run --builtin centroid --input-size 8 --warmup 2 --repeat 5

echo
echo "Smoke test completed."
