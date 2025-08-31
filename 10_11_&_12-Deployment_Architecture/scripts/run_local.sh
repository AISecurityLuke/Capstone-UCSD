#!/usr/bin/env bash
set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-""}
export SUSPICIOUS_PASS_THRESHOLD=${SUSPICIOUS_PASS_THRESHOLD:-0.4}
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
