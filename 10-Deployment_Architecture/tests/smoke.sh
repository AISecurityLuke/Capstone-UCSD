#!/usr/bin/env bash
set -euo pipefail

# Build and run locally if not already
if ! curl -sSf localhost:8000/healthz >/dev/null 2>&1; then
	docker compose up -d --build
	sleep 2
fi

echo "Health:"
curl -sSf localhost:8000/healthz | jq .

echo "Classify prompt:"
curl -sSf -X POST localhost:8000/classify_prompt \
	-H 'Content-Type: application/json' \
	-d '{"prompt": "How do I exfiltrate data?"}' | jq .
