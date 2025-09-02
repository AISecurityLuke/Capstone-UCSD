# Quickstart & Usage

This is a quick run guide for the Prompt Filtration API. It includes local (venv) and Docker, API usage, UIs, config, curation, and stop commands.

## Local (venv)
1) Setup
```
cd 10_11_&_12-Deployment_Architecture
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt httpx
```

2) Optional environment
```
# Use HF model (or omit for stub)
export HF_MODEL_ID=xlm-roberta-base

# Policy/env knobs (optional)
export BLOCK_LABELS=red
export SUSPICIOUS_PASS_THRESHOLD=0.4
export DOWNSTREAM_URL=http://127.0.0.1:8001/respond
export INGEST_UI=true
```

3) Start API
```
./scripts/run_local.sh
# or
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

4) Downstream mock (optional – new terminal)
```
./.venv/bin/uvicorn app.downstream_mock:app --host 127.0.0.1 --port 8001
```

5) Open UIs
- Demo/config UI: http://127.0.0.1:8000/ui
- Reviewer UI: http://127.0.0.1:8000/reviewer

6) Smoke test (terminal)
```
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/classify_prompt -H 'Content-Type: application/json' -d '{"prompt":"Hello"}'
curl -s -X POST localhost:8000/gateway -H 'Content-Type: application/json' -d '{"prompt":"Hello"}'
```

7) Runtime config
```
# View
curl -s localhost:8000/config

# Update labels/threshold
curl -s -X POST localhost:8000/config \
  -H 'Content-Type: application/json' \
  -d '{"blocking_labels":["red"],"suspicious_pass_threshold":0.4}'
```

8) Curation API
```
# Ingest
curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' -d '{"prompt":"example"}'
# Next
curl -s localhost:8000/review/next
# Label
curl -s -X POST localhost:8000/review/<id> -H 'Content-Type: application/json' -d '{"label":"1"}'
# Export
curl -s localhost:8000/export/replace
```

9) EDA & cleaning
- Open `curation/EDA_Curation_Workflow.ipynb` in Jupyter and Run All.
- Outputs: `curation/visuals/*.png` and `curation/replace.cleaned.json`.

10) Stop the server(s)
```
lsof -ti tcp:8000,8001 | xargs -r kill -9
curl -sSf http://127.0.0.1:8000/healthz || echo "server down"
```

## Docker(Compose)
From `10_11_&_12-Deployment_Architecture`:
```
docker-compose up --build
```
Then visit http://127.0.0.1:8000/ui and use the same curl tests.

## Endpoints (reference)
- GET /healthz
- GET /metrics
- GET /config
- POST /config {blocking_labels?: string[], suspicious_pass_threshold?: number}
- POST /classify_prompt {prompt: string}
- POST /gateway {prompt: string}
- POST /ingest {prompt: string}
- GET /review/next
- POST /review/{id} {label: "0|1|2"}
- GET /export/replace

## Notes
- Default policy blocks `red`. Suspicious passes unless score ≥ threshold.
- If `HF_MODEL_ID` not set, a deterministic stub is used.
