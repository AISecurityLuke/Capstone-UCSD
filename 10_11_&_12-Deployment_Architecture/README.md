# Steps 10–12 — Deployment Architecture, Implementation, and Share/Presentation

Reference (architecture doc): `https://docs.google.com/document/d/1rChOwLfyqGnaobRxRNCCuE68UuOTyOgJ/edit?tab=t.0`

This folder contains the production-ready deployment for the Chatbot Filtration Service and serves as the primary README for the whole `Capstone-UCSD` project’s Steps 10, 11, and 12.

## What this project provides
- End-to-end FastAPI service with a policy gateway that classifies prompts and blocks according to configurable policy.
- Model-backed classification with a deterministic stub fallback when no model is configured.
- Human-in-the-loop curation workflow with both a compact in-app panel (`/ui`) and a table-based reviewer UI (`/reviewer`).
- EDA/Cleaning notebook that produces balanced, sanitized replacement datasets and visuals.
- Downstream mock to simulate integration with a RAG/LLM system.
- Smoke/load tests and Docker support for local deployment.

## TL;DR Demo Flow (Steps 10–12)
1) Start the API
   - cd `Capstone-UCSD/10_11_&_12-Deployment_Architecture`
   - python 3.9+ recommended
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt httpx`
   - Optional model: `export HF_MODEL_ID=xlm-roberta-base` (stub used if unset)
   - `./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`

2) Try the web UIs
   - Config + test UI: `http://127.0.0.1:8000/ui`
   - Reviewer UI (table view + labeling): `http://127.0.0.1:8000/reviewer`

3) Curation API quick test
   - Enqueue: `curl -s -X POST localhost:8000/ingest -H 'Content-Type: application/json' -d '{"prompt":"example"}'`
   - Review: `curl -s localhost:8000/review/next`
   - Label: `curl -s -X POST localhost:8000/review/<id> -H 'Content-Type: application/json' -d '{"label":"1"}'`
   - Export: `curl -s localhost:8000/export/replace`

4) Optional downstream mock
   - Terminal 2: `uvicorn app.downstream_mock:app --host 127.0.0.1 --port 8001`
   - Then exercise `POST /gateway` from the UI or via curl

## Architecture (Step 10)
### Components
- `app/`: FastAPI service exposing health, UI, gateway, classify, and curation APIs
- `service/`: model abstraction and decision logic (policy + thresholds)
- `curation/`: tools and state for analyst-in-the-loop balanced updates (100/100/100)
- `scripts/`: deployment utilities
- `tests/`: smoke and load test scaffolding

### Diagram
```mermaid
graph TD
    Client --> API[FastAPI]
    API --> ModelSvc[Model Service]
    ModelSvc --> Decision{Thresholds}
    Decision -- pass --> Downstream[Downstream RAG/LLM]
    Decision -- reject --> Logger[Structured Logs]
    API --> Metrics[Monitoring]
    ModelSvc --> Artifacts[(Model Store)]
```

## Deployment & Configuration
### Quickstart
1) Create venv and install:
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt httpx
```

2) Start API (optional env):
```
# Optional: export HF_MODEL_ID=distilroberta-base
# Optional: export BLOCK_LABELS=red,yellow  # default: red
# Optional: export SUSPICIOUS_PASS_THRESHOLD=0.85
# Optional: export DOWNSTREAM_URL=http://127.0.0.1:8001/respond
# Optional: export INGEST_UI=true
python scripts/run_local.sh
```

3) Start downstream mock (new terminal):
```
uvicorn app.downstream_mock:app --host 0.0.0.0 --port 8001
```

4) Test:
```
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/classify_prompt -H 'Content-Type: application/json' -d '{"prompt":"Hello"}'
curl -s -X POST localhost:8000/gateway -H 'Content-Type: application/json' -d '{"prompt":"Hello"}'
```

### Environment variables
- `HF_MODEL_ID` or `MODEL_PATH` — model loader (HuggingFace id or local path)
- `BLOCK_LABELS` — comma-separated labels to block; default `red`
- `SUSPICIOUS_PASS_THRESHOLD` — float [0,1]; if `label==yellow` and score ≥ threshold → reject
- `DOWNSTREAM_URL` — where `/gateway` forwards on pass; default `http://127.0.0.1:8001/respond`
- `INGEST_UI` — `true` to auto-ingest UI/gateway prompts

Note: Model service will use deterministic stub unless a valid `HF_MODEL_ID` or `MODEL_PATH` is provided.

### Docker (optional)
- From this folder: `docker-compose up --build`

## Implementation (Step 11)
### Key API Endpoints
- `GET /healthz` — health
- `POST /classify_prompt` — classify only (no forward)
- `POST /gateway` — enforce policy and optionally forward to downstream
- `GET|POST /config` — runtime policy and threshold
- `GET /ui` — config + testing UI
- `GET /reviewer` — manual review UI (table + labeling)
- Curation:
  - `POST /ingest` — enqueue a prompt
  - `GET /review/next` — next pending item
  - `POST /review/{id}` — classify a pending record
  - `GET /export/replace` — export cleaned `replace.json`

### Curation Workflow
- Ingest prompts via `/ingest` or toggle `INGEST_UI=true` to auto-enqueue UI traffic
- Label through `/ui` (classification panel) or `/reviewer` (table view)
- Export replacement slice with `/export/replace` when ready

### Tests
- Smoke: `tests/smoke.sh`
- Load: `tests/load_test.py`

## Share/Presentation (Step 12)
Use the notebook and UIs to generate artifacts for slides/demo.

### EDA & Cleaning Notebook
Open: `curation/EDA_Curation_Workflow.ipynb`

What it does:
- Loads `curation/classified.jsonl` and `curation/pending.jsonl`
- Applies the same sanitizer used in training (`7-Experiment_With_Models/utils/sanitizer.py`)
- Shows class balance/length histograms
- De-duplicates and trims extreme lengths
- Optionally balances a 100/100/100 slice for replacement
- Exports `curation/replace.cleaned.json` in training-compatible format
- Saves figures to `curation/visuals/`:
  - `curation_overview.png` (label distribution, char lengths, word counts)
  - `wordcount_by_class.png` (overlaid per-class word counts)

Notebook quickstart:
1) Do labeling in `/ui` or `/reviewer`
2) Open the notebook and run all cells
3) Use `replace.cleaned.json` to feed retraining

### Screenshots
- UI reject example (malicious prompt classified as red):

  ![UI reject example](curation/visuals/ui_reject_example.jpeg)

  Save your screenshot at the above path to display it here.

- Reviewer ready-to-classify view:

  ![Reviewer ready example](curation/visuals/reviewer_ready_example.jpeg)

  Save your screenshot at the above path to display it here.

## See also
- `QUICKSTART_AND_USAGE.md` for a full, copy/paste run guide
- Top-level README: `Capstone-UCSD/README.md` (demo flow and visuals)

## External Docs
- Slides/Supporting doc: https://docs.google.com/document/d/1XR2jrcj17HJbwI5lyBWgFgL517oNfUaZQqr605-QXXA/edit?tab=t.0
- Step 10 – Deployment Architecture: https://docs.google.com/document/d/199UTvpOHjqiwnlPuibljDTAG5V9u6j6OfCBQuk_0fDw/edit?tab=t.0
- Step 12 – Share/Presentation: https://docs.google.com/document/d/199UTvpOHjqiwnlPuibljDTAG5V9u6j6OfCBQuk_0fDw/edit?tab=t.0

## Rubric mapping
- Step 9 — Deployment Method & Engineering Plan
  - See: `Capstone-UCSD/9-Deployment_Method_&_Engineering_Plan/Deployment_Method_and_Engineering_Plan.md`
- Step 10 — Deployment Architecture (this README)
- Step 11 — Deployment Implementation (endpoints, policy config, UIs, curation API, tests)
- Step 12 — Share/Presentation (demo flow, visuals, and slides)
