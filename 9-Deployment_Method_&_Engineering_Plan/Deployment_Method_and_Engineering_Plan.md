# Step 9 — Deployment Method & Engineering Plan

Author: Luke Johnson (AISecurityLuke)

Last updated: 2025-08-30

Reference: Capstone Step 9 rubric — see `https://docs.google.com/document/d/1Od13NaksdmMh7OIcpePPmPTou2Qbptz8/edit?tab=t.0`

---

## 1) Objectives
- Document and justify the chosen deployment method for the chatbot filtration system
- Provide an end-to-end engineering plan from package → serve → monitor → iterate
- Define an integration surface (API spec) and operational runbooks

## 2) Summary of Options Considered
- FastAPI REST service + Docker + Cloud VM/Container Service
- Serverless function (e.g., Cloud Run/Lambda) with model artifact
- On-prem/edge binary with model bundle

Rationale for preferred approach will consider: latency, cost, complexity, security, and team ops maturity.

## 3) Chosen Deployment Method (Draft)
- Method: FastAPI REST service
- Packaging: Docker image
- Hosting target: Cloud VM / Managed container (TBD)
- Model artifact: Best-performing model from `7-Experiment_With_Models/results/`
- Inference device: CPU by default; optional GPU if needed

## 4) High-Level Architecture
- Client → Filtration API (FastAPI) → Classifier → Decision (green/yellow/red)
- If PASS: forward to downstream RAG/service (out-of-scope integration stub)
- If REJECT: log event and return safe denial message

## 5) API Specification (Initial)
- POST `/classify_prompt`
  - Request: `{ "prompt": string }`
  - Response: `{ "label": "green|yellow|red", "scores": {"safe": float, "suspicious": float, "malicious": float}, "decision": "pass|reject" }`
- GET `/healthz`
  - 200 OK when model loaded and ready

## 6) Model Packaging & Loading
- Load selected model from `Capstone-UCSD/7-Experiment_With_Models/models/<best_model_dir>`
- Normalize preprocessing to match training (tokenization/vectorization config)
- Provide a single `predict(text: str) -> Dict` function used by the API layer

## 7) Operational Concerns
- Logging: structured JSON logs; request id; latency; decision; class scores
- Monitoring: basic metrics (QPS, p95 latency, error rate); model drift hooks (future)
- Config: environment variables for thresholds and model path
- Secrets: no secrets in repo; use env or secret manager
- Security: input validation, request size limit, rate limiting at ingress

## 8) CI/CD & Release Strategy
- Build: Dockerfile builds API with pinned requirements
- Test: unit tests for `predict` contract and API routes
- Release: tag image, push to registry, deploy via script or GitHub Actions
- Rollback: keep N-1 image; blue/green or canary when feasible

## 9) Runbooks
- Local run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Health check: `GET /healthz`
- Log inspection: `docker logs <container>`
- Threshold tuning: adjust env `SUSPICIOUS_PASS_THRESHOLD`

## 10) Milestones & Timeline (Draft)
- M1: Select best model and freeze artifact
- M2: Implement FastAPI wrapper and local inference
- M3: Containerize with Docker; add health/readiness probes
- M4: Basic load test; validate latency and correctness
- M5: Deploy to chosen environment; add monitoring
- M6: Documentation handoff and demo

## 11) Next Actions Checklist
- [ ] Identify top model from `results/results.csv` and freeze artifact
- [ ] Create API scaffold (`FastAPI`) and wire `predict`
- [ ] Add configurable thresholds for yellow/red decisions
- [ ] Write Dockerfile and compose file for local testing
- [ ] Draft deployment script and minimal CI
- [ ] Document runbook and SLOs

## 12) Deployment Options Trade-offs and Justification
- FastAPI + Docker (Chosen):
  - Pros: Lowest latency, full control, simple ops; easy model/runtime pinning; smooth path to GPU.
  - Cons: You manage infra and scale.
- Serverless (Cloud Run/Lambda):
  - Pros: Zero-manage infra, scale-to-zero, quick to ship.
  - Cons: Cold starts; limited memory/time; packaging size limits; less control over performance.
- On-Prem/Edge:
  - Pros: Data sovereignty, strict security, offline.
  - Cons: Highest ops overhead; hardware constraints.

Decision: FastAPI + Docker on a managed container/VM for predictable latency and straightforward integration with existing code.

## 13) Architecture Diagram
```mermaid
graph TD
    A[Client / Upstream App] --> B[Filtration API (FastAPI)]
    B --> C[Classifier Service]
    C --> D{Decision}
    D -- pass --> E[Downstream RAG / LLM]
    D -- reject --> F[Structured Logger]
    B --> G[Metrics/Monitoring]
    C --> H[(Model Artifact Store)]
```

## 14) Environment & Infrastructure Plan
- Runtime: Dockerized FastAPI, Python 3.10–3.11, uvicorn workers=2–4 (CPU-bound models higher workers; transformer GPU workers=1).
- Instance: Start CPU `c5.xlarge`/`n2-standard-4`; optional GPU `g4dn.xlarge` if transformer latency needs.
- Storage: Model artifacts stored in `Capstone-UCSD/7-Experiment_With_Models/models/` and pushed to object store (S3/GCS) with versioning.
- Networking: Managed ingress (ALB/Cloud HTTP LB), TLS (Let’s Encrypt/ACM), private subnet for service.
- Config: Env vars for thresholds, model path, logging level.

## 15) Cost Estimate (Rough Order)
- CPU VM (4 vCPU): $35–$60/mo; GPU `g4dn.xlarge`: $120–$300/mo if needed.
- Object storage + egress: $1–$10/mo at current scale.
- Monitoring/logs: $0–$20/mo depending on chosen stack.

## 16) Security & Compliance
- TLS everywhere; HSTS.
- AuthN: API key or mTLS at ingress; optional JWT for user-level attribution.
- Rate limiting: e.g., 10 RPS per API key, burst 50.
- Input validation: max payload size (e.g., 8 KB), UTF-8 only.
- PII hygiene: do not log raw prompts unless analyst mode is enabled; redact emails/phones.
- Secrets via env/secret manager; no secrets in repo.
- Logs retention: 30 days by default; configurable.

## 17) SLOs & Success Metrics
- Availability: 99.5% monthly.
- Latency (CPU classical/DL): p50 ≤ 100 ms, p95 ≤ 250 ms.
- Latency (transformer small, CPU/GPU): p50 ≤ 300 ms, p95 ≤ 800 ms.
- Quality: malicious recall ≥ 0.90; suspicious precision ≥ 0.80; benign F1 ≥ 0.90.
- Error budget and paging on sustained p95 breach.

## 18) Testing & Validation Plan
- Unit tests: `predict` contract, thresholding logic, API schema.
- Regression: lock model version; golden prompts with expected labels/scores.
- Load: step to 50 RPS (CPU) or 20 RPS (GPU transformer) while meeting p95; record saturation point.
- Chaos: dependency failure simulation (object store unreachable), ensure startup degrades gracefully.

## 19) Risks & Mitigations
- Model drift → monitor class priors and performance on analyst labels; scheduled evaluation; retraining policy below.
- Memory/OOM → enforce max sequence length; batch size=1 at inference; container limits.
- Tokenizer/encoding errors → strict UTF-8 validation; fallback sanitization.
- Dependency CVEs → weekly dependabot/`pip-audit`; pin versions.
- Infra misconfig → IaC templates, least-privilege IAM roles.

## 20) Versioning & Artifact Management
- Model versioning: `model_name-YYYYMMDD-hhmm` with SHA256 of weights and config.
- Store: object storage path `models/<model_name>/<version>/` with `MODEL_INFO.json` containing checksum, training data snapshot id, metrics, and thresholds.
- Rollback: maintain N-1 version; `MODEL_CURRENT` pointer for active; flip pointer to rollback.

## 21) Deployment Steps (Command-Level)
- Build:
  - `docker build -t prompt-filter:$(git rev-parse --short HEAD) .`
- Local run:
  - `docker run -p 8000:8000 -e MODEL_PATH=/app/models/<best> -e SUSPICIOUS_PASS_THRESHOLD=0.4 prompt-filter:$(git rev-parse --short HEAD)`
- Uvicorn (dev):
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`
- Deploy (example):
  - Push image to registry; update service with new tag; wait for health `GET /healthz` green; cut traffic.
- Rollback:
  - Redeploy previous tag or flip `MODEL_CURRENT` pointer in config and restart.

## 22) Rubric Mapping
- Deployment method chosen and justified → Sections 12, 14–16.
- Architecture + API spec → Sections 13, 5.
- Ops plan (monitoring, security, runbooks) → Sections 7, 16, 9.
- CI/CD and versioning → Sections 8, 20–21.
- Success metrics → Section 17.
- Cost and trade-offs → Sections 12, 15.

## 23) Human-in-the-Loop Data Curation & Continuous Training Policy
Goal: incorporate analyst feedback while enforcing strict bounds to preserve dataset stability and class balance.

Rules (as specified):
- Keep total dataset size ~30,000 at all times.
- Never store more than 10,050 replacements per type/class at a time.
- Update in even class batches: exactly 100 items per class label (0, 1, 2) per batch (total 300), to keep the distribution even.
- Maintain markers for where the last retrain occurred and where to replace next.

Operational design:
- State file: `Capstone-UCSD/5-Data_Wrangling/curation_state.json`
  - Example schema:
    - `{"classes": {"0": {"last_retrain_idx": 0, "next_start_idx": 0, "replacements_in_window": 0}, "1": {...}, "2": {...}}, "window_max": 10050, "window_start_timestamp": "ISO8601"}`
- Replacement window:
  - Sliding window per class capped at 10,050 replacements; once a replacement exits the window (by time or by being included in a retrain snapshot), it frees capacity.
- Batch process:
  - Accumulate analyst-approved items; when at least 100 items exist for each class, perform a balanced swap-in of 300:
    - Remove 100 oldest items per class from the current dataset segment determined by `next_start_idx`.
    - Insert 100 new items per class.
    - Advance `next_start_idx` by 100 per class (wrap-around when reaching class partition end).
- Retraining trigger:
  - Minimum threshold policy: when cumulative new items since last retrain ≥ K per class (e.g., K=1,000) and all classes meet K, freeze a snapshot and retrain.
  - After retrain:
    - Set `last_retrain_idx = next_start_idx` for each class.
    - Reset per-class `replacements_in_window` counters appropriately.
- Snapshotting:
  - Persist full dataset snapshot and `curation_state.json` with version id matching model training run.
- Audit & logging:
  - Log each replacement event with `{id_old, id_new, class, timestamp, operator}`.

API & tooling notes:
- Add an analyst endpoint or tool to submit reviewed items with class labels and metadata.
- Validation ensures class labels ∈ {0,1,2}; size limits; duplicates prevented by id.

This policy guarantees constant dataset size, per-class caps, balanced updates, and deterministic pointers for the next replacement batch and retrain points.
