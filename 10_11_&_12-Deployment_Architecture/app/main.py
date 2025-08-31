from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import time

from app.service.model_service import ModelService

app = FastAPI(title="Prompt Filtration API", version="0.1.0")


class ClassifyRequest(BaseModel):
	prompt: str = Field(..., min_length=1, max_length=8000)


class ClassifyResponse(BaseModel):
	label: str
	scores: dict
	decision: str


MODEL_PATH = os.getenv("MODEL_PATH", "")
SUSPICIOUS_PASS_THRESHOLD = float(os.getenv("SUSPICIOUS_PASS_THRESHOLD", "0.4"))

# Policy: labels blocked from passing to downstream (default: red only)
_BLOCK_LABELS = os.getenv("BLOCK_LABELS", "red")
BLOCK_LABELS = set([lbl.strip().lower() for lbl in _BLOCK_LABELS.split(",") if lbl.strip()])


model_service = ModelService(model_path=MODEL_PATH, suspicious_pass_threshold=SUSPICIOUS_PASS_THRESHOLD)


@app.get("/healthz")
async def healthz():
	return {"status": "ok", "model_loaded": model_service.is_ready()}

from app.metrics import metrics


def apply_policy(label: str, base_decision: str) -> str:
	# Enforce blocklist policy on labels (green/yellow/red)
	return "reject" if label.lower() in BLOCK_LABELS else base_decision


@app.post("/classify_prompt", response_model=ClassifyResponse)
async def classify_prompt(req: ClassifyRequest):
	t0 = time.time()
	try:
		label, scores, decision = model_service.classify(req.prompt)
		final_decision = apply_policy(label, decision)
		return {"label": label, "scores": scores, "decision": final_decision}
	except Exception as e:
		metrics.record((time.time()-t0)*1000.0, error=True)
		raise HTTPException(status_code=500, detail=str(e))
	finally:
		metrics.record((time.time()-t0)*1000.0, error=False)


@app.get("/metrics")
async def get_metrics():
	return metrics.snapshot()


# Gateway: filter then forward to downstream if pass
from typing import Optional
import httpx

class GatewayRequest(ClassifyRequest):
	pass

@app.post("/gateway")
async def gateway(req: GatewayRequest):
	label, scores, decision = model_service.classify(req.prompt)
	final_decision = apply_policy(label, decision)
	if final_decision == "reject":
		return {"label": label, "scores": scores, "decision": final_decision, "downstream": None}
	# forward to downstream mock (or real URL via env)
	downstream_url = os.getenv("DOWNSTREAM_URL", "http://127.0.0.1:8001/respond")
	async with httpx.AsyncClient() as client:
		try:
			resp = await client.post(downstream_url, json={"prompt": req.prompt}, timeout=10.0)
			payload: Optional[dict] = resp.json() if resp.headers.get("content-type"," ").startswith("application/json") else {"text": resp.text}
			return {"label": label, "scores": scores, "decision": final_decision, "downstream": payload}
		except Exception as e:
			return {"label": label, "scores": scores, "decision": final_decision, "downstream_error": str(e)}
