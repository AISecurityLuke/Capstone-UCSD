from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

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

model_service = ModelService(model_path=MODEL_PATH, suspicious_pass_threshold=SUSPICIOUS_PASS_THRESHOLD)


@app.get("/healthz")
async def healthz():
	return {"status": "ok", "model_loaded": model_service.is_ready()}


@app.post("/classify_prompt", response_model=ClassifyResponse)
async def classify_prompt(req: ClassifyRequest):
	label, scores, decision = model_service.classify(req.prompt)
	return {"label": label, "scores": scores, "decision": decision}
