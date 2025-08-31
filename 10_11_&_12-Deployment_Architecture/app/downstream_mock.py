from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Downstream Mock", version="0.1.0")

class DownstreamRequest(BaseModel):
	prompt: str = Field(..., min_length=1, max_length=8000)

@app.post("/respond")
async def respond(req: DownstreamRequest):
	# Simple echo with safe banner to simulate a RAG/LLM
	return {"reply": f"[SAFE RESPONSE] {req.prompt[:200]}"}
