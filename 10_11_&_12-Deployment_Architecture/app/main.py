from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
	_maybe_auto_ingest(req.prompt)
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


# Runtime-configurable policy endpoints and simple UI for demo
from typing import List, Dict, Optional

@app.get("/config")
async def get_config():
	return {
		"blocking_labels": sorted(list(BLOCK_LABELS)),
		"suspicious_pass_threshold": model_service.suspicious_pass_threshold
	}

class UpdateConfig(BaseModel):
	blocking_labels: Optional[List[str]] = None
	suspicious_pass_threshold: Optional[float] = None

@app.post("/config")
async def update_config(cfg: UpdateConfig):
	global BLOCK_LABELS
	updated: Dict[str, object] = {}
	if cfg.blocking_labels is not None:
		BLOCK_LABELS = set([lbl.strip().lower() for lbl in cfg.blocking_labels if lbl.strip()])
		updated["blocking_labels"] = sorted(list(BLOCK_LABELS))
	if cfg.suspicious_pass_threshold is not None:
		try:
			val = float(cfg.suspicious_pass_threshold)
			model_service.suspicious_pass_threshold = val
			updated["suspicious_pass_threshold"] = val
		except Exception:
			raise HTTPException(status_code=400, detail="Invalid suspicious_pass_threshold")
	return {"updated": updated, **await get_config()}

class UIClassifyRequest(ClassifyRequest):
	pass

@app.post("/ui_classify")
async def ui_classify(req: UIClassifyRequest):
	label, scores, decision = model_service.classify(req.prompt)
	final_decision = apply_policy(label, decision)
	return {"label": label, "scores": scores, "decision": final_decision}


UI_HTML = """
<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Prompt Filter Demo</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 20px auto; }
.card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
label { margin-right: 12px; }
pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow: auto; }
button { padding: 8px 12px; }
</style>
</head>
<body>
<h2>Prompt Filter — Config & Test</h2>
<div class=\"card\">
  <h3>Config</h3>
  <div>
    <label><input type=\"checkbox\" id=\"blk-green\"> Block green</label>
    <label><input type=\"checkbox\" id=\"blk-yellow\"> Block yellow</label>
    <label><input type=\"checkbox\" id=\"blk-red\"> Block red</label>
  </div>
  <div style=\"margin-top:8px;\">
    Suspicious pass threshold: <input type=\"number\" id=\"thr\" step=\"0.01\" min=\"0\" max=\"1\" style=\"width:120px\" />
    <button onclick=\"saveCfg()\">Save</button>
  </div>
</div>
<div class=\"card\">
  <h3>Test Prompt</h3>
  <textarea id=\"prompt\" rows=\"4\" style=\"width:100%\" placeholder=\"Type your prompt...\"></textarea>
  <div style=\"margin-top:8px;\">
    <button onclick=\"doClassify()\">Classify</button>
  </div>
</div>
<div class=\"card\">
  <h3>Result</h3>
  <pre id=\"out\">(nothing yet)</pre>
</div>
<script>
async function loadCfg(){
  const r = await fetch('/config');
  const j = await r.json();
  document.getElementById('blk-green').checked = j.blocking_labels.includes('green');
  document.getElementById('blk-yellow').checked = j.blocking_labels.includes('yellow');
  document.getElementById('blk-red').checked = j.blocking_labels.includes('red');
  document.getElementById('thr').value = j.suspicious_pass_threshold;
}
async function saveCfg(){
  const labels = [];
  if(document.getElementById('blk-green').checked) labels.push('green');
  if(document.getElementById('blk-yellow').checked) labels.push('yellow');
  if(document.getElementById('blk-red').checked) labels.push('red');
  const thr = parseFloat(document.getElementById('thr').value);
  const r = await fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({blocking_labels: labels, suspicious_pass_threshold: thr})});
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
async function doClassify(){
  const p = document.getElementById('prompt').value;
  const r = await fetch('/ui_classify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({prompt: p})});
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
loadCfg();
</script>
</body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
	return HTMLResponse(content=UI_HTML)

UI_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Prompt Filter Demo</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 800px; margin: 20px auto; }
.card { border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }
label { margin-right: 12px; }
pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow: auto; }
button { padding: 8px 12px; }
</style>
</head>
<body>
<h2>Prompt Filter — Config & Test</h2>
<div class="card">
  <h3>Config</h3>
  <div>
    <label><input type="checkbox" id="blk-green"> Block green</label>
    <label><input type="checkbox" id="blk-yellow"> Block yellow</label>
    <label><input type="checkbox" id="blk-red"> Block red</label>
  </div>
  <div style="margin-top:8px;">
    Suspicious pass threshold: <input type="number" id="thr" step="0.01" min="0" max="1" style="width:120px" />
    <button onclick="saveCfg()">Save</button>
  </div>
</div>
<div class="card">
  <h3>Test Prompt</h3>
  <textarea id="prompt" rows="4" style="width:100%" placeholder="Type your prompt..."></textarea>
  <div style="margin-top:8px;">
    <button onclick="doClassify()">Classify</button>
  </div>
</div>
<div class="card">
  <h3>Result</h3>
  <pre id="out">(nothing yet)</pre>
</div>
<script>
async function loadCfg(){
  const r = await fetch('/config');
  const j = await r.json();
  document.getElementById('blk-green').checked = j.blocking_labels.includes('green');
  document.getElementById('blk-yellow').checked = j.blocking_labels.includes('yellow');
  document.getElementById('blk-red').checked = j.blocking_labels.includes('red');
  document.getElementById('thr').value = j.suspicious_pass_threshold;
}
async function saveCfg(){
  const labels = [];
  if(document.getElementById('blk-green').checked) labels.push('green');
  if(document.getElementById('blk-yellow').checked) labels.push('yellow');
  if(document.getElementById('blk-red').checked) labels.push('red');
  const thr = parseFloat(document.getElementById('thr').value);
  const r = await fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({blocking_labels: labels, suspicious_pass_threshold: thr})});
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
async function doClassify(){
  const p = document.getElementById('prompt').value;
  const r = await fetch('/ui_classify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({prompt: p})});
  const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j, null, 2);
}
loadCfg();
</script>
</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
	return HTMLResponse(content=UI_HTML)


@app.get("/model")
async def get_model():
	import os
	return {"model_source": os.getenv("MODEL_PATH") or os.getenv("HF_MODEL_ID") or "stub"}


# ------------------------------
# Curation storage and review
# ------------------------------
import json, uuid, threading, time

CURATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'curation')
PENDING_PATH = os.path.abspath(os.path.join(CURATION_DIR, 'pending.jsonl'))
CLASSIFIED_PATH = os.path.abspath(os.path.join(CURATION_DIR, 'classified.jsonl'))
REPLACE_PATH = os.path.abspath(os.path.join(CURATION_DIR, 'replace.json'))
PENDING_MAX = 1000
PER_CLASS_MAX = 10050

_file_lock = threading.Lock()
AUTO_INGEST = os.getenv('AUTO_INGEST', 'false').lower() in ('1','true','yes')


def _ensure_paths():
	os.makedirs(os.path.abspath(CURATION_DIR), exist_ok=True)
	for path in (PENDING_PATH, CLASSIFIED_PATH):
		if not os.path.exists(path):
			with open(path, 'w', encoding='utf-8') as f:
				pass


def _append_jsonl(path: str, obj: dict) -> None:
	with _file_lock:
		with open(path, 'a', encoding='utf-8') as f:
			f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def _read_jsonl(path: str):
	with _file_lock:
		if not os.path.exists(path):
			return []
		with open(path, 'r', encoding='utf-8') as f:
			return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, items):
	with _file_lock:
		with open(path, 'w', encoding='utf-8') as f:
			for it in items:
				f.write(json.dumps(it, ensure_ascii=False) + '\n')


def _pending_count() -> int:
	return len(_read_jsonl(PENDING_PATH))


def _classified_counts():
	items = _read_jsonl(CLASSIFIED_PATH)
	counts = {'0':0,'1':0,'2':0}
	for it in items:
		lbl = str(it.get('classification'))
		if lbl in counts:
			counts[lbl] += 1
	return counts


def _maybe_auto_ingest(prompt: str):
	if not AUTO_INGEST:
		return
	_ensure_paths()
	pending = _read_jsonl(PENDING_PATH)
	if len(pending) >= PENDING_MAX:
		return
	item = {"id": str(uuid.uuid4()), "user_message": prompt, "timestamp": int(time.time())}
	_append_jsonl(PENDING_PATH, item)


@app.post('/ingest')
async def ingest(item: dict):
	"""Body: {"prompt": str}"""
	prompt = (item or {}).get('prompt')
	if not isinstance(prompt, str) or not prompt.strip():
		raise HTTPException(status_code=400, detail='prompt required')
	_ensure_paths()
	pending = _read_jsonl(PENDING_PATH)
	if len(pending) >= PENDING_MAX:
		raise HTTPException(status_code=400, detail='pending queue full (>=1000)')
	obj = {"id": str(uuid.uuid4()), "user_message": prompt.strip(), "timestamp": int(time.time())}
	_append_jsonl(PENDING_PATH, obj)
	return {"queued": True, "id": obj['id'], "pending": len(pending)+1}


@app.get('/review/next')
async def review_next():
	_ensure_paths()
	pending = _read_jsonl(PENDING_PATH)
	if not pending:
		return {"item": None, "pending": 0, "counts": _classified_counts()}
	item = pending[0]
	return {"item": {"id": item['id'], "user_message": item['user_message']}, "pending": len(pending), "counts": _classified_counts()}


@app.post('/review/{item_id}')
async def review_submit(item_id: str, body: dict):
	label = str((body or {}).get('label'))
	if label not in ('0','1','2'):
		raise HTTPException(status_code=400, detail='label must be 0,1,2')
	_ensure_paths()
	pending = _read_jsonl(PENDING_PATH)
	idx = next((i for i,it in enumerate(pending) if it.get('id')==item_id), None)
	if idx is None:
		raise HTTPException(status_code=404, detail='pending item not found')
	counts = _classified_counts()
	if counts[label] >= PER_CLASS_MAX:
		raise HTTPException(status_code=400, detail=f'class {label} cap reached (>{PER_CLASS_MAX})')
	item = pending.pop(idx)
	_write_jsonl(PENDING_PATH, pending)
	_append_jsonl(CLASSIFIED_PATH, {"id": item['id'], "user_message": item['user_message'], "classification": label, "timestamp": int(time.time())})
	counts[label] += 1
	return {"classified": True, "pending": len(pending), "counts": counts}


@app.get('/export/replace')
async def export_replace():
	_ensure_paths()
	classified = _read_jsonl(CLASSIFIED_PATH)
	export = [{"user_message": it['user_message'], "classification": str(it['classification'])} for it in classified]
	with _file_lock:
		with open(REPLACE_PATH, 'w', encoding='utf-8') as f:
			json.dump(export, f, ensure_ascii=False, indent=2)
	return {"wrote": REPLACE_PATH, "count": len(export)}


@app.get('/curation/stats')
async def curation_stats():
	_ensure_paths()
	return {"pending": _pending_count(), "counts": _classified_counts()}
