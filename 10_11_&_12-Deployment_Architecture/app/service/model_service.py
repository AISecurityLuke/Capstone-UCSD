from typing import Dict, Tuple
import os
from typing import Optional

# Optional HF imports (fallback to stub if unavailable)
try:
	from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
	import torch  # type: ignore
	_HAS_HF = True
except Exception:  # ImportError or runtime env issues
	_HAS_HF = False
	torch = None  # type: ignore
	AutoTokenizer = None  # type: ignore
	AutoModelForSequenceClassification = None  # type: ignore


class ModelService:
	def __init__(self, model_path: str, suspicious_pass_threshold: float = 0.4) -> None:
		self.model_path = model_path
		self.suspicious_pass_threshold = suspicious_pass_threshold
		self._ready: bool = False
		self._hf_model = None  # type: ignore
		self._tokenizer = None  # type: ignore
		self._device = None  # type: ignore
		self._maybe_load_hf()

	def _maybe_load_hf(self) -> None:
		model_id = (self.model_path or os.getenv("HF_MODEL_ID", "")).strip()
		if not (_HAS_HF and model_id):
			# No HF available or no model specified → operate in stub mode
			self._ready = True
			return
		# Set device
		self._device = "cuda" if torch and torch.cuda.is_available() else "cpu"  # type: ignore
		# Load tokenizer and model
		self._tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore
		self._hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)  # type: ignore
		if torch is not None:
			self._hf_model = self._hf_model.to(self._device)  # type: ignore
		self._hf_model.eval()  # type: ignore
		self._ready = True

	def is_ready(self) -> bool:
		return self._ready

	def predict_scores(self, text: str) -> Dict[str, float]:
		# HF path
		if self._hf_model is not None and self._tokenizer is not None and torch is not None:
			with torch.no_grad():  # type: ignore
				inputs = self._tokenizer(  # type: ignore
					text, return_tensors="pt", truncation=True, max_length=512
				)
				if self._device:
					inputs = {k: v.to(self._device) for k, v in inputs.items()}
				outputs = self._hf_model(**inputs)  # type: ignore
				probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()  # type: ignore
				if isinstance(probs, float):  # single value edge-case
					probs = [probs]
				# Map to expected keys if 3-class; otherwise fall back to generic indices
				keys = ["safe", "suspicious", "malicious"] if len(probs) == 3 else [str(i) for i in range(len(probs))]
				return {k: float(v) for k, v in zip(keys, probs)}
		# Stub path: deterministic pseudo-scores
		base = (hash(text) % 1000) / 1000.0
		safe = max(0.0, min(1.0, 1 - base))
		suspicious = max(0.0, min(1.0, abs(0.5 - base)))
		malicious = max(0.0, min(1.0, base))
		s = safe + suspicious + malicious
		return {"safe": safe/s, "suspicious": suspicious/s, "malicious": malicious/s}

	def decide(self, scores: Dict[str, float]) -> Tuple[str, str]:
		# Labels: green(safe), yellow(suspicious), red(malicious)
		label_key = max(scores, key=scores.get)
		if label_key == "malicious":
			return "red", "reject"
		if label_key == "suspicious":
			return "yellow", ("pass" if scores[label_key] < self.suspicious_pass_threshold else "reject")
		return "green", "pass"

	def classify(self, text: str):
		scores = self.predict_scores(text)
		label, decision = self.decide(scores)
		return label, scores, decision
