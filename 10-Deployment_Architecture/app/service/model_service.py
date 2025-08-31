from typing import Dict, Tuple


class ModelService:
	def __init__(self, model_path: str, suspicious_pass_threshold: float = 0.4) -> None:
		self.model_path = model_path
		self.suspicious_pass_threshold = suspicious_pass_threshold
		# TODO: Load the actual model/artifacts here.
		self._ready = True

	def is_ready(self) -> bool:
		return self._ready

	def predict_scores(self, text: str) -> Dict[str, float]:
		# TODO: Replace with real inference. Return normalized scores for classes.
		# Placeholder deterministic stub for now.
		base = hash(text) % 1000 / 1000.0
		safe = max(0.0, min(1.0, 1 - base))
		suspicious = max(0.0, min(1.0, abs(0.5 - base)))
		malicious = max(0.0, min(1.0, base))
		s = safe + suspicious + malicious
		return {"safe": safe/s, "suspicious": suspicious/s, "malicious": malicious/s}

	def decide(self, scores: Dict[str, float]) -> Tuple[str, str]:
		# Labels: 0=green(safe), 1=yellow(suspicious), 2=red(malicious)
		label = max(scores, key=scores.get)
		if label == "malicious":
			return "red", "reject"
		if label == "suspicious":
			return "yellow", ("pass" if scores[label] < self.suspicious_pass_threshold else "reject")
		return "green", "pass"

	def classify(self, text: str):
		scores = self.predict_scores(text)
		label, decision = self.decide(scores)
		return label, scores, decision
