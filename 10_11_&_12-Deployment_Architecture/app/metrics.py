import time
import threading
from statistics import mean


class _Metrics:
	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._latencies_ms = []
		self._max_samples = 1000
		self.total_requests = 0
		self.total_errors = 0

	def record(self, latency_ms: float, error: bool = False) -> None:
		with self._lock:
			self.total_requests += 1
			if error:
				self.total_errors += 1
			self._latencies_ms.append(latency_ms)
			if len(self._latencies_ms) > self._max_samples:
				self._latencies_ms = self._latencies_ms[-self._max_samples:]

	def _percentile(self, p: float) -> float:
		with self._lock:
			if not self._latencies_ms:
				return 0.0
			arr = sorted(self._latencies_ms)
			k = int(round((p/100.0) * (len(arr)-1)))
			return arr[max(0, min(k, len(arr)-1))]

	def snapshot(self) -> dict:
		with self._lock:
			avg = mean(self._latencies_ms) if self._latencies_ms else 0.0
			return {
				"total_requests": self.total_requests,
				"total_errors": self.total_errors,
				"latency_ms": {
					"avg": round(avg, 2),
					"p50": round(self._percentile(50), 2),
					"p95": round(self._percentile(95), 2)
				}
			}


metrics = _Metrics()
