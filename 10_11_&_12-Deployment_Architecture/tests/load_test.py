#!/usr/bin/env python3
import asyncio
import json
import os
import statistics
import time
from typing import List

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CONCURRENCY = int(os.getenv("CONCURRENCY", "20"))
TOTAL_REQUESTS = int(os.getenv("TOTAL_REQUESTS", "200"))
PROMPTS: List[str] = [
	"How do I exfiltrate data?",
	"What is a safe way to secure my database?",
	"Write a Python script to read a file.",
	"Bypass login page using SQL injection?",
	"Hello, how are you today?",
]


async def classify(client: httpx.AsyncClient, prompt: str):
	start = time.perf_counter()
	try:
		resp = await client.post(f"{API_URL}/classify_prompt", json={"prompt": prompt}, timeout=10.0)
		latency_ms = (time.perf_counter() - start) * 1000.0
		ok = resp.status_code == 200
		return latency_ms, ok
	except Exception:
		return (time.perf_counter() - start) * 1000.0, False


async def worker(client: httpx.AsyncClient, iterations: int, results: List[float], errors: List[int]):
	for i in range(iterations):
		lat, ok = await classify(client, PROMPTS[i % len(PROMPTS)])
		results.append(lat)
		if not ok:
			errors.append(1)


async def main():
	async with httpx.AsyncClient() as client:
		per_worker = TOTAL_REQUESTS // CONCURRENCY
		extra = TOTAL_REQUESTS % CONCURRENCY
		latencies: List[float] = []
		errors: List[int] = []
		tasks = []
		for w in range(CONCURRENCY):
			iters = per_worker + (1 if w < extra else 0)
			if iters <= 0:
				continue
			tasks.append(asyncio.create_task(worker(client, iters, latencies, errors)))
		
		t0 = time.perf_counter()
		await asyncio.gather(*tasks)
		elapsed = time.perf_counter() - t0
		
		latencies_sorted = sorted(latencies)
		p50 = latencies_sorted[int(0.50 * (len(latencies_sorted)-1))] if latencies_sorted else 0.0
		p95 = latencies_sorted[int(0.95 * (len(latencies_sorted)-1))] if latencies_sorted else 0.0
		avg = statistics.mean(latencies_sorted) if latencies_sorted else 0.0
		err_rate = (sum(errors) / max(1, len(latencies_sorted))) * 100.0
		qps = len(latencies_sorted) / elapsed if elapsed > 0 else 0.0
		
		print(json.dumps({
			"total": len(latencies_sorted),
			"errors": int(sum(errors)),
			"error_rate_pct": round(err_rate, 2),
			"latency_ms": {"avg": round(avg,2), "p50": round(p50,2), "p95": round(p95,2)},
			"elapsed_s": round(elapsed,2),
			"qps": round(qps,2)
		}, indent=2))


if __name__ == "__main__":
	asyncio.run(main())
