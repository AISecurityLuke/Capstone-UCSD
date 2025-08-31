#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

STATE_PATH = Path(__file__).resolve().parents[1] / "curation" / "curation_state.json"

BATCH_SIZE_PER_CLASS = 100
CLASSES = ["0", "1", "2"]
WINDOW_MAX = 10050


def load_state():
	with STATE_PATH.open("r") as f:
		return json.load(f)


def save_state(state):
	state["window_start_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
	with STATE_PATH.open("w") as f:
		json.dump(state, f, indent=2)


def can_replace(state):
	for c in CLASSES:
		if state["classes"][c]["replacements_in_window"] + BATCH_SIZE_PER_CLASS > WINDOW_MAX:
			return False
	return True


def apply_batch(state):
	for c in CLASSES:
		state["classes"][c]["next_start_idx"] += BATCH_SIZE_PER_CLASS
		state["classes"][c]["replacements_in_window"] += BATCH_SIZE_PER_CLASS
	return state


def mark_retrain(state):
	for c in CLASSES:
		state["classes"][c]["last_retrain_idx"] = state["classes"][c]["next_start_idx"]
	return state


def main():
	if len(sys.argv) < 2:
		print("Usage: manage_curation.py [apply-batch|mark-retrain]")
		return 1
	cmd = sys.argv[1]
	state = load_state()
	if cmd == "apply-batch":
		if not can_replace(state):
			print("ERROR: Replacement window cap reached; cannot apply batch.")
			return 2
		state = apply_batch(state)
		save_state(state)
		print("Applied balanced batch (100/100/100); updated next_start_idx and counters.")
		return 0
	elif cmd == "mark-retrain":
		state = mark_retrain(state)
		save_state(state)
		print("Marked retrain point at current next_start_idx for all classes.")
		return 0
	else:
		print("Unknown command")
		return 3


if __name__ == "__main__":
	sys.exit(main())
