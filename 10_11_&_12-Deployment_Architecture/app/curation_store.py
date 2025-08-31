import os
import json
import time
import uuid
import threading
from typing import Optional, Dict, Any, List, Tuple


class CurationStore:
    def __init__(self, base_dir: Optional[str] = None,
                 max_pending: int = 5000,
                 max_per_class: int = 10050) -> None:
        self.base_dir = base_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'curation')
        self.base_dir = os.path.abspath(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.pending_path = os.path.join(self.base_dir, 'pending.jsonl')
        self.classified_path = os.path.join(self.base_dir, 'classified.jsonl')
        self.replace_path = os.path.join(self.base_dir, 'replace.json')
        self.max_pending = max_pending
        self.max_per_class = max_per_class
        self._lock = threading.Lock()

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        items: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items

    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    def _rewrite_jsonl(self, path: str, items: List[Dict[str, Any]]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            pending = self._read_jsonl(self.pending_path)
            classified = self._read_jsonl(self.classified_path)
            class_counts = {'0': 0, '1': 0, '2': 0}
            for it in classified:
                lbl = str(it.get('label', ''))
                if lbl in class_counts:
                    class_counts[lbl] += 1
            return {
                'pending': len(pending),
                'classified_counts': class_counts,
            }

    def enqueue(self, prompt: str) -> Tuple[bool, Dict[str, Any]]:
        ts = int(time.time())
        with self._lock:
            pending = self._read_jsonl(self.pending_path)
            if len(pending) >= self.max_pending:
                return False, {'error': 'pending_capacity_reached', 'max_pending': self.max_pending}
            rec = {
                'id': str(uuid.uuid4()),
                'prompt': prompt,
                'ts': ts,
            }
            self._append_jsonl(self.pending_path, rec)
            return True, rec

    def next_unclassified(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            pending = self._read_jsonl(self.pending_path)
            return pending[0] if pending else None

    def list_pending(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            items = self._read_jsonl(self.pending_path)
            return items[: max(0, min(limit, len(items)))]

    def classify(self, rec_id: str, label: str) -> Tuple[bool, Dict[str, Any]]:
        label = str(label)
        if label not in {'0', '1', '2'}:
            return False, {'error': 'invalid_label'}
        with self._lock:
            # enforce per-class cap
            classified = self._read_jsonl(self.classified_path)
            class_counts = {'0': 0, '1': 0, '2': 0}
            for it in classified:
                lbl = str(it.get('label', ''))
                if lbl in class_counts:
                    class_counts[lbl] += 1
            if class_counts[label] >= self.max_per_class:
                return False, {'error': 'class_capacity_reached', 'label': label, 'max_per_class': self.max_per_class}

            # pop from pending
            pending = self._read_jsonl(self.pending_path)
            idx = next((i for i, it in enumerate(pending) if it.get('id') == rec_id), None)
            if idx is None:
                return False, {'error': 'not_found'}
            rec = pending.pop(idx)
            rec['label'] = label
            rec['classified_ts'] = int(time.time())
            # persist
            self._rewrite_jsonl(self.pending_path, pending)
            self._append_jsonl(self.classified_path, rec)
            return True, rec

    def export_replace(self) -> Dict[str, Any]:
        with self._lock:
            classified = self._read_jsonl(self.classified_path)
            # map to combined.json structure
            data = [{'user_message': it.get('prompt', ''), 'classification': str(it.get('label', ''))} for it in classified]
            with open(self.replace_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return {
                'path': self.replace_path,
                'count': len(data),
                'counts': {
                    '0': sum(1 for d in data if d['classification'] == '0'),
                    '1': sum(1 for d in data if d['classification'] == '1'),
                    '2': sum(1 for d in data if d['classification'] == '2'),
                }
            }


# singleton store
store = CurationStore()


