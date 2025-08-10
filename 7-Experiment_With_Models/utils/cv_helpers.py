from typing import List, Tuple
import re
import numpy as np
from sklearn.model_selection import GroupKFold

__all__ = ["create_group_folds"]

def _fingerprint(text: str) -> str:
    """Simple heuristic to fingerprint a prompt for grouping (same as data_loader)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def create_group_folds(X: List[str], y: np.ndarray, k: int = 4) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) for GroupKFold on text fingerprints."""
    groups = np.array([_fingerprint(t) for t in X])
    gkf = GroupKFold(n_splits=k)
    return list(gkf.split(X, y, groups)) 