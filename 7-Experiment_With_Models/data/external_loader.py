#!/usr/bin/env python3
"""Small helper to load an external “blind-set” dataset for final validation.
The file must be a JSON list of objects with the same schema as training
( keys: `user_message`, `classification` ).

This loader re-uses the existing sanitiser and the label encoder that was
already fitted on the main dataset so label mapping stays consistent.
"""
from __future__ import annotations
import json, os, sys
from typing import Dict, List

# Make project root importable
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
sys.path.append(ROOT_DIR)

from utils.sanitizer import sanitize_text


def load_external_dataset(path: str, label_encoder, cleaning_cfg: Dict) -> Dict[str, List]:
    """Return dict with keys X_ext, y_ext.
    Args:
        path: path to blind-set json
        label_encoder: fitted sklearn LabelEncoder
        cleaning_cfg: config["preprocessing"]["text_cleaning"] sub-dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"External validation file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    texts, labels = [], []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        txt = obj.get("user_message", "")
        lbl = str(obj.get("classification", ""))
        # apply same cleaning as training loader
        if cleaning_cfg.get("remove_html", True):
            txt = sanitize_text(txt)
        if cleaning_cfg.get("normalize_whitespace", True):
            txt = " ".join(txt.split())

        texts.append(txt)
        labels.append(lbl)

    y_encoded = label_encoder.transform(labels)
    return {"X_ext": texts, "y_ext": y_encoded} 