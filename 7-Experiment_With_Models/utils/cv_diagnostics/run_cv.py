#!/usr/bin/env python3
"""Cross-validation diagnostics for all three model families.

Usage examples
--------------
python utils/cv_diagnostics/run_cv.py --family traditional_ml --model logistic_regression
python utils/cv_diagnostics/run_cv.py --family keras --model cnn
python utils/cv_diagnostics/run_cv.py --family transformer --model bert

If --family all (default) it will iterate over the models specified in
config/config.json for each family.

Output is printed to stdout and appended to logs/cv_report.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support

# project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from config.config import load_config  # type: ignore
from data.data_loader import load_data  # type: ignore
from training.trainer import get_model_configs  # type: ignore
from models.keras_models import create_model  # type: ignore
from models.utils.model_manager import model_manager  # type: ignore
from utils.data_issues import prepare_text_data  # reuse tokenization helper if exists

# Tensorflow / transformers imports – guarded because traditional ML path may not need them
try:
    import tensorflow as tf
except Exception:
    tf = None  # type: ignore

try:
    import torch
    from transformers import TrainingArguments, Trainer
except Exception:
    torch = None  # type: ignore

LOG_PATH = ROOT / "logs" / "cv_report.txt"
LOG_PATH.parent.mkdir(exist_ok=True)

def log(msg: str):
    print(msg)
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(msg + "\n")

def cv_traditional(model_key: str, X: List[str], y: np.ndarray):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    cfg = get_model_configs()
    if model_key not in cfg:
        log(f"[traditional] Unknown model: {model_key}")
        return
    model = cfg[model_key]["model"]
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english"),
        model,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        pipeline.fit([X[i] for i in train_idx], y[train_idx])
        y_pred = pipeline.predict([X[i] for i in test_idx])
        f1 = f1_score(y[test_idx], y_pred, average="macro")
        scores.append(f1)
    scores = np.array(scores)
    log(f"[traditional][{model_key}] macro-F1 mean={scores.mean():.4f} std={scores.std():.4f} scores={np.round(scores,4)}")

def cv_keras(model_key: str, X: List[str], y: np.ndarray, num_classes: int):
    if tf is None:
        log("Tensorflow not available – skipping Keras CV")
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # reuse simple tokeniser
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tok = Tokenizer(num_words=10000, oov_token="<OOV>")
        tok.fit_on_texts(X_train)
        def encode(texts):
            seqs = tok.texts_to_sequences(texts)
            return pad_sequences(seqs, maxlen=100, padding="post", truncating="post")
        X_train_pad = encode(X_train)
        X_test_pad = encode(X_test)

        model = create_model(model_key, input_shape=(100, 1), num_classes=num_classes, config={})
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train_pad, y_train, epochs=3, batch_size=32, verbose=0)
        y_pred = model.predict(X_test_pad, verbose=0).argmax(axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")
        scores.append(f1)
        tf.keras.backend.clear_session()
    scores = np.array(scores)
    log(f"[keras][{model_key}] macro-F1 mean={scores.mean():.4f} std={scores.std():.4f} scores={np.round(scores,4)}")

def cv_transformer(model_key: str, X: List[str], y: np.ndarray, num_classes: int):
    if torch is None:
        log("PyTorch/transformers not available – skipping transformer CV")
        return
    model_name, tokenizer_cls, model_cls = model_manager.get_model_components(model_key)
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold to save time
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        enc_train = encode(X_train)
        enc_test = encode(X_test)
        enc_train["labels"] = torch.tensor(y_train)
        enc_test["labels"] = torch.tensor(y_test)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc):
                self.enc = enc
            def __len__(self):
                return len(self.enc["input_ids"])
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.enc.items()}

        train_ds, test_ds = DS(enc_train), DS(enc_test)
        model = model_cls.from_pretrained(model_name, num_labels=num_classes).to(device)
        args = TrainingArguments(output_dir="tmp", num_train_epochs=1, per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8, learning_rate=2e-5, logging_steps=50,
                                  disable_tqdm=True, evaluation_strategy="no", report_to=None)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds)
        trainer.train()
        preds = model(**{k: v.to(device) for k, v in enc_test.items() if k != "labels"}).logits.argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_test, preds, average="macro")
        scores.append(f1)
        del model; torch.cuda.empty_cache()
    scores = np.array(scores)
    log(f"[transformer][{model_key}] macro-F1 mean={scores.mean():.4f} std={scores.std():.4f} scores={np.round(scores,4)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="all", choices=["all", "traditional_ml", "keras", "transformer"], help="Model family to run CV on")
    parser.add_argument("--model", default=None, help="Specific model key (overrides list in config)")
    args = parser.parse_args()

    LOG_PATH.write_text("")  # clear old

    cfg = load_config()
    data = load_data(cfg)
    X = data["X_train"] + data["X_test"]
    y = np.concatenate([data["y_train"], data["y_test"]])
    num_classes = data["num_classes"]

    if args.family in ("all", "traditional_ml"):
        models = [args.model] if args.model else cfg["traditional_ml"]["models"]
        for m in models:
            cv_traditional(m, X, y)

    if args.family in ("all", "keras"):
        models = [args.model] if args.model else [m for m in cfg["deep_learning"]["models"] if m not in {"bert","distilbert","roberta","albert","distilroberta"}]
        for m in models:
            cv_keras(m, X, y, num_classes)

    if args.family in ("all", "transformer"):
        models = [args.model] if args.model else [m for m in cfg["transformers"]["models"]]
        for m in models:
            cv_transformer(m, X, y, num_classes)

    log("CV diagnostics complete. Report saved to logs/cv_report.txt")


if __name__ == "__main__":
    main() 