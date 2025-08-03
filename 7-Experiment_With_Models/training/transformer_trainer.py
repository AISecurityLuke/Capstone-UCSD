#!/usr/bin/env python3
"""
Transformer model training with HuggingFace Trainer
Separated from adaptive_trainer so we keep DL (Keras) and transformer (HF) flows isolated.
Includes recommended settings: warm-up steps, learning-rate scheduler, weight decay, gradient clipping.
"""

import os
import csv
import logging
import numpy as np
from typing import Dict, List, Tuple
import json

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from models.utils.model_manager import model_manager

# -----------------------------------------------------------------------------
# Group-aware K-Fold helper (non-intrusive)
# -----------------------------------------------------------------------------

from utils.cv_helpers import create_group_folds


def _aggregate_cv_results(results):
    """Return (mean, std) of a list of floats."""
    import numpy as np
    arr = np.array(results, dtype=float)
    return float(arr.mean()), float(arr.std())

# -----------------------------------------------------------------------------
# Core training logic
# -----------------------------------------------------------------------------

TRANSFORMER_TYPES = {"bert", "distilbert", "roberta", "albert", "distilroberta", "bert_large", "roberta_large"}


def compute_metrics(eval_pred):
    """Compute metrics function for HuggingFace Trainer (uses macro F1)"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}


def train_single_transformer(model_key: str,
                              X_train: List[str],
                              y_train: List[int],
                              X_test: List[str],
                              y_test: List[int],
                              num_classes: int,
                              config: Dict,
                              ext_data=None) -> Dict:
    """Train a single HuggingFace transformer model and return metrics."""
    logging.info(f"Training HuggingFace model: {model_key}")
    
    # Disable integrations to prevent conflicts
    import os
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = ""

    # Tokenizers parallelism (configurable)
    if config.get('transformers', {}).get('tokenizers_parallelism', False):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Debug: Check for None in X_train/X_test
    if any(x is None for x in X_train):
        logging.error(f"[DEBUG] None found in X_train at indices: {[i for i, x in enumerate(X_train) if x is None]}")
    if any(x is None for x in X_test):
        logging.error(f"[DEBUG] None found in X_test at indices: {[i for i, x in enumerate(X_test) if x is None]}")
    # Debug: Print type and value of first element
    logging.info(f"[DEBUG] Type of X_train[0]: {type(X_train[0])}, value: {X_train[0]}")
    logging.info(f"[DEBUG] Type of X_test[0]: {type(X_test[0])}, value: {X_test[0]}")

    model_name, tokenizer_cls, model_cls = model_manager.get_model_components(model_key)
    max_length = (
        config.get("model_architectures", {})
        .get(model_key, {})
        .get("default_config", {})
        .get("max_length", 256)
    )

    tokenizer = tokenizer_cls.from_pretrained(model_name)

    # Create datasets using HuggingFace datasets library
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    # Create train dataset
    train_dict = {"text": X_train, "label": y_train}
    train_dataset = Dataset.from_dict(train_dict)
    num_proc_tok = config.get('transformers', {}).get('tokenization_num_proc') or None
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc_tok)
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset = train_dataset.remove_columns(["text"])
    
    # Create eval dataset
    eval_dict = {"text": X_test, "label": y_test}
    eval_dataset = Dataset.from_dict(eval_dict)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=num_proc_tok)
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.remove_columns(["text"])
    
    # Debug: Check dataset structure
    logging.info(f"[DEBUG] Train dataset features: {train_dataset.features}")
    logging.info(f"[DEBUG] Eval dataset features: {eval_dataset.features}")
    logging.info(f"[DEBUG] Train dataset length: {len(train_dataset)}")
    logging.info(f"[DEBUG] Eval dataset length: {len(eval_dataset)}")

    model = model_cls.from_pretrained(model_name, num_labels=num_classes)

    # --------------------
    # Class-weight handling
    # --------------------
    class_weights = config.get('class_weights', [1.0, 1.0, 1.0])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logging.info(f"[DEBUG] Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # TrainingArguments with warm-up, scheduler, weight decay, clipping
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Parallelism parameters
    # ------------------------------------------------------------------
    num_workers = config.get('transformers', {}).get('dataloader_num_workers', 0)
    pin_mem    = config.get('transformers', {}).get('dataloader_pin_memory', False)

    # ------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------
    logging_steps_cfg = config.get('transformers', {}).get('logging_steps', 0)
    logging_kwargs = {}
    if logging_steps_cfg and logging_steps_cfg > 0:
        logging_kwargs.update({
            "logging_strategy": "steps",
            "logging_steps": logging_steps_cfg,
            "logging_first_step": True,
        })

    training_args = TrainingArguments(
        output_dir=f"models/{model_key}_hf",
        num_train_epochs=config.get('transformers', {}).get('max_epochs', 3),
        per_device_train_batch_size=config.get('transformers', {}).get('batch_size', 8),
        per_device_eval_batch_size=config.get('transformers', {}).get('batch_size', 8),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=config.get('model_architectures', {}).get(model_key, {}).get('default_config', {}).get('learning_rate', 2e-5),
        weight_decay=0.01,
        warmup_ratio=config.get('transformers', {}).get('warmup_ratio', 0.1),
        lr_scheduler_type="linear",
        fp16=(
            config.get('transformers', {}).get('fp16', False)
            and torch.cuda.is_available()
        ),
        max_grad_norm=1.0,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=num_workers,
        **logging_kwargs,
    )

    # --------------------------------------------------------------
    # Custom Trainer with class-weighted loss
    # --------------------------------------------------------------
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights_tensor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights_tensor = class_weights_tensor

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.device.type == "mps":
                # MPS backend ignores weight tensor – compute weighted loss on CPU instead
                cpu_logits = logits.detach().to("cpu")
                cpu_labels = labels.detach().to("cpu")
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor.to("cpu"))
                loss = loss_fct(cpu_logits.view(-1, cpu_logits.shape[-1]), cpu_labels.view(-1))
                # bring loss back to original device for backward pass
                loss = loss.to(logits.device)
            else:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor.to(logits.device))
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Use DataCollatorWithPadding to ensure consistent sequence lengths
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None, return_tensors="pt")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights_tensor=class_weights_tensor.to(model.device),
        data_collator=data_collator,
    )

    # Remove MLflow, W&B, TensorBoard callbacks (HF may auto-add)
    from transformers.integrations import MLflowCallback, WandbCallback, TensorBoardCallback
    for cb in (MLflowCallback, WandbCallback, TensorBoardCallback):
        try:
            trainer.remove_callback(cb)
        except ValueError:
            pass  # not present
    logging.info("[DEBUG] Integration callbacks removed from Trainer.")

    # ------------------------------------------------------------------
    # Early stopping (patience from config or default 3)
    # ------------------------------------------------------------------
    from transformers import EarlyStoppingCallback

    patience = config.get("transformers", {}).get("early_stopping_patience", 3)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    # Debug prints
    logging.info(
        f"[HF-DEBUG] {model_key}: batch_size={config.get('transformers', {}).get('batch_size', 8)}, total_steps={len(train_dataset) // (config.get('transformers', {}).get('batch_size', 8) * 10)}"
    )

    try:
        trainer.train()
    except Exception as e:
        import traceback
        logging.error(f"[ERROR] Trainer.train() failed for {model_key}: {str(e)}")
        logging.error(f"[ERROR] Full traceback: {traceback.format_exc()}")
        raise e

    # Evaluation
    preds = trainer.predict(eval_dataset)
    y_pred = preds.predictions.argmax(axis=1)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    import scipy.special

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    # ROC-AUC (macro, OVR)
    try:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        probs = scipy.special.softmax(preds.predictions, axis=1)
        roc_auc = roc_auc_score(y_test_binarized, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float('nan')
    accuracy = accuracy_score(y_test, y_pred)

    # Detailed per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join('results', 'reports', f"{model_key}.json"), 'w') as fp:
        json.dump(report, fp, indent=2)

    # Save model
    model_dir = os.path.join("models", f"{model_key}_hf_best")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # External blind-set evaluation
    f1_ext = float('nan')
    if ext_data:
        try:
            ext_dict = {"text": ext_data['X_ext'], "label": ext_data['y_ext']}
            ext_ds = Dataset.from_dict(ext_dict)
            ext_ds = ext_ds.rename_column("label", "labels")
            ext_ds = ext_ds.map(tokenize_function, batched=True, num_proc=num_proc_tok)
            preds_ext = trainer.predict(ext_ds).predictions.argmax(axis=1)
            _, _, f1_ext, _ = precision_recall_fscore_support(ext_data['y_ext'], preds_ext, average="macro")
        except Exception as e:
            logging.warning(f"{model_key}: external validation failed – {e}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1_ext": f1_ext,
    }


def train_and_evaluate_transformer_models(data_dict: Dict, config: Dict, ext_data=None) -> Tuple[List[str], List[str]]:
    """Train and evaluate transformer models"""
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    num_classes = data_dict["num_classes"]

    models_to_train = [m for m in config.get("transformers", {}).get("models", []) if m in TRANSFORMER_TYPES]

    successful_models: List[str] = []
    failed_models: List[str] = []

    # Ensure results CSV exists
    results_path = os.path.join("results", "results.csv")
    os.makedirs("results", exist_ok=True)

    for model_key in models_to_train:
        try:
            k = config.get('transformers', {}).get('cv_folds', 1)
            if k <= 1:
                # Original single split behaviour
                fold_results = [train_single_transformer(
                    model_key, X_train, y_train, X_test, y_test,
                    num_classes, config, ext_data)
                ]
            else:
                # Group-aware K-fold
                folds = create_group_folds(X_train, y_train, k=k)
                fold_results = []
                for fold_idx, (tr_idx, val_idx) in enumerate(folds):
                    logging.info(f"{model_key} – Fold {fold_idx+1}/{k}")
                    res = train_single_transformer(
                        model_key,
                        [X_train[i] for i in tr_idx], y_train[tr_idx],
                        [X_train[i] for i in val_idx], y_train[val_idx],
                        num_classes, config, ext_data)
                    fold_results.append(res)
                    torch.cuda.empty_cache()

            # Aggregate
            mean_f1, std_f1 = _aggregate_cv_results([r['f1'] for r in fold_results])
            best_res = max(fold_results, key=lambda r: r['f1'])

            # Run external validation on the best model
            if ext_data:
                try:
                    logging.info(f"{model_key}: Running external validation...")
                    # Load the best model for external validation
                    best_model = model_cls.from_pretrained(f"models/{model_key}_hf_best")
                    best_tokenizer = tokenizer_cls.from_pretrained(f"models/{model_key}_hf_best")
                    
                    # Create external dataset
                    ext_dict = {"text": ext_data['X_ext'], "label": ext_data['y_ext']}
                    ext_ds = Dataset.from_dict(ext_dict)
                    ext_ds = ext_ds.rename_column("label", "labels")
                    
                    def ext_tokenize_function(examples):
                        return best_tokenizer(examples["text"], truncation=True, max_length=max_length)
                    
                    ext_ds = ext_ds.map(ext_tokenize_function, batched=True, num_proc=num_proc_tok)
                    
                    # Create trainer for prediction
                    ext_trainer = WeightedTrainer(
                        model=best_model,
                        args=TrainingArguments(output_dir="temp", per_device_eval_batch_size=16),
                        compute_metrics=compute_metrics,
                    )
                    
                    # Predict on external data
                    preds_ext = ext_trainer.predict(ext_ds).predictions.argmax(axis=1)
                    _, _, f1_ext, _ = precision_recall_fscore_support(ext_data['y_ext'], preds_ext, average="macro")
                    best_res['f1_ext'] = f1_ext
                    logging.info(f"{model_key}: External F1 = {f1_ext:.4f}")
                except Exception as e:
                    logging.warning(f"{model_key}: External validation failed – {e}")
                    best_res['f1_ext'] = float('nan')
            else:
                best_res['f1_ext'] = float('nan')

            # Calculate per-class F1 scores
            class_f1_scores = precision_recall_fscore_support(y_test, y_pred, average=None)[2]
            class_f1_dict = {i: f1 for i, f1 in enumerate(class_f1_scores)}
            
            # Add class F1 scores to the results
            best_res['class_f1'] = class_f1_dict
            
            # ------------------------------------------------------------
            # Log misclassified examples (max 50 per class per model)
            # ------------------------------------------------------------
            misclassified_examples = {}
            for true_class in range(3):
                for pred_class in range(3):
                    if true_class != pred_class:
                        mask = (y_test == true_class) & (y_pred == pred_class)
                        if mask.any():
                            misclassified_texts = [X_test[i] for i in np.where(mask)[0][:50]]  # Max 50 per confusion
                            key = f"true_{true_class}_predicted_{pred_class}"
                            misclassified_examples[key] = misclassified_texts
            
            # Save misclassified examples
            misclassified_path = os.path.join('results', 'results_misclassification.json')
            try:
                if os.path.exists(misclassified_path):
                    with open(misclassified_path, 'r') as f:
                        all_misclassified = json.load(f)
                else:
                    all_misclassified = {}
                
                all_misclassified[model_key] = {
                    "model_type": "huggingface_transformer",
                    "misclassified_examples": misclassified_examples
                }
                
                with open(misclassified_path, 'w') as f:
                    json.dump(all_misclassified, f, indent=2)
            except Exception as e:
                logging.warning(f"Failed to save misclassified examples for {model_key}: {e}")

            with open(results_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_key,
                    f"{best_res['precision']:.4f}",
                    f"{best_res['recall']:.4f}",
                    f"{best_res['f1']:.4f}",
                    f"{best_res['roc_auc']:.4f}",
                    str(config.get('model_architectures', {}).get(model_key, 'hf_transformer')),
                    "success",
                    f"{best_res['f1_ext']:.4f}",
                    f"{best_res['class_f1']}"
                ])

            logging.info(f"{model_key}: CV mean_f1={mean_f1:.4f} ± {std_f1:.4f} – Best fold F1={best_res['f1']:.4f}")
            successful_models.append(model_key)
        except Exception as e:
            logging.error(f"{model_key} failed: {e}")
            failed_models.append(model_key)

    logging.info(
        f"Transformer Summary: {len(successful_models)} successful, {len(failed_models)} failed"
    )
    return successful_models, failed_models 