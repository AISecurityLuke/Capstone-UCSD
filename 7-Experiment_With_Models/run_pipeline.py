#!/usr/bin/env python3
"""
Main entry point for the refactored ML experiment framework.

Model categories & where to enable them (all configured via config/config.json):

1. traditional_ml.models  → Classic scikit-learn models
   Possible values: "logistic_regression", "random_forest", "svm", "naive_bayes", "xgboost"

2. deep_learning.models   → (a) Keras architectures  +  (b) HuggingFace transformers
   Keras      : "cnn", "lstm", "bilstm", "transformer", "hybrid"
   Transformers: "bert", "distilbert", "roberta", "albert", "distilroberta"

   • Keras models are trained in training/adaptive_trainer.py
   • Transformers are filtered and trained in training/transformer_trainer.py

Adjust those lists in the JSON file to include or exclude models for a run.
"""

import sys
import os

# NEW -- ensure the working directory is the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Disable wandb to prevent interactive prompts
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_DISABLED', 'true')

# Disable Hugging-Face’s auto-added MLflow callback
os.environ.setdefault('HF_MLFLOW_DISABLE', '1')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------
# Load config EARLY to propagate logging settings (minimal invasive)
# ------------------------------------------------------------
from config.config import load_config
cfg_early = load_config()

# Propagate detailed logging preference via env var so that downstream
# modules can pick it up at import-time.
if cfg_early.get("detailed_logging", False):
    os.environ["DETAILED_LOGGING"] = "true"

# ------------------------------------------------------------
# Late imports (they will see env var)
# ------------------------------------------------------------
from experiment.experiment_runner import main
from utils.reproducibility import set_global_seed

if __name__ == "__main__":
    # Set reproducibility
    cfg = cfg_early  # reuse already-loaded config
    set_global_seed(cfg.get("random_seed", 42))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()