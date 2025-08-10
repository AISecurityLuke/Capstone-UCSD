#!/usr/bin/env python3
"""Utility functions for custom class-specific metrics.

Focus metrics:
  • recall for class 2 (malicious)
  • precision for class 1 (suspicious)
  • F1 for class 0 (benign)
Also provides a combined "custom score" to use in GridSearchCV.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, make_scorer

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_custom_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return dict with recall_c2, precision_c1, f1_c0 and combined score.
    
    The custom score heavily penalizes any misclassification between classes 0 and 2:
    - Exponential penalty scaling means even a few critical errors severely impact score
    - Separate tracking for benign→malicious vs malicious→benign errors
    - Score can approach zero with enough critical errors
    """
    # Ensure numpy arrays for fancy indexing
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Order labels explicitly 0,1,2 to get per-class arrays in a known order
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )

    # Extract required metrics
    precision_c1 = precision_arr[1]
    recall_c2 = recall_arr[2]
    f1_c0 = f1_arr[0]

    # Calculate critical misclassifications with separate tracking
    total_samples = len(y_true)
    
    # Count benign predicted as malicious (false alarms)
    misclass_0_to_2 = np.sum((y_true == 0) & (y_pred == 2))
    rate_0_to_2 = misclass_0_to_2 / total_samples if total_samples > 0 else 0
    
    # Count malicious predicted as benign (missed threats)
    misclass_2_to_0 = np.sum((y_true == 2) & (y_pred == 0))
    rate_2_to_0 = misclass_2_to_0 / total_samples if total_samples > 0 else 0
    
    # Combined critical error rate
    critical_error_rate = rate_0_to_2 + rate_2_to_0
    
    # Exponential penalty that grows quickly with error rate
    # exp(-5) ≈ 0.0067, so 20% critical errors will devastate the score
    penalty_factor = np.exp(-5 * critical_error_rate)
    
    # Base score calculation (unchanged weights)
    base_score = 0.5 * recall_c2 + 0.3 * precision_c1 + 0.2 * f1_c0
    
    # Apply exponential penalty to base score
    custom_score = base_score * penalty_factor

    return {
        "recall_c2": recall_c2,
        "precision_c1": precision_c1,
        "f1_c0": f1_c0,
        "custom_score": custom_score,
        "critical_misclass_rate": critical_error_rate,
        "false_alarm_rate": rate_0_to_2,    # benign→malicious rate
        "missed_threat_rate": rate_2_to_0,   # malicious→benign rate
        "penalty_factor": penalty_factor     # for debugging
    }

# ---------------------------------------------------------------------------
# Scorer for sklearn grid-search
# ---------------------------------------------------------------------------

def _custom_score_func(y_true, y_pred):
    return compute_custom_metrics(y_true, y_pred)["custom_score"]

sklearn_custom_scorer = make_scorer(_custom_score_func, greater_is_better=True) 