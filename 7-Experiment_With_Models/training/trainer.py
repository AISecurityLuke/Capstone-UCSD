#!/usr/bin/env python3
"""
Traditional ML model training with hyperparameter optimization
"""

import logging
import numpy as np
import pandas as pd
import os
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import joblib
import csv
import signal
from sklearn.utils.class_weight import compute_class_weight
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from utils.sanitizer import escape_for_logging

# Ensure reports directory exists
os.makedirs(os.path.join('results', 'reports'), exist_ok=True)

# ------------------------------------------------------------
# Optional detailed logging controlled by env var
# ------------------------------------------------------------
if os.environ.get("DETAILED_LOGGING", "false").lower() == "true":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("[trainer] Detailed logging enabled.")

def get_model_configs():
    """Get model configurations for hyperparameter tuning"""
    return {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': [
                {'C': [0.1, 1, 10], 'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
                {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs', 'saga']}
            ]
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'svm': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'naive_bayes': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }

def train_single_model(model_name, X_train, y_train, X_test, y_test, config, ext_data=None):
    """Train a single traditional ML model with comprehensive tuning"""
    start_time = time.time()
    logging.info(f"Training {model_name} with {len(X_train)} samples")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    if model_name not in model_configs:
        logging.error(f"Unknown model: {model_name}")
        return None
    
    # Vectorize text data
    logging.info(f"Vectorizing text data for {model_name}...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logging.info(f"Vectorization complete. Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}")
    
    # Handle class weights / sample weights
    use_custom_weights = config.get('use_class_weights', False)
    custom_weights = config.get('class_weights', [])
    unique_classes = np.unique(y_train)

    if use_custom_weights and custom_weights and len(custom_weights) == len(unique_classes):
        class_weights = np.array(custom_weights)
    else:
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)

    class_weight_dict = dict(zip(unique_classes, class_weights))

    # Build sample_weight vector for algorithms that do not take class_weight
    sample_weight = np.array([class_weight_dict[label] for label in y_train])

    # If estimator supports class_weight param, set it
    if 'class_weight' in model_configs[model_name]['model'].get_params().keys():
        model_configs[model_name]['model'].set_params(class_weight=class_weight_dict)

    # Get model and parameters
    model = model_configs[model_name]['model']
    param_grid = model_configs[model_name]['params']
    
    # Hyperparameter tuning
    cv_folds = config.get('traditional_ml', {}).get('cv_folds', 5)
    n_jobs = config.get('traditional_ml', {}).get('n_jobs', -1)
    timeout = config.get('traditional_ml', {}).get('timeout', 600)
    
    logging.info(f"Starting GridSearchCV for {model_name} with {cv_folds} folds, {n_jobs} jobs")
    logging.info(f"Parameter grid size: {len(param_grid)} parameters to test")
    
    try:
        search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring='f1_macro',
            n_jobs=n_jobs, verbose=2, error_score='raise'  # More verbose, raise errors
        )
        class TimeoutException(Exception):
            pass
        def handler(signum, frame):
            raise TimeoutException(f"GridSearchCV timed out after {timeout} seconds")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            search.fit(X_train_vec, y_train, sample_weight=sample_weight)
        except Exception as e:
            logging.error(f"GridSearchCV failed for {model_name}: {e}")
            raise
        finally:
            signal.alarm(0)
    except Exception as e:
        logging.error(f"{model_name} - Exception during GridSearchCV: {e}")
        return None
    
    cv_start_time = time.time()
    cv_time = time.time() - cv_start_time
    
    logging.info(f"GridSearchCV completed for {model_name} in {cv_time:.2f} seconds")
    logging.info(f"CV Results - Mean CV score: {search.best_score_:.4f}")
    
    # Evaluate best model on hold-out
    y_pred = search.predict(X_test_vec)
    # Training set performance for overfitting check
    y_pred_train = search.predict(X_train_vec)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    # Probabilities for ROC-AUC (only if estimator exposes predict_proba)
    try:
        y_score = search.best_estimator_.predict_proba(X_test_vec)
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        roc_auc = roc_auc_score(y_test_binarized, y_score, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float('nan')
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Overfitting check: compare train vs test F1
    overfit_threshold = config.get('overfit_threshold', 0.05)
    if train_f1 - f1 > overfit_threshold:
        logging.warning(
            f"⚠️  Potential overfitting for {model_name}: train_f1={train_f1:.4f} vs test_f1={f1:.4f} "
            f"(Δ={(train_f1 - f1):.4f} > {overfit_threshold})"
        )
    
    # Per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join('results', 'reports', f"{model_name}.json"), 'w') as f_report:
        json.dump(report, f_report, indent=2)
    
    total_time = time.time() - start_time
    
    # Log detailed results
    logging.info(f"{model_name} - Best F1: {f1:.4f}, Best params: {search.best_params_}")
    logging.info(f"{model_name} - Total training time: {total_time:.2f} seconds")
    
    # ---------- External / blind-set evaluation ----------
    f1_ext = float('nan')
    if ext_data:
        try:
            X_ext_vec = vectorizer.transform(ext_data['X_ext'])
            y_ext_pred = search.predict(X_ext_vec)
            _, _, f1_ext, _ = precision_recall_fscore_support(ext_data['y_ext'], y_ext_pred, average='macro')
            logging.info(f"{model_name} – external F1: {f1_ext:.4f}")
        except Exception as e:
            logging.warning(f"{model_name}: external validation failed – {e}")
    
    # Calculate per-class F1 scores
    class_f1_scores = precision_recall_fscore_support(y_test, y_pred, average=None)[2]
    class_f1_dict = {i: f1 for i, f1 in enumerate(class_f1_scores)}
    
    # Add class F1 scores to the results
    result = {
        'model': search.best_estimator_,
        'vectorizer': vectorizer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_params': search.best_params_,
        'cv_time': cv_time,
        'total_time': total_time,
        'f1_ext': f1_ext,
        'class_f1': class_f1_dict
    }
    
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
        
        all_misclassified[model_name] = {
            "model_type": "traditional_ml",
            "misclassified_examples": misclassified_examples
        }
        
        with open(misclassified_path, 'w') as f:
            json.dump(all_misclassified, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save misclassified examples for {model_name}: {e}")
    
    return result

def train_and_evaluate_models(data_dict, config, ext_data=None):
    """Train and evaluate all traditional ML models"""
    logging.info("Starting traditional ML models...")
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    # Get models to train
    models_to_train = config.get('traditional_ml', {}).get('models', [])
    
    successful_models = []
    failed_models = []
    
    # Results file path
    results_path = os.path.join('results', 'results.csv')
    
    for model_name in models_to_train:
        try:
            logging.info(f"Starting {model_name}")
            
            result = train_single_model(
                model_name, X_train, y_train, X_test, y_test, config, ext_data
            )
            
            if result is None:
                failed_models.append(model_name)
                continue
            
            # Save results to CSV
            with open(results_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1']:.4f}",
                    f"{result['roc_auc']:.4f}",
                    str(result['best_params']),
                    'success',
                    f"{result['f1_ext']:.4f}",
                    f"{result['class_f1']}"
                ])
            
            # Save model
            model_dir = os.path.join('models', model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(result['model'], os.path.join(model_dir, 'model.pkl'))
            joblib.dump(result['vectorizer'], os.path.join(model_dir, 'vectorizer.pkl'))
            
            successful_models.append(model_name)
            logging.info(f"{model_name} completed successfully")
            
        except Exception as e:
            logging.error(f"{model_name} failed: {e}")
            failed_models.append(model_name)
            
            # Log failure to CSV
            with open(results_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    '0',
                    '0',
                    '0',
                    '0',
                    '{}',
                    f'error: {str(e)}',
                    'nan'
                ])
    
    # Summary
    logging.info(f"Training Summary: {len(successful_models)} successful, {len(failed_models)} failed")
    if failed_models:
        logging.warning(f"Failed: {', '.join(failed_models)}")
    
    return successful_models, failed_models 