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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb
import joblib
import csv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from utils.sanitizer import escape_for_logging

def get_model_configs():
    """Get model configurations for hyperparameter tuning"""
    return {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
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

def train_single_model(model_name, X_train, y_train, X_test, y_test, config):
    """Train a single traditional ML model with comprehensive tuning"""
    logging.info(f"Training {model_name} with {len(X_train)} samples")
    
    # Get model configurations
    model_configs = get_model_configs()
    
    if model_name not in model_configs:
        logging.error(f"Unknown model: {model_name}")
        return None
    
    # Vectorize text data
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Get model and parameters
    model_config = model_configs[model_name]
    model = model_config['model']
    param_grid = model_config['params']
    
    # Hyperparameter tuning
    cv_folds = config.get('traditional_ml', {}).get('cv_folds', 5)
    n_jobs = config.get('traditional_ml', {}).get('n_jobs', -1)
    
    search = GridSearchCV(
        model, param_grid, cv=cv_folds, scoring='f1_macro',
        n_jobs=n_jobs, verbose=0
    )
    
    search.fit(X_train_vec, y_train)
    
    # Evaluate best model
    y_pred = search.predict(X_test_vec)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log detailed results
    logging.info(f"{model_name} - Best F1: {f1:.4f}, Best params: {search.best_params_}")
    
    return {
        'model': search.best_estimator_,
        'vectorizer': vectorizer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'best_params': search.best_params_
    }

def train_and_evaluate_models(data_dict, config):
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
                model_name, X_train, y_train, X_test, y_test, config
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
                    str(result['best_params']),
                    'success'
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
                    '{}',
                    f'error: {str(e)}'
                ])
    
    # Summary
    logging.info(f"Training Summary: {len(successful_models)} successful, {len(failed_models)} failed")
    if failed_models:
        logging.warning(f"Failed: {', '.join(failed_models)}")
    
    return successful_models, failed_models 