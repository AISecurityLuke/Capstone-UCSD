#!/usr/bin/env python3
"""
Enhanced data loader with configurable preprocessing pipeline
"""

import json
import logging
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from utils.sanitizer import sanitize_text

def load_data(config):
    """
    Load and preprocess data with enhanced configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing processed data and metadata
    """
    data_path = config['data_path']
    print("[data_loader.py] Loading data from:", data_path)
    
    # Load raw data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = []
    labels = []
    seen_hashes = set()
    
    for item in data:
        if isinstance(item, dict) and 'user_message' in item and 'classification' in item:
            text_raw = item['user_message']
            label = str(item['classification'])
            
            # Apply text cleaning based on config
            if config.get('preprocessing', {}).get('text_cleaning', {}).get('remove_html', True):
                text_clean = sanitize_text(text_raw)
            else:
                text_clean = text_raw
            
            if config.get('preprocessing', {}).get('text_cleaning', {}).get('normalize_whitespace', True):
                text_clean = ' '.join(text_clean.split())
            
            # Use a simple hash to detect duplicates (case-insensitive)
            content_hash = text_clean.strip().lower()
            if content_hash in seen_hashes:
                continue  # Skip duplicate
            seen_hashes.add(content_hash)
            
            texts.append(text_clean)
            labels.append(label)
    
    if len(seen_hashes) < len(data):
        logging.info(f"Removed {len(data) - len(seen_hashes)} duplicate prompts (exact text match, case-insensitive)")
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    logging.info(f"Dataset loaded: {len(texts)} samples")
    logging.info(f"Classes: {list(le.classes_)}")
    
    # Log class distribution
    unique, counts = np.unique(labels_encoded, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        logging.info(f"Class {cls} ({le.inverse_transform([cls])[0]}): {count} samples")
    
    # ------------------------------------------------------------------
    # Group-aware split: ensure prompts with highly similar text stay in
    # the same split to avoid information leakage.
    # ------------------------------------------------------------------
    def fingerprint(t: str) -> str:
        return re.sub(r"[^a-z0-9]", "", t.lower())

    groups = np.array([fingerprint(t) for t in texts])

    test_size = config.get('test_size', 0.2)
    random_seed = config.get('random_seed', 42)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(texts, labels_encoded, groups))

    X_train = [texts[i] for i in train_idx]
    y_train = labels_encoded[train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = labels_encoded[test_idx]
    
    logging.info(
        f"Group-aware split: {len(X_train)} train prompts, {len(X_test)} test prompts "
        f"across {len(np.unique(groups))} unique text groups"
    )
    
    # --------------------------------------------------------------
    # Compute / assign class weights
    # --------------------------------------------------------------
    use_custom_weights = config.get('use_class_weights', False)
    custom_weights = config.get('class_weights', [])
    n_classes = len(np.unique(y_train))

    if use_custom_weights:
        if len(custom_weights) != n_classes:
            raise ValueError(
                f"class_weights list length ({len(custom_weights)}) does not match number of classes ({n_classes})."
            )
        class_weights = np.array(custom_weights)
        logging.info(f"Using custom class weights from config (order follows label_encoder.classes_): {class_weights}")
    else:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(n_classes), y=y_train)
        logging.info(f"Using automatically computed class weights: {class_weights}")

    class_weight_dict = {cls: class_weights[idx] for idx, cls in enumerate(np.arange(n_classes))}
    
    logging.info(f"Class weights: {class_weight_dict}")
    logging.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': le,
        'num_classes': len(le.classes_),
        'class_weight_dict': class_weight_dict,
        'class_names': list(le.classes_)
    } 