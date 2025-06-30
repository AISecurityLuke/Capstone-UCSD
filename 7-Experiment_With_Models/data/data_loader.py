#!/usr/bin/env python3
"""
Enhanced data loader with configurable preprocessing pipeline
"""

import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    # Load raw data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for item in data:
        if isinstance(item, dict) and 'user_message' in item and 'classification' in item:
            text = item['user_message']
            label = str(item['classification'])
            
            # Apply text cleaning based on config
            if config.get('preprocessing', {}).get('text_cleaning', {}).get('remove_html', True):
                text = sanitize_text(text)
            
            if config.get('preprocessing', {}).get('text_cleaning', {}).get('normalize_whitespace', True):
                text = ' '.join(text.split())
            
            texts.append(text)
            labels.append(label)
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    logging.info(f"Dataset loaded: {len(texts)} samples")
    logging.info(f"Classes: {list(le.classes_)}")
    
    # Log class distribution
    unique, counts = np.unique(labels_encoded, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        logging.info(f"Class {cls} ({le.inverse_transform([cls])[0]}): {count} samples")
    
    # Split data
    test_size = config.get('test_size', 0.2)
    random_seed = config.get('random_seed', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels_encoded, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=labels_encoded
    )
    
    # Compute class weights for imbalanced datasets
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
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