#!/usr/bin/env python3
"""
Temporal validation strategies for testing model generalization over time
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import random

def create_temporal_splits(data_path: str, time_ratio: float = 0.8) -> Dict[str, List[int]]:
    """
    Create temporal train/test splits to simulate real-world deployment
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # If no timestamps, simulate temporal ordering
    # In real scenarios, you'd use actual timestamps
    n_samples = len(data)
    train_size = int(n_samples * time_ratio)
    
    # Simulate temporal ordering (earlier = train, later = test)
    temporal_splits = {
        'temporal_forward': {
            'train_indices': list(range(train_size)),
            'test_indices': list(range(train_size, n_samples)),
            'description': 'Train on earlier data, test on later data'
        },
        'temporal_reverse': {
            'train_indices': list(range(n_samples - train_size, n_samples)),
            'test_indices': list(range(n_samples - train_size)),
            'description': 'Train on later data, test on earlier data'
        }
    }
    
    return temporal_splits

def create_rolling_window_splits(data_path: str, window_size: float = 0.6, step_size: float = 0.2) -> List[Dict]:
    """
    Create rolling window validation splits
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    n_samples = len(data)
    window_samples = int(n_samples * window_size)
    step_samples = int(n_samples * step_size)
    
    splits = []
    
    for start in range(0, n_samples - window_samples, step_samples):
        end = start + window_samples
        test_start = end
        test_end = min(end + step_samples, n_samples)
        
        if test_end > test_start:
            splits.append({
                'train_indices': list(range(start, end)),
                'test_indices': list(range(test_start, test_end)),
                'window': f"{start}-{end}",
                'test_window': f"{test_start}-{test_end}"
            })
    
    return splits

def evaluate_temporal_generalization(model, data_dict: Dict, temporal_splits: Dict, config: Dict) -> Dict:
    """
    Evaluate model performance across temporal splits
    """
    results = {}
    
    for split_name, split_data in temporal_splits.items():
        logging.info(f"Evaluating temporal split: {split_name}")
        
        # Extract data for this temporal split
        X_train = [data_dict['X_train'][i] for i in split_data['train_indices']]
        y_train = [data_dict['y_train'][i] for i in split_data['train_indices']]
        X_test = [data_dict['X_test'][i] for i in split_data['test_indices']]
        y_test = [data_dict['y_test'][i] for i in split_data['test_indices']]
        
        if len(X_train) > 0 and len(X_test) > 0:
            # Train and evaluate model on this temporal split
            results[split_name] = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'description': split_data.get('description', ''),
                'temporal_ordering': split_data
            }
    
    return results 