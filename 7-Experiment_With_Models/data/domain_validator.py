#!/usr/bin/env python3
"""
Domain-based validation strategies for testing model generalization
"""

import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import re

def analyze_text_domains(texts: List[str]) -> Dict[str, List[int]]:
    """
    Analyze texts and categorize them by domain characteristics
    """
    domains = {
        'technical': [],
        'casual': [],
        'multilingual': [],
        'code_heavy': [],
        'security_focused': [],
        'jailbreak_attempts': []
    }
    
    for i, text in enumerate(texts):
        text_lower = text.lower()
        
        # Technical domain
        if any(keyword in text_lower for keyword in ['algorithm', 'protocol', 'api', 'database', 'framework', 'architecture']):
            domains['technical'].append(i)
        
        # Casual domain  
        elif any(keyword in text_lower for keyword in ['bro', 'fam', 'yo', 'dude', 'hey', 'whats up']):
            domains['casual'].append(i)
        
        # Multilingual domain
        if re.search(r'[áéíóúñ¿¡]', text) or any(word in text_lower for word in ['como', 'que', 'por', 'para', 'con']):
            domains['multilingual'].append(i)
        
        # Code-heavy domain
        if any(keyword in text for keyword in ['function', 'class', 'import', 'def ', 'const ', 'var ', '{}', '()']):
            domains['code_heavy'].append(i)
        
        # Security-focused domain
        if any(keyword in text_lower for keyword in ['pentest', 'vulnerability', 'exploit', 'bypass', 'injection', 'xss', 'csrf']):
            domains['security_focused'].append(i)
        
        # Jailbreak attempts
        if any(keyword in text_lower for keyword in ['ignore', 'bypass', 'jailbreak', 'system prompt', 'override', 'ignore safety']):
            domains['jailbreak_attempts'].append(i)
    
    return domains

def create_domain_splits(data_path: str, test_size: float = 0.2) -> Dict[str, Dict]:
    """
    Create domain-based train/test splits
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = [item['user_message'] for item in data]
    labels = [item['classification'] for item in data]
    
    # Analyze domains
    domains = analyze_text_domains(texts)
    
    splits = {}
    
    # 1. Technical vs Non-Technical Split
    technical_indices = set(domains['technical'])
    non_technical_indices = set(range(len(texts))) - technical_indices
    
    if len(technical_indices) > 100 and len(non_technical_indices) > 100:
        splits['technical_vs_non'] = {
            'train_tech': list(technical_indices)[:int(len(technical_indices) * (1-test_size))],
            'test_tech': list(technical_indices)[int(len(technical_indices) * (1-test_size)):],
            'train_non': list(non_technical_indices)[:int(len(non_technical_indices) * (1-test_size))],
            'test_non': list(non_technical_indices)[int(len(non_technical_indices) * (1-test_size)):]
        }
    
    # 2. Language-based Split
    multilingual_indices = set(domains['multilingual'])
    english_indices = set(range(len(texts))) - multilingual_indices
    
    if len(multilingual_indices) > 50 and len(english_indices) > 100:
        splits['language_based'] = {
            'train_english': list(english_indices)[:int(len(english_indices) * (1-test_size))],
            'test_english': list(english_indices)[int(len(english_indices) * (1-test_size)):],
            'train_multilingual': list(multilingual_indices)[:int(len(multilingual_indices) * (1-test_size))],
            'test_multilingual': list(multilingual_indices)[int(len(multilingual_indices) * (1-test_size)):]
        }
    
    # 3. Complexity-based Split (code-heavy vs natural language)
    code_indices = set(domains['code_heavy'])
    natural_indices = set(range(len(texts))) - code_indices
    
    if len(code_indices) > 50 and len(natural_indices) > 100:
        splits['complexity_based'] = {
            'train_code': list(code_indices)[:int(len(code_indices) * (1-test_size))],
            'test_code': list(code_indices)[int(len(code_indices) * (1-test_size)):],
            'train_natural': list(natural_indices)[:int(len(natural_indices) * (1-test_size))],
            'test_natural': list(natural_indices)[int(len(natural_indices) * (1-test_size)):]
        }
    
    return splits

def evaluate_domain_generalization(model, data_dict: Dict, domain_splits: Dict, config: Dict) -> Dict:
    """
    Evaluate model performance across different domains
    """
    results = {}
    
    for split_name, split_indices in domain_splits.items():
        logging.info(f"Evaluating domain split: {split_name}")
        
        # Extract data for this split
        X_train = [data_dict['X_train'][i] for i in split_indices.get('train_tech', []) + split_indices.get('train_non', [])]
        y_train = [data_dict['y_train'][i] for i in split_indices.get('train_tech', []) + split_indices.get('train_non', [])]
        X_test = [data_dict['X_test'][i] for i in split_indices.get('test_tech', []) + split_indices.get('test_non', [])]
        y_test = [data_dict['y_test'][i] for i in split_indices.get('test_tech', []) + split_indices.get('test_non', [])]
        
        if len(X_train) > 0 and len(X_test) > 0:
            # Train and evaluate model on this domain split
            # This would need to be implemented based on your model type
            results[split_name] = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'domain_breakdown': split_indices
            }
    
    return results 