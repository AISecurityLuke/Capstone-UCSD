#!/usr/bin/env python3
"""
Adversarial validation strategies for testing model robustness
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple
import re
import random

def create_adversarial_examples(texts: List[str], labels: List[str]) -> List[Dict]:
    """
    Create adversarial examples to test model robustness
    """
    adversarial_examples = []
    
    for text, label in zip(texts, labels):
        # 1. Text perturbation (typos, case changes)
        perturbed_text = text
        if random.random() < 0.3:
            # Add typos
            words = perturbed_text.split()
            if len(words) > 3:
                typo_idx = random.randint(0, len(words) - 1)
                word = words[typo_idx]
                if len(word) > 3:
                    char_idx = random.randint(1, len(word) - 2)
                    word = word[:char_idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[char_idx + 1:]
                    words[typo_idx] = word
                    perturbed_text = ' '.join(words)
        
        # 2. Case manipulation
        if random.random() < 0.2:
            perturbed_text = perturbed_text.upper()
        elif random.random() < 0.2:
            perturbed_text = perturbed_text.lower()
        
        # 3. Add noise (extra spaces, punctuation)
        if random.random() < 0.2:
            perturbed_text = perturbed_text.replace(' ', '  ')  # Double spaces
        
        adversarial_examples.append({
            'original_text': text,
            'adversarial_text': perturbed_text,
            'label': label,
            'perturbation_type': 'text_manipulation'
        })
    
    return adversarial_examples

def create_domain_shift_examples(main_data_path: str, external_data_path: str) -> Dict:
    """
    Create examples that simulate domain shift
    """
    with open(main_data_path, 'r') as f:
        main_data = json.load(f)
    
    with open(external_data_path, 'r') as f:
        external_data = json.load(f)
    
    # Analyze characteristics of external data
    external_texts = [item['user_message'] for item in external_data]
    
    # Find main data examples that are most similar to external data style
    domain_shift_examples = []
    
    for item in main_data:
        text = item['user_message']
        
        # Check if text has external data characteristics
        has_multilingual = bool(re.search(r'[áéíóúñ¿¡]', text))
        has_casual_language = any(word in text.lower() for word in ['bro', 'fam', 'yo', 'dude'])
        has_emojis = bool(re.search(r'[😀-🙏🌀-🗿]', text))
        has_code = any(keyword in text for keyword in ['function', 'class', 'import', 'def '])
        
        if has_multilingual or has_casual_language or has_emojis or has_code:
            domain_shift_examples.append({
                'text': text,
                'label': item['classification'],
                'shift_characteristics': {
                    'multilingual': has_multilingual,
                    'casual': has_casual_language,
                    'emojis': has_emojis,
                    'code': has_code
                }
            })
    
    return {
        'domain_shift_examples': domain_shift_examples,
        'external_style_characteristics': {
            'multilingual_ratio': sum(1 for t in external_texts if re.search(r'[áéíóúñ¿¡]', t)) / len(external_texts),
            'casual_ratio': sum(1 for t in external_texts if any(word in t.lower() for word in ['bro', 'fam', 'yo', 'dude'])) / len(external_texts),
            'emoji_ratio': sum(1 for t in external_texts if re.search(r'[😀-🙏🌀-🗿]', t)) / len(external_texts)
        }
    }

def evaluate_adversarial_robustness(model, adversarial_examples: List[Dict], config: Dict) -> Dict:
    """
    Evaluate model performance on adversarial examples
    """
    results = {
        'adversarial_accuracy': 0,
        'perturbation_analysis': {},
        'robustness_score': 0
    }
    
    correct_predictions = 0
    perturbation_types = {}
    
    for example in adversarial_examples:
        # Get model prediction on adversarial example
        # This would need to be implemented based on your model type
        predicted_label = "0"  # Placeholder
        
        if predicted_label == example['label']:
            correct_predictions += 1
        
        # Track performance by perturbation type
        p_type = example['perturbation_type']
        if p_type not in perturbation_types:
            perturbation_types[p_type] = {'correct': 0, 'total': 0}
        
        perturbation_types[p_type]['total'] += 1
        if predicted_label == example['label']:
            perturbation_types[p_type]['correct'] += 1
    
    results['adversarial_accuracy'] = correct_predictions / len(adversarial_examples)
    results['perturbation_analysis'] = {
        p_type: {
            'accuracy': data['correct'] / data['total'],
            'count': data['total']
        } for p_type, data in perturbation_types.items()
    }
    
    # Calculate overall robustness score
    results['robustness_score'] = results['adversarial_accuracy']
    
    return results 