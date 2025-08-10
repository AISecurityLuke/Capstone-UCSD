#!/usr/bin/env python3
"""
Comprehensive validation strategies for testing model generalization
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Tuple
import os

from data.domain_validator import create_domain_splits, evaluate_domain_generalization
from data.temporal_validator import create_temporal_splits, evaluate_temporal_generalization
from data.adversarial_validator import create_adversarial_examples, create_domain_shift_examples, evaluate_adversarial_robustness

def run_comprehensive_validation(data_dict: Dict, config: Dict, model, model_name: str) -> Dict:
    """
    Run all validation strategies and return comprehensive results
    """
    logging.info(f"Running comprehensive validation for {model_name}")
    
    results = {
        'model_name': model_name,
        'validation_strategies': {}
    }
    
    # 1. Domain-based validation
    try:
        logging.info("Running domain-based validation...")
        domain_splits = create_domain_splits(config['data_path'])
        domain_results = evaluate_domain_generalization(model, data_dict, domain_splits, config)
        results['validation_strategies']['domain_based'] = domain_results
    except Exception as e:
        logging.error(f"Domain validation failed: {e}")
        results['validation_strategies']['domain_based'] = {'error': str(e)}
    
    # 2. Temporal validation
    try:
        logging.info("Running temporal validation...")
        temporal_splits = create_temporal_splits(config['data_path'])
        temporal_results = evaluate_temporal_generalization(model, data_dict, temporal_splits, config)
        results['validation_strategies']['temporal'] = temporal_results
    except Exception as e:
        logging.error(f"Temporal validation failed: {e}")
        results['validation_strategies']['temporal'] = {'error': str(e)}
    
    # 3. Adversarial validation
    try:
        logging.info("Running adversarial validation...")
        adversarial_examples = create_adversarial_examples(data_dict['X_test'], data_dict['y_test'])
        adversarial_results = evaluate_adversarial_robustness(model, adversarial_examples, config)
        results['validation_strategies']['adversarial'] = adversarial_results
    except Exception as e:
        logging.error(f"Adversarial validation failed: {e}")
        results['validation_strategies']['adversarial'] = {'error': str(e)}
    
    # 4. Domain shift analysis
    try:
        logging.info("Running domain shift analysis...")
        external_data_path = 'validation_jsons/outside_test_data.json'
        if os.path.exists(external_data_path):
            domain_shift_results = create_domain_shift_examples(config['data_path'], external_data_path)
            results['validation_strategies']['domain_shift'] = domain_shift_results
    except Exception as e:
        logging.error(f"Domain shift analysis failed: {e}")
        results['validation_strategies']['domain_shift'] = {'error': str(e)}
    
    # 5. Calculate overall generalization score
    results['generalization_score'] = calculate_generalization_score(results)
    
    return results

def calculate_generalization_score(validation_results: Dict) -> float:
    """
    Calculate an overall generalization score from all validation strategies
    """
    scores = []
    
    # Extract scores from different validation strategies
    if 'domain_based' in validation_results['validation_strategies']:
        domain_results = validation_results['validation_strategies']['domain_based']
        # Add domain-based scores if available
        pass
    
    if 'temporal' in validation_results['validation_strategies']:
        temporal_results = validation_results['validation_strategies']['temporal']
        # Add temporal scores if available
        pass
    
    if 'adversarial' in validation_results['validation_strategies']:
        adversarial_results = validation_results['validation_strategies']['adversarial']
        if 'robustness_score' in adversarial_results:
            scores.append(adversarial_results['robustness_score'])
    
    # Calculate average score
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0

def save_validation_results(results: Dict, output_path: str = 'results/validation_results.json'):
    """
    Save comprehensive validation results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Validation results saved to {output_path}")

def create_validation_report(results: Dict, output_path: str = 'results/validation_report.md'):
    """
    Create a comprehensive validation report
    """
    report = f"""# Model Validation Report: {results['model_name']}

## Overview
This report summarizes the model's performance across multiple validation strategies.

## Generalization Score
**Overall Score: {results.get('generalization_score', 'N/A'):.4f}**

## Validation Strategies

### 1. Domain-Based Validation
"""
    
    if 'domain_based' in results['validation_strategies']:
        domain_results = results['validation_strategies']['domain_based']
        for split_name, split_data in domain_results.items():
            report += f"- **{split_name}**: Train={split_data.get('train_size', 'N/A')}, Test={split_data.get('test_size', 'N/A')}\n"
    
    report += """
### 2. Temporal Validation
"""
    
    if 'temporal' in results['validation_strategies']:
        temporal_results = results['validation_strategies']['temporal']
        for split_name, split_data in temporal_results.items():
            report += f"- **{split_name}**: {split_data.get('description', 'N/A')}\n"
    
    report += """
### 3. Adversarial Validation
"""
    
    if 'adversarial' in results['validation_strategies']:
        adv_results = results['validation_strategies']['adversarial']
        report += f"- **Robustness Score**: {adv_results.get('robustness_score', 'N/A'):.4f}\n"
        report += f"- **Adversarial Accuracy**: {adv_results.get('adversarial_accuracy', 'N/A'):.4f}\n"
    
    report += """
### 4. Domain Shift Analysis
"""
    
    if 'domain_shift' in results['validation_strategies']:
        shift_results = results['validation_strategies']['domain_shift']
        if 'external_style_characteristics' in shift_results:
            chars = shift_results['external_style_characteristics']
            report += f"- **Multilingual Ratio**: {chars.get('multilingual_ratio', 'N/A'):.3f}\n"
            report += f"- **Casual Language Ratio**: {chars.get('casual_ratio', 'N/A'):.3f}\n"
            report += f"- **Emoji Ratio**: {chars.get('emoji_ratio', 'N/A'):.3f}\n"
    
    report += """
## Recommendations

Based on the validation results, consider the following:

1. **Domain Generalization**: If domain-based validation shows poor performance, consider data augmentation or domain adaptation techniques.

2. **Temporal Stability**: If temporal validation shows degradation, the model may be overfitting to temporal patterns.

3. **Adversarial Robustness**: If adversarial validation shows poor performance, consider adversarial training or input preprocessing.

4. **Domain Shift**: If domain shift analysis shows significant differences, consider collecting more diverse training data.
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logging.info(f"Validation report saved to {output_path}") 