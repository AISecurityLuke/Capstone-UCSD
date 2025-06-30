"""
Data quality validation
"""

import logging
from typing import Dict, List, Any
from collections import Counter

class QualityValidator:
    """Validates data quality metrics"""
    
    def __init__(self, validation_config: Dict):
        self.config = validation_config.get('data_quality', {})
        
    def validate_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate data quality metrics"""
        issues = []
        metrics = {}
        
        # Check for missing values
        missing_threshold = self.config.get('missing_threshold', 0.05)
        missing_count = sum(1 for item in data if not item.get('user_message') or not item.get('classification'))
        missing_ratio = missing_count / len(data) if data else 0
        
        metrics['missing_ratio'] = missing_ratio
        if missing_ratio > missing_threshold:
            issues.append(f"High missing value ratio: {missing_ratio:.3f} > {missing_threshold}")
        
        # Check for duplicates
        duplicate_threshold = self.config.get('duplicate_threshold', 0.1)
        texts = [item.get('user_message', '') for item in data]
        text_counts = Counter(texts)
        duplicate_count = sum(count - 1 for count in text_counts.values() if count > 1)
        duplicate_ratio = duplicate_count / len(data) if data else 0
        
        metrics['duplicate_ratio'] = duplicate_ratio
        if duplicate_ratio > duplicate_threshold:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.3f} > {duplicate_threshold}")
        
        # Check class balance
        class_balance_threshold = self.config.get('class_balance_threshold', 0.1)
        classifications = [str(item.get('classification', '')) for item in data]
        class_counts = Counter(classifications)
        
        if class_counts:
            total = sum(class_counts.values())
            class_ratios = {cls: count/total for cls, count in class_counts.items()}
            min_ratio = min(class_ratios.values())
            max_ratio = max(class_ratios.values())
            balance_ratio = min_ratio / max_ratio if max_ratio > 0 else 0
            
            metrics['class_balance_ratio'] = balance_ratio
            metrics['class_distribution'] = class_ratios
            
            if balance_ratio < class_balance_threshold:
                issues.append(f"Poor class balance: {balance_ratio:.3f} < {class_balance_threshold}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metrics": metrics
        } 