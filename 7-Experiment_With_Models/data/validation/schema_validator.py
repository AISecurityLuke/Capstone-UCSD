"""
Schema validation for training data
"""

import logging
from typing import Dict, List, Any

class SchemaValidator:
    """Validates data schema and structure"""
    
    def __init__(self, validation_config: Dict):
        self.config = validation_config.get('text_validation', {})
        
    def validate_schema(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate data schema and structure"""
        errors = []
        warnings = []
        
        required_fields = self.config.get('required_fields', ['user_message', 'classification'])
        allowed_classifications = self.config.get('allowed_classifications', [])
        min_length = self.config.get('min_length', 5)
        max_length = self.config.get('max_length', 1000)
        
        for i, item in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i}: Missing required field '{field}'")
                    break
            
            # Check text length
            if 'user_message' in item:
                text = item['user_message']
                if len(text) < min_length:
                    warnings.append(f"Item {i}: Text too short ({len(text)} chars, min {min_length})")
                elif len(text) > max_length:
                    warnings.append(f"Item {i}: Text too long ({len(text)} chars, max {max_length})")
            
            # Check classification values
            if 'classification' in item:
                classification = str(item['classification'])
                if allowed_classifications and classification not in allowed_classifications:
                    errors.append(f"Item {i}: Invalid classification '{classification}'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_items": len(data)
        } 