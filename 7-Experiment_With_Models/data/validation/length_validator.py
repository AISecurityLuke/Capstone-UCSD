from typing import Dict, List, Any
import logging
import numpy as np

class LengthValidator:
    """Validates overall distribution of text lengths in dataset."""

    def __init__(self, validation_config: Dict):
        cfg = validation_config.get('length_validation', {})
        # Acceptable ratio of very short or very long documents
        self.short_threshold = cfg.get('short_threshold', 5)  # characters
        self.long_threshold = cfg.get('long_threshold', 1000)  # characters
        self.max_short_ratio = cfg.get('max_short_ratio', 0.05)
        self.max_long_ratio = cfg.get('max_long_ratio', 0.05)

    def validate_length(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate proportion of too-short / too-long messages."""
        lengths = np.array([len(item.get('user_message', '')) for item in data])
        if len(lengths) == 0:
            return {"valid": False, "issues": ["No text samples found"], "metrics": {}}

        short_ratio = np.mean(lengths < self.short_threshold)
        long_ratio = np.mean(lengths > self.long_threshold)

        issues = []
        if short_ratio > self.max_short_ratio:
            issues.append(
                f"High short-text ratio: {short_ratio:.3f} > {self.max_short_ratio} (threshold)"
            )
        if long_ratio > self.max_long_ratio:
            issues.append(
                f"High long-text ratio: {long_ratio:.3f} > {self.max_long_ratio} (threshold)"
            )

        metrics = {
            "avg_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "short_ratio": float(short_ratio),
            "long_ratio": float(long_ratio),
        }

        return {"valid": len(issues) == 0, "issues": issues, "metrics": metrics} 