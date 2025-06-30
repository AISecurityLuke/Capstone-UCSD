"""
Training package for ML models
"""

from .trainer import train_and_evaluate_models
from .adaptive_trainer import train_and_evaluate_deep_models

__all__ = ['train_and_evaluate_models', 'train_and_evaluate_deep_models'] 