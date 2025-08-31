"""
Adaptive Weight Controller for dynamic class weight adjustment based on metric performance.
This module provides a unified weight adjustment mechanism for both Keras and HuggingFace pipelines.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import json
import os


class AdaptiveWeightController:
    """
    Controller that dynamically adjusts class weights/focal alphas based on metric performance.
    
    This controller monitors specified metrics and adjusts class weights when metrics
    are off-target for a specified patience period.
    """
    
    def __init__(self, config: Dict, num_classes: int = 3, model_name: str = "model"):
        """
        Initialize the controller with configuration.
        
        Args:
            config: Full configuration dictionary (expects 'adaptive_objectives' key)
            num_classes: Number of classes (default: 3)
            model_name: Model name for logging and trajectory saving
        """
        self.config = config.get('adaptive_objectives', {})
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize alpha with current class weights or default to 1.0
        initial_weights = config.get('class_weights', [1.0] * num_classes)
        self.alpha = np.array(initial_weights, dtype=np.float32)
        self.initial_alpha = self.alpha.copy()
        
        # History tracking
        self.metric_history = []
        self.alpha_history = []
        self.alpha_history.append({
            'epoch': 0,
            'alpha': self.alpha.copy().tolist(),
            'reason': 'initial'
        })
        
        # Extract configuration
        self.metric_weights = self.config.get('metric_weights', {})
        self.targets = self.config.get('targets', {})
        self.adaptation = self.config.get('adaptation', {})
        
        # Set defaults if not in config
        self.alpha_step = self.adaptation.get('alpha_step', 0.1)
        self.alpha_min = self.adaptation.get('alpha_min', 0.5)
        self.alpha_max = self.adaptation.get('alpha_max', 3.0)
        self.patience = self.adaptation.get('patience', 2)
        
        logging.info(f"[AdaptiveWeightController] Initialized for {model_name}")
        logging.info(f"[AdaptiveWeightController] Initial alpha: {self.alpha.tolist()}")
        logging.info(f"[AdaptiveWeightController] Targets: {self.targets}")
        
    def update(self, metrics: Dict[str, float], epoch: int) -> np.ndarray:
        """
        Update weights based on current metrics.
        
        Args:
            metrics: Dictionary of metric values (must include keys from targets)
            epoch: Current epoch number
            
        Returns:
            Updated alpha array
        """
        # Store metrics
        self.metric_history.append({
            'epoch': epoch,
            'metrics': metrics.copy()
        })
        
        # Check if we have enough history for patience
        if len(self.metric_history) < self.patience:
            return self.alpha.copy()
        
        # Analyze which metrics are off-target
        off_target_metrics = self._identify_off_target_metrics()
        
        if off_target_metrics:
            # Adjust alpha based on off-target metrics
            old_alpha = self.alpha.copy()
            self._adjust_alpha(off_target_metrics)
            
            # Log changes
            if not np.allclose(old_alpha, self.alpha):
                change_summary = {
                    'epoch': epoch,
                    'alpha': self.alpha.copy().tolist(),
                    'reason': f"Off-target metrics: {', '.join(off_target_metrics)}",
                    'old_alpha': old_alpha.tolist(),
                    'metrics': metrics
                }
                self.alpha_history.append(change_summary)
                
                logging.info(f"[AdaptiveWeightController] Epoch {epoch}: Alpha adjusted")
                logging.info(f"  Old alpha: {old_alpha.tolist()}")
                logging.info(f"  New alpha: {self.alpha.tolist()}")
                logging.info(f"  Reason: {change_summary['reason']}")
        
        return self.alpha.copy()
    
    def _identify_off_target_metrics(self) -> List[str]:
        """
        Identify metrics that have been off-target for the patience period.
        
        Returns:
            List of metric names that are off-target
        """
        off_target = []
        
        # Get recent history
        recent_history = self.metric_history[-self.patience:]
        
        for metric_name, target_value in self.targets.items():
            # Check if this metric has been off-target for all recent epochs
            values = [h['metrics'].get(metric_name, 0) for h in recent_history]
            
            # Different logic for different metric types
            if 'recall' in metric_name or 'f1' in metric_name:
                # For recall/F1: want value >= target
                if all(v < target_value for v in values):
                    off_target.append(metric_name)
            elif 'precision' in metric_name:
                # For precision: want value >= target
                if all(v < target_value for v in values):
                    off_target.append(metric_name)
            elif 'rate' in metric_name:
                # For error rates: want value <= target
                if all(v > target_value for v in values):
                    off_target.append(metric_name)
        
        return off_target
    
    def _adjust_alpha(self, off_target_metrics: List[str]):
        """
        Adjust alpha values based on which metrics are off-target.
        
        Args:
            off_target_metrics: List of metric names that are off-target
        """
        step = self.alpha_step
        
        # Class 2 (malicious) adjustments
        if 'recall_c2' in off_target_metrics or 'missed_threat_rate' in off_target_metrics:
            self.alpha[2] += step
            logging.debug(f"  Increasing alpha[2] by {step} (recall_c2 or missed_threat)")
        
        # Class 1 (suspicious) adjustments
        if 'precision_c1' in off_target_metrics:
            self.alpha[1] += step
            logging.debug(f"  Increasing alpha[1] by {step} (precision_c1)")
        
        # Class 0 (benign) adjustments
        if 'f1_c0' in off_target_metrics or 'false_alarm_rate' in off_target_metrics:
            self.alpha[0] += step
            logging.debug(f"  Increasing alpha[0] by {step} (f1_c0 or false_alarm)")
        
        # Clip to bounds
        self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)
        
        # Normalize to prevent alpha explosion (optional)
        # self.alpha = self.alpha / self.alpha.sum() * self.num_classes
    
    def get_weighted_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted score based on metric weights.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            Weighted score (before penalty)
        """
        score = 0.0
        for metric_name, weight in self.metric_weights.items():
            if metric_name != 'critical_penalty' and metric_name in metrics:
                score += weight * metrics[metric_name]
        return score
    
    def save_trajectory(self, output_dir: str = 'results'):
        """
        Save alpha trajectory to CSV file.
        
        Args:
            output_dir: Directory to save the trajectory file
        """
        os.makedirs(output_dir, exist_ok=True)
        trajectory_path = os.path.join(output_dir, f'alpha_trajectory_{self.model_name}.json')
        
        with open(trajectory_path, 'w') as f:
            json.dump({
                'model': self.model_name,
                'initial_alpha': self.initial_alpha.tolist(),
                'final_alpha': self.alpha.tolist(),
                'history': self.alpha_history,
                'metric_history': self.metric_history[-10:]  # Last 10 epochs
            }, f, indent=2)
        
        logging.info(f"[AdaptiveWeightController] Saved trajectory to {trajectory_path}")
    
    def get_final_alpha(self) -> List[float]:
        """
        Get the final alpha values as a list.
        
        Returns:
            Final alpha values
        """
        return self.alpha.tolist() 