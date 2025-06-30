#!/usr/bin/env python3
"""
Production-grade experiment runner with MLflow, W&B, and comprehensive tracking
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import time
import csv
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from data.data_loader import load_data
from training.trainer import train_and_evaluate_models
from training.adaptive_trainer import train_and_evaluate_deep_models
from data.validation import SchemaValidator, QualityValidator
from utils.sanitizer import escape_for_logging
from utils.monitor import monitor_resources

# Import MLflow and W&B if available
try:
    import mlflow
    import mlflow.keras
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Install with: pip install wandb")

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    experiment_name: str = "prompt_classification"
    enable_mlflow: bool = True
    enable_wandb: bool = False
    enable_ray: bool = False
    tracking_uri: str = "sqlite:///mlruns.db"
    model_registry_name: str = "prompt_classifier"

class ProductionExperimentRunner:
    """Production-grade experiment runner with comprehensive tracking"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.current_run = None
        self.model_registry = {}
        self.setup_tracking()
        
    def setup_tracking(self):
        """Setup experiment tracking systems"""
        # Setup MLflow
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            
        # Setup Weights & Biases
        if self.config.enable_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="prompt-classification",
                name=self.config.experiment_name,
                config=asdict(self.config)
            )
            
    def start_experiment(self, run_name: str, tags: Dict[str, str] = None):
        """Start a new experiment run"""
        tags = tags or {}
        
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
            
        if self.config.enable_wandb and WANDB_AVAILABLE:
            wandb.run.name = run_name
            if tags:
                wandb.run.tags = list(tags.values())
                
        logging.info(f"Started experiment: {run_name}")
        
    def end_experiment(self):
        """End current experiment run"""
        if self.config.enable_mlflow and self.current_run and MLFLOW_AVAILABLE:
            mlflow.end_run()
            
        if self.config.enable_wandb and WANDB_AVAILABLE:
            wandb.finish()
            
        logging.info("Experiment completed")
        
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to tracking systems"""
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_params(params)
            
        if self.config.enable_wandb and WANDB_AVAILABLE:
            wandb.config.update(params)
            
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking systems"""
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics)
            
        if self.config.enable_wandb and WANDB_AVAILABLE:
            wandb.log(metrics)
            
    def log_model(self, model, model_name: str):
        """Log model to tracking systems"""
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            if hasattr(model, 'predict_proba'):  # sklearn model
                mlflow.sklearn.log_model(model, model_name)
            else:  # keras model
                mlflow.keras.log_model(model, model_name)
                
        # Store in local registry
        self.model_registry[model_name] = {
            'model': model,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'type': 'sklearn' if hasattr(model, 'predict_proba') else 'keras'
        }
        
    def export_results(self, output_path: str = "experiment_results.json"):
        """Export all experiment results"""
        results = {
            "experiment_config": asdict(self.config),
            "model_registry": {k: {'timestamp': v['timestamp'], 'type': v['type']} 
                             for k, v in self.model_registry.items()},
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logging.info(f"Results exported to: {output_path}")
        return results

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/experiment.log'),
            logging.StreamHandler()
        ]
    )

def create_clean_results_csv():
    """Create a clean results CSV file"""
    results_path = os.path.join('results', 'results.csv')
    os.makedirs('results', exist_ok=True)
    
    with open(results_path, mode='w', newline='') as f:
        f.write('model,precision_macro,recall_macro,f1_macro,best_params,status,experiment_id\n')
    
    logging.info("Created clean results CSV: results/results.csv")

def validate_data_pipeline(data_dict: Dict, config: Dict) -> bool:
    """Validate data before training"""
    logging.info("Validating training data...")
    
    # Initialize validators
    schema_validator = SchemaValidator(config['data_validation']['validation_config'])
    quality_validator = QualityValidator(config['data_validation']['validation_config'])
    
    # Convert data back to original format for validation
    original_data = []
    for i, text in enumerate(data_dict['X_train']):
        original_data.append({
            'user_message': text,
            'classification': data_dict['label_encoder'].inverse_transform([data_dict['y_train'][i]])[0]
        })
    
    # Validate schema
    schema_results = schema_validator.validate_schema(original_data)
    if not schema_results['valid']:
        logging.error("Schema validation failed - stopping experiment")
        return False
        
    # Validate quality
    quality_results = quality_validator.validate_quality(original_data)
    if not quality_results['valid']:
        logging.error("Quality validation failed - stopping experiment")
        return False
        
    # Log validation results
    logging.info("Data validation passed")
    if schema_results['warnings']:
        logging.warning(f"Schema warnings: {len(schema_results['warnings'])}")
    if quality_results['issues']:
        logging.warning(f"Quality issues: {len(quality_results['issues'])}")
            
    return True

@monitor_resources
def main():
    """Main experiment pipeline"""
    # Setup
    setup_logging()
    create_clean_results_csv()
    
    # Load configuration
    config = load_config()
    logging.info("Configuration loaded")
    
    # Initialize production experiment runner
    exp_config = ExperimentConfig(
        experiment_name=config['experiment_tracking']['experiment_name'],
        enable_mlflow=config['experiment_tracking']['enable_mlflow'],
        enable_wandb=config['experiment_tracking']['enable_wandb'],
        enable_ray=config['experiment_tracking']['enable_ray'],
        tracking_uri=config['experiment_tracking']['tracking_uri']
    )
    
    experiment_runner = ProductionExperimentRunner(exp_config)
    
    # Start experiment
    run_name = f"experiment_{int(time.time())}"
    experiment_runner.start_experiment(run_name)
    
    # Load data
    data_dict = load_data(config)
    logging.info(f"Data loaded: {len(data_dict['X_train'])} train, {len(data_dict['X_test'])} test samples")
    
    # Validate data
    if config['data_validation']['enable_validation']:
        if not validate_data_pipeline(data_dict, config):
            experiment_runner.end_experiment()
            return
    
    # Log data statistics
    experiment_runner.log_metrics({
        "data/train_samples": len(data_dict['X_train']),
        "data/test_samples": len(data_dict['X_test']),
        "data/num_classes": data_dict['num_classes'],
    })
    
    # Log class weights as separate metrics
    for class_id, weight in data_dict['class_weight_dict'].items():
        experiment_runner.log_metrics({f"data/class_weight_{class_id}": float(weight)})
    
    # Train traditional ML models
    logging.info("Starting traditional ML models...")
    trad_success, trad_failed = train_and_evaluate_models(data_dict, config)
    
    # Train deep learning models if enabled
    dl_success, dl_failed = [], []
    if config.get('enable_dl', True):
        logging.info("Starting deep learning models...")
        dl_success, dl_failed = train_and_evaluate_deep_models(data_dict, config)
    
    # Summary
    total_success = len(trad_success) + len(dl_success)
    total_failed = len(trad_failed) + len(dl_failed)
    
    logging.info("Experiment completed!")
    logging.info("Final Results:")
    logging.info(f"   Successful: {total_success} models")
    logging.info(f"   Failed: {total_failed} models")
    
    if trad_success:
        logging.info(f"   Traditional ML: {', '.join(trad_success)}")
    if dl_success:
        logging.info(f"   Deep Learning: {', '.join(dl_success)}")
    if trad_failed or dl_failed:
        logging.warning(f"   Failed models: {', '.join(trad_failed + dl_failed)}")
    
    # End experiment
    experiment_runner.end_experiment()
    
    # Export results
    experiment_runner.export_results("experiment_results.json")
    
    logging.info("Production experiment pipeline completed successfully!")

    # Generate visualizations
    import subprocess
    try:
        logging.info("Generating visualizations with results/visual.py ...")
        subprocess.run(["python", "results/visual.py"], check=True)
        logging.info("Visualizations generated successfully.")
    except Exception as e:
        logging.error(f"Visualization generation failed: {e}")

if __name__ == "__main__":
    main() 