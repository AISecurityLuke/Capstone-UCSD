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
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from data.data_loader import load_data
from training.trainer import train_and_evaluate_models
from training.adaptive_trainer import train_and_evaluate_deep_models
from training.transformer_trainer import train_and_evaluate_transformer_models
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
        # Setup MLflow with timeout
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("MLflow initialization timed out")
                
                # Set a 30-second timeout for MLflow setup
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    mlflow.set_tracking_uri(self.config.tracking_uri)
                    mlflow.set_experiment(self.config.experiment_name)
                    logging.info("MLflow tracking setup completed")
                finally:
                    signal.alarm(0)  # Cancel the alarm
                    
            except TimeoutError:
                logging.warning("MLflow initialization timed out, disabling MLflow tracking")
                self.config.enable_mlflow = False
            except Exception as e:
                logging.warning(f"MLflow setup failed: {e}, disabling MLflow tracking")
                self.config.enable_mlflow = False
            
        # Setup Weights & Biases
        if self.config.enable_wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="prompt-classification",
                    name=self.config.experiment_name,
                    config=asdict(self.config)
                )
            except Exception as e:
                logging.warning(f"W&B setup failed: {e}, disabling W&B tracking")
                self.config.enable_wandb = False
            
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
        # Added f1_external column for blind-set evaluation
        f.write('model,precision_macro,recall_macro,f1_macro,roc_auc,best_params,status,f1_external,per_class_f1\n')
    
    logging.info("Created clean results CSV: results/results.csv")

def validate_data_pipeline(data_dict: Dict, config: Dict) -> bool:
    """Validate data before training"""
    logging.info("Validating training data...")
    
    try:
        # Determine which validators to run based on config
        selected_validators = config.get('data_validation', {}).get('validators', [
            'schema', 'quality']
        )

        validator_instances = []
        for name in selected_validators:
            name = name.lower()
            if name == 'schema':
                validator_instances.append(
                    ('schema', SchemaValidator(config['data_validation']['validation_config']))
                )
            elif name == 'quality':
                validator_instances.append(
                    ('quality', QualityValidator(config['data_validation']['validation_config']))
                )
            elif name == 'length':
                from data.validation import LengthValidator
                validator_instances.append(
                    ('length', LengthValidator(config['data_validation']['validation_config']))
                )
            else:
                logging.warning(f"Unknown validator '{name}' requested; skipping")
        
        # Convert data back to original format for validation
        original_data = []
        total_samples = len(data_dict['X_train']) + len(data_dict['X_test'])
        logging.info(f"Converting {total_samples} total samples (train+test) for validation...")
        
        full_texts = list(data_dict['X_train']) + list(data_dict['X_test'])
        full_labels = list(data_dict['y_train']) + list(data_dict['y_test'])

        for i, text in enumerate(full_texts):
            try:
                label = full_labels[i]
                if hasattr(data_dict['label_encoder'], 'inverse_transform'):
                    original_label = data_dict['label_encoder'].inverse_transform([label])[0]
                else:
                    original_label = str(label)
                    
                original_data.append({
                    'user_message': str(text),
                    'classification': str(original_label)
                })
            except Exception as e:
                logging.warning(f"Error converting sample {i}: {e}")
                continue
        
        logging.info(f"Successfully converted {len(original_data)} samples for validation")
        
        if not original_data:
            logging.error("No data available for validation")
            return False
        
        # Run selected validators
        for vname, validator in validator_instances:
            logging.info(f"Running {vname} validation...")
            if vname == 'schema':
                res = validator.validate_schema(original_data)
            elif vname == 'quality':
                res = validator.validate_quality(original_data)
            elif vname == 'length':
                res = validator.validate_length(original_data)
            else:
                continue

            if not res.get('valid', False):
                logging.error(
                    f"{vname.capitalize()} validation failed: {len(res.get('errors', res.get('issues', [])))} issues"
                )
                for msg in res.get('errors', res.get('issues', []))[:5]:
                    logging.error(f"  {msg}")
                return False
            
        # Log validation results
        logging.info("Data validation passed")
        # Optionally log individual validator metrics
        # (omitted for brevity)
                
        return True
        
    except Exception as e:
        logging.error(f"Error during data validation: {e}")
        import traceback
        logging.error(f"Validation traceback: {traceback.format_exc()}")
        return False

def monitor_progress():
    """Monitor progress by watching the results.csv file"""
    results_path = os.path.join('results', 'results.csv')
    last_modified = 0
    last_count = 0
    
    while True:
        try:
            if os.path.exists(results_path):
                current_modified = os.path.getmtime(results_path)
                if current_modified > last_modified:
                    # Read current results
                    with open(results_path, 'r') as f:
                        lines = f.readlines()
                    current_count = len(lines) - 1  # Subtract header
                    
                    if current_count > last_count:
                        # New models completed
                        for i in range(last_count + 1, current_count + 1):
                            if i < len(lines):
                                parts = lines[i].strip().split(',')
                                if len(parts) >= 7:
                                    model_name = parts[0]
                                    status = parts[6]
                                    if status == 'success':
                                        f1_score = parts[3] if len(parts) > 3 else 'N/A'
                                        print(f"✅ {model_name} completed - F1: {f1_score}")
                                    else:
                                        print(f"❌ {model_name} failed: {status}")
                        
                        last_count = current_count
                    
                    last_modified = current_modified
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Progress monitor error: {e}")
            time.sleep(5)

def start_progress_monitor():
    """Start the progress monitor in a separate thread"""
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    return monitor_thread

@monitor_resources
def main():
    """Main experiment pipeline"""
    try:
        # Setup
        setup_logging()
        create_clean_results_csv()
        
        # Load configuration
        config = load_config()
        logging.info("Configuration loaded")
        
        # Initialize production experiment runner
        try:
            exp_config = ExperimentConfig(
                experiment_name=config['experiment_tracking']['experiment_name'],
                enable_mlflow=config['experiment_tracking']['enable_mlflow'],
                enable_wandb=config['experiment_tracking']['enable_wandb'],
                enable_ray=config['experiment_tracking']['enable_ray'],
                tracking_uri=config['experiment_tracking']['tracking_uri']
            )
            
            experiment_runner = ProductionExperimentRunner(exp_config)
            logging.info("Experiment runner initialized successfully")
        except Exception as e:
            logging.error(f"Experiment runner initialization failed: {e}")
            import traceback
            logging.error(f"Initialization traceback: {traceback.format_exc()}")
            raise
        
        # Start experiment
        run_name = f"experiment_{int(time.time())}"
        experiment_runner.start_experiment(run_name)
        
        # Load data
        logging.info("Loading data...")
        try:
            data_dict = load_data(config)
            logging.info(f"Data loaded: {len(data_dict['X_train'])} train, {len(data_dict['X_test'])} test samples")
        except Exception as e:
            logging.error(f"Data loading failed: {e}")
            import traceback
            logging.error(f"Data loading traceback: {traceback.format_exc()}")
            raise

        # ------------------------------------------------------------
        # External blind-set loading (optional)
        # ------------------------------------------------------------
        ext_data = None
        ext_cfg = config.get('external_validation', {})
        if ext_cfg.get('enable', False):
            try:
                from data.external_loader import load_external_dataset
                ext_data = load_external_dataset(
                    ext_cfg['data_path'],
                    data_dict['label_encoder'],
                    config.get('preprocessing', {}).get('text_cleaning', {})
                )
                logging.info(f"External validation set loaded: {len(ext_data['X_ext'])} samples")
            except Exception as e:
                logging.warning(f"Could not load external validation set: {e}")
                ext_data = None
        
        # Validate data
        if config['data_validation']['enable_validation']:
            logging.info("Starting data validation...")
            try:
                if not validate_data_pipeline(data_dict, config):
                    logging.error("Data validation failed - stopping experiment")
                    experiment_runner.end_experiment()
                    return
            except Exception as e:
                logging.error(f"Data validation error: {e}")
                logging.info("Skipping data validation due to error, continuing with experiment...")
        
        # Log data statistics
        experiment_runner.log_metrics({
            "data/train_samples": len(data_dict['X_train']),
            "data/test_samples": len(data_dict['X_test']),
            "data/num_classes": data_dict['num_classes'],
        })
        
        # Log class weights as separate metrics
        for class_id, weight in data_dict['class_weight_dict'].items():
            experiment_runner.log_metrics({f"data/class_weight_{class_id}": float(weight)})
        
        # Start progress monitor
        print("🚀 Starting experiment with progress monitoring...")
        print("📊 Models to train:")
        print(f"   Traditional ML: {config.get('traditional_ml', {}).get('models', [])}")
        print(f"   Deep Learning: {config.get('deep_learning', {}).get('models', [])}")
        print(f"   Transformers: {config.get('transformers', {}).get('models', [])}")
        print("=" * 60)
        
        monitor_thread = start_progress_monitor()

        # Train traditional ML models
        logging.info("Starting traditional ML models...")
        try:
            trad_success, trad_failed = train_and_evaluate_models(data_dict, config, ext_data)
            logging.info(f"Traditional ML completed: {len(trad_success)} success, {len(trad_failed)} failed")
        except Exception as e:
            logging.error(f"Traditional ML training error: {e}")
            trad_success, trad_failed = [], []
        
        # Train deep learning (Keras) models if enabled
        dl_success, dl_failed = [], []
        if config.get('enable_dl', True):
            logging.info("Starting deep learning (Keras) models...")
            try:
                dl_success, dl_failed = train_and_evaluate_deep_models(data_dict, config, ext_data)
                logging.info(f"Deep learning completed: {len(dl_success)} success, {len(dl_failed)} failed")
            except Exception as e:
                logging.error(f"Deep learning training error: {e}")
                dl_success, dl_failed = [], []

        # Train transformer models (HuggingFace) separately
        logging.info("Starting transformer models...")
        try:
            tf_success, tf_failed = train_and_evaluate_transformer_models(data_dict, config, ext_data)
            logging.info(f"Transformers completed: {len(tf_success)} success, {len(tf_failed)} failed")
        except Exception as e:
            logging.error(f"Transformer training error: {e}")
            tf_success, tf_failed = [], []
        
        # Summary
        total_success = len(trad_success) + len(dl_success) + len(tf_success)
        total_failed = len(trad_failed) + len(dl_failed) + len(tf_failed)
        
        logging.info("Experiment completed!")
        logging.info("Final Results:")
        logging.info(f"   Successful: {total_success} models")
        logging.info(f"   Failed: {total_failed} models")
        
        if trad_success:
            logging.info(f"   Traditional ML: {', '.join(trad_success)}")
        if dl_success:
            logging.info(f"   Deep Learning (Keras): {', '.join(dl_success)}")
        if tf_success:
            logging.info(f"   Transformers: {', '.join(tf_success)}")
        if trad_failed or dl_failed or tf_failed:
            logging.warning(f"   Failed models: {', '.join(trad_failed + dl_failed + tf_failed)}")
        
        # Display final results summary
        print("\n" + "=" * 60)
        print("🏁 EXPERIMENT COMPLETED!")
        print("=" * 60)
        
        # Read and display final results
        results_path = os.path.join('results', 'results.csv')
        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path)
                successful_models = df[df['status'] == 'success']
                failed_models = df[df['status'] != 'success']
                
                if not successful_models.empty:
                    print("✅ SUCCESSFUL MODELS:")
                    for _, row in successful_models.iterrows():
                        print(f"   {row['model']}: F1={row['f1_macro']:.4f}")
                
                if not failed_models.empty:
                    print("❌ FAILED MODELS:")
                    for _, row in failed_models.iterrows():
                        print(f"   {row['model']}: {row['status']}")
                
                print(f"\n📈 Total: {len(successful_models)} successful, {len(failed_models)} failed")
                
            except Exception as e:
                print(f"Error reading results: {e}")
        
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

        # Generate training analysis
        try:
            logging.info("Generating training analysis with results/training_analysis.py ...")
            subprocess.run(["python", "results/training_analysis.py"], check=True)
            logging.info("Training analysis generated successfully.")
        except Exception as e:
            logging.error(f"Training analysis generation failed: {e}")

        # Generate DL-specific visualizations
        try:
            logging.info("Generating deep learning visualizations with results/dl_visualizations.py ...")
            subprocess.run(["python", "results/dl_visualizations.py"], check=True)
            logging.info("Deep learning visualizations generated successfully.")
        except Exception as e:
            logging.error(f"Deep learning visualization generation failed: {e}")

        # Run comprehensive validation strategies
        try:
            logging.info("Running comprehensive validation strategies...")
            from validation_strategies import run_comprehensive_validation, save_validation_results, create_validation_report
            
            # Run validation for each successful model
            for _, row in successful_models.iterrows():
                model_name = row['model']
                try:
                    # Load the trained model (implementation depends on model type)
                    # This is a placeholder - you'd need to implement model loading
                    validation_results = run_comprehensive_validation(data_dict, config, None, model_name)
                    save_validation_results(validation_results, f'results/validation_{model_name}.json')
                    create_validation_report(validation_results, f'results/validation_report_{model_name}.md')
                    logging.info(f"Validation completed for {model_name}")
                except Exception as e:
                    logging.error(f"Validation failed for {model_name}: {e}")
                    
        except Exception as e:
            logging.error(f"Comprehensive validation failed: {e}")

    except Exception as e:
        logging.error(f"Critical error in main experiment pipeline: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main() 