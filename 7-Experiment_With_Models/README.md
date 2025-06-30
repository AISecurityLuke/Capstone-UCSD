# Production-Grade ML Experiment Framework

A comprehensive machine learning experiment framework for prompt classification with production-ready features including experiment tracking, data validation, distributed training, and model serving.

## Key Features

### Production-Grade Infrastructure
- Experiment Tracking: MLflow and Weights & Biases integration
- Model Registry: Versioned model storage and management
- Data Validation: Great Expectations and Evidently for data quality
- Distributed Training: Ray and TensorFlow distributed strategies
- Hyperparameter Optimization: Optuna integration with Ray Tune
- Model Serving: BentoML integration for production deployment

### Advanced Training Features
- Adaptive Training: Real-time monitoring and automatic adaptation
- Architecture Search: Automatic testing of different model architectures
- Gradient Monitoring: Explosion/vanishing gradient detection and recovery
- Resource Management: GPU/CPU monitoring and optimization
- Class Imbalance Handling: Automatic class weight computation

### Security & Monitoring
- Input Sanitization: Protection against injection attacks
- Performance Monitoring: Prometheus metrics and health checks
- Rate Limiting: Configurable request throttling
- Data Lineage: Complete experiment tracking and reproducibility

## Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies
- `scikit-learn`, `xgboost` - Traditional ML models
- `tensorflow`, `tf-keras` - Deep learning framework
- `transformers[torch]` - HuggingFace models
- `accelerate>=0.26.0` - Distributed training support

### Production Tools
- `mlflow>=2.8.0` - Experiment tracking and model registry
- `wandb>=0.16.0` - Weights & Biases integration
- `ray[tune]>=2.8.0` - Distributed training and hyperparameter optimization
- `optuna>=3.4.0` - Hyperparameter optimization
- `great-expectations>=0.17.0` - Data validation
- `evidently>=0.3.0` - Data drift detection
- `bentoml>=1.2.0` - Model serving

## Architecture

```
├── experiment_manager.py    # Production experiment tracking
├── data_validator.py        # Data quality validation
├── data_loader.py          # Data loading and preprocessing
├── trainer.py              # Traditional ML training
├── deep_trainer.py         # Deep learning training
├── run_experiments.py      # Main experiment pipeline
├── config.json            # Configuration management
└── utils/                 # Utility modules
    ├── model_manager.py   # Model management
    ├── monitor.py         # Resource monitoring
    └── sanitizer.py       # Input sanitization
```

## Configuration

The framework is configured via `config.json`:

```json
{
  "data_path": "../5-Data_Wrangling/combined_demo.json",
  "experiment_tracking": {
    "enable_mlflow": true,
    "enable_wandb": false,
    "enable_ray": false
  },
  "data_validation": {
    "enable_validation": true
  },
  "distributed_training": {
    "enable_distributed": false
  }
}
```

### Key Configuration Sections

#### Experiment Tracking
- `enable_mlflow`: Enable MLflow experiment tracking
- `enable_wandb`: Enable Weights & Biases integration
- `enable_ray`: Enable Ray for distributed training

#### Data Validation
- `enable_validation`: Enable data quality validation
- `validation_config`: Data validation rules and thresholds

#### Distributed Training
- `enable_distributed`: Enable distributed training
- `strategy`: TensorFlow distribution strategy
- `num_workers`: Number of worker processes

---

## Configuration Reference: Every Option Explained

Below is a detailed reference for every option in `config.json`, including what it does, why it's needed, when/why you would change it, and example values.

### Top-Level Options

#### `data_path`
- **What it does:** Specifies the path to your input data file (JSON format).
- **Why it's needed:** The pipeline needs to know where to load your training data from.
- **When/why you'd change it:** Change this if your data file moves or you want to use a different dataset.
- **Example values:**
  - `"../5-Data_Wrangling/combined_demo.json"`
  - `"/absolute/path/to/data.json"`

#### `test_size`
- **What it does:** Fraction of data to use for the test set (e.g., 0.2 = 20%).
- **Why it's needed:** Controls the split between training and testing data for model evaluation.
- **When/why you'd change it:**
  - Increase for more robust test evaluation (e.g., 0.3 for 30% test).
  - Decrease to maximize training data (e.g., 0.1 for 10% test).
- **Example values:** `0.2`, `0.1`, `0.3`

#### `random_seed`
- **What it does:** Sets the random seed for reproducibility.
- **Why it's needed:** Ensures that data splits and model results are consistent across runs.
- **When/why you'd change it:**
  - Change only if you want to test different random splits.
- **Example values:** `42`, `1234`

#### `max_train_samples`
- **What it does:** Maximum number of training samples to use (null = use all).
- **Why it's needed:** Useful for debugging or quick tests on a subset of data.
- **When/why you'd change it:**
  - Set to a small number for fast prototyping.
  - Leave as `null` for full dataset training.
- **Example values:** `null`, `1000`, `5000`

#### `enable_dl`
- **What it does:** Enables or disables deep learning models in the pipeline.
- **Why it's needed:** Lets you quickly toggle deep learning training on/off.
- **When/why you'd change it:**
  - Set to `false` to run only traditional ML models.
- **Example values:** `true`, `false`

#### `architecture_search`
- **What it does:** Enables automatic architecture search for deep learning models.
- **Why it's needed:** Allows the system to try different model architectures and pick the best.
- **When/why you'd change it:**
  - Set to `false` to use only default architectures (faster, less thorough).
- **Example values:** `true`, `false`

#### `adaptive_training`
- **What it does:** Enables adaptive training (dynamic learning rate, gradient monitoring, etc.).
- **Why it's needed:** Improves model robustness and convergence by automatically adjusting training parameters.
- **When/why you'd change it:**
  - Set to `false` for static training (less adaptive, but more predictable).
- **Example values:** `true`, `false`

---

### `traditional_ml` Section

#### `models`
- **What it does:** List of traditional ML models to train.
- **Why it's needed:** Specifies which algorithms to include in the experiment.
- **When/why you'd change it:**
  - Add/remove models based on your needs or experiment goals.
- **Example values:** `["logistic_regression", "random_forest", "svm", "naive_bayes", "xgboost"]`

#### `cv_folds`
- **What it does:** Number of cross-validation folds for hyperparameter tuning.
- **Why it's needed:** Controls the robustness of model evaluation during grid search.
- **When/why you'd change it:**
  - Increase for more robust evaluation (slower).
  - Decrease for faster tuning (less robust).
- **Example values:** `5`, `10`

#### `n_jobs`
- **What it does:** Number of CPU cores to use for parallel processing.
- **Why it's needed:** Controls resource usage and training speed.
- **When/why you'd change it:**
  - `-1` uses all available cores (fastest, but uses all resources).
  - Set to a specific number (e.g., `4`) to limit usage.
- **Example values:** `-1`, `1`, `4`

#### `timeout`
- **What it does:** Maximum time (in seconds) allowed for training each model.
- **Why it's needed:** Prevents long-running models from blocking the pipeline.
- **When/why you'd change it:**
  - Increase for complex models or large datasets.
  - Decrease for quick experiments.
- **Example values:** `300`, `600`

---

### `deep_learning` Section

#### `models`
- **What it does:** List of deep learning model architectures to train.
- **Why it's needed:** Specifies which neural network types to include.
- **When/why you'd change it:**
  - Add/remove architectures based on your experiment or hardware.
- **Example values:** `["cnn", "lstm", "bilstm", "transformer", "hybrid"]`

#### `max_epochs`
- **What it does:** Maximum number of training epochs for deep learning models.
- **Why it's needed:** Controls how long each model is trained.
- **When/why you'd change it:**
  - Increase for more thorough training (risk of overfitting).
  - Decrease for faster runs or to avoid overfitting.
- **Example values:** `3`, `10`, `20`

#### `batch_size`
- **What it does:** Number of samples per gradient update.
- **Why it's needed:** Affects training speed and memory usage.
- **When/why you'd change it:**
  - Increase for faster training (if you have enough memory).
  - Decrease if you run out of memory.
- **Example values:** `16`, `32`, `64`

#### `learning_rate`
- **What it does:** Initial learning rate for the optimizer.
- **Why it's needed:** Controls how quickly the model adapts to the data.
- **When/why you'd change it:**
  - Decrease if training is unstable or loss oscillates.
  - Increase if training is too slow.
- **Example values:** `0.001`, `0.01`, `0.0001`

#### `early_stopping_patience`
- **What it does:** Number of epochs with no improvement before stopping training.
- **Why it's needed:** Prevents overfitting and saves time.
- **When/why you'd change it:**
  - Increase for more patience (risk of overfitting).
  - Decrease for faster stopping.
- **Example values:** `3`, `5`, `10`

#### `timeout`
- **What it does:** Maximum time (in seconds) allowed for training each deep learning model.
- **Why it's needed:** Prevents long-running models from blocking the pipeline.
- **When/why you'd change it:**
  - Increase for complex models or large datasets.
  - Decrease for quick experiments.
- **Example values:** `600`, `1200`

#### `enable_gpu`
- **What it does:** Enables GPU acceleration for deep learning models.
- **Why it's needed:** Greatly speeds up training on compatible hardware.
- **When/why you'd change it:**
  - Set to `false` if you want to force CPU training.
- **Example values:** `true`, `false`

#### `mixed_precision`
- **What it does:** Enables mixed precision training (float16 + float32) for speed/memory efficiency.
- **Why it's needed:** Can speed up training and reduce memory usage on modern GPUs.
- **When/why you'd change it:**
  - Set to `false` if you encounter numerical instability.
- **Example values:** `true`, `false`

---

### `experiment_tracking` Section

#### `enable_mlflow`, `enable_wandb`, `enable_ray`
- **What they do:** Enable integration with MLflow, Weights & Biases, and Ray for experiment tracking and distributed training.
- **Why needed:** Allows for experiment logging, visualization, and distributed runs.
- **When/why you'd change:**
  - Enable/disable based on your tracking and infrastructure needs.
- **Example values:** `true`, `false`

#### `tracking_uri`
- **What it does:** URI for MLflow tracking server/database.
- **Why it's needed:** Tells MLflow where to store experiment data.
- **When/why you'd change it:**
  - Change if using a remote MLflow server or different database.
- **Example values:** `"sqlite:///mlruns.db"`, `"http://mlflow-server:5000"`

#### `experiment_name`, `model_registry_name`
- **What they do:** Names for the experiment and model registry in MLflow.
- **Why needed:** Organizes and tracks experiments and models.
- **When/why you'd change:**
  - Use descriptive names for different projects or tasks.
- **Example values:** `"prompt_classification"`, `"my_model_registry"`

---

### `data_validation` Section

#### `enable_validation`
- **What it does:** Enables or disables data validation before training.
- **Why it's needed:** Ensures data quality and prevents training on bad data.
- **When/why you'd change it:**
  - Set to `false` to skip validation (not recommended).
- **Example values:** `true`, `false`

#### `validation_config`
- **What it does:** Contains rules for text validation, data quality, and drift detection.
- **Why it's needed:** Customizes how your data is checked before training.
- **When/why you'd change it:**
  - Adjust thresholds or allowed values to match your data.
- **Example values:** See below for subfields.

##### `text_validation`
- **min_length**: Minimum allowed text length.
- **max_length**: Maximum allowed text length.
- **required_fields**: List of required fields in each data item.
- **allowed_classifications**: List of valid class labels.

##### `data_quality`
- **missing_threshold**: Max allowed fraction of missing values.
- **duplicate_threshold**: Max allowed fraction of duplicate entries.
- **class_balance_threshold**: Min allowed ratio for class balance.

##### `drift_detection`
- **enabled**: Enable/disable drift detection.
- **reference_data_path**: Path to reference data for drift comparison.
- **drift_threshold**: Threshold for detecting drift.

---

### `distributed_training` Section

#### `enable_distributed`
- **What it does:** Enables distributed training across multiple devices or nodes.
- **Why it's needed:** Allows scaling to larger datasets or models.
- **When/why you'd change it:**
  - Enable for multi-GPU or multi-node setups.
- **Example values:** `true`, `false`

#### `strategy`
- **What it does:** Specifies the TensorFlow distribution strategy.
- **Why it's needed:** Controls how training is distributed.
- **When/why you'd change it:**
  - Use `"mirrored"` for multi-GPU, `"multiworker"` for multi-node.
- **Example values:** `"mirrored"`, `"multiworker"`

#### `num_workers`
- **What it does:** Number of worker processes for distributed training.
- **Why it's needed:** Controls parallelism in distributed setups.
- **When/why you'd change it:**
  - Increase for more workers (if hardware allows).
- **Example values:** `4`, `8`

#### `batch_size_per_replica`
- **What it does:** Batch size for each replica in distributed training.
- **Why it's needed:** Controls memory usage and speed per device.
- **When/why you'd change it:**
  - Increase for faster training (if memory allows).
- **Example values:** `32`, `64`

---

### `hyperparameter_optimization` Section

#### `enable_optuna`
- **What it does:** Enables Optuna for hyperparameter search.
- **Why it's needed:** Automates finding the best hyperparameters.
- **When/why you'd change it:**
  - Enable for automated tuning.
- **Example values:** `true`, `false`

#### `num_trials`
- **What it does:** Number of trials for hyperparameter search.
- **Why it's needed:** Controls thoroughness of search.
- **When/why you'd change it:**
  - Increase for more thorough search (slower).
- **Example values:** `50`, `100`

#### `timeout`
- **What it does:** Maximum time (in seconds) for hyperparameter search.
- **Why it's needed:** Prevents long searches from blocking pipeline.
- **When/why you'd change it:**
  - Increase for more time to search.
- **Example values:** `1800`, `3600`

#### `search_spaces`
- **What it does:** Defines the hyperparameter ranges for each model.
- **Why it's needed:** Customizes what parameters are tuned.
- **When/why you'd change it:**
  - Adjust ranges or add/remove parameters as needed.
- **Example values:** See config for structure.

---

### `model_serving` Section

#### `enable_bentoml`
- **What it does:** Enables BentoML for model serving.
- **Why it's needed:** Allows deployment as a REST/gRPC service.
- **When/why you'd change it:**
  - Enable when ready to deploy models.
- **Example values:** `true`, `false`

#### `service_name`, `api_type`, `max_batch_size`, `timeout`
- **What they do:** Configure the name, API type, batch size, and timeout for serving.
- **Why needed:** Controls how the model is exposed and used in production.
- **When/why you'd change it:**
  - Adjust for your deployment needs.
- **Example values:** `"prompt_classifier"`, `"rest"`, `100`, `30`

---

### `monitoring` Section

#### `enable_prometheus`, `metrics_port`, `health_check_interval`, `performance_thresholds`
- **What they do:** Enable Prometheus monitoring, set metrics port, health check interval, and performance thresholds.
- **Why needed:** For production monitoring and alerting.
- **When/why you'd change it:**
  - Enable for production, adjust thresholds for your SLAs.
- **Example values:** `true`, `8000`, `60`, `{ "max_latency_ms": 1000 }`

---

### `security` Section

#### `enable_input_validation`, `max_input_length`, `allowed_characters`, `rate_limiting`
- **What they do:** Control input validation, max input length, allowed characters, and rate limiting.
- **Why needed:** Protects against bad input and abuse.
- **When/why you'd change it:**
  - Tighten for production, relax for research.
- **Example values:** `true`, `1000`, `"utf-8"`, `{ "enabled": false, "requests_per_minute": 100 }`

---

This section is intended as a living reference. Update as you add new config options!

## Usage

### Basic Experiment
```bash
python run_experiments.py
```

### With Production Features
```bash
# Enable MLflow tracking
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Enable Weights & Biases (requires API key)
export WANDB_API_KEY=your_api_key

# Run with distributed training
python run_experiments.py
```

### Custom Configuration
```python
from experiment_manager import ProductionExperimentManager, ExperimentConfig

# Initialize with custom config
config = ExperimentConfig(
    experiment_name="my_experiment",
    enable_mlflow=True,
    enable_wandb=True,
    enable_ray=True
)

manager = ProductionExperimentManager(config)
manager.start_experiment("custom_run")
```

## Experiment Tracking

### MLflow Integration
- Automatic experiment logging
- Model versioning and registry
- Parameter and metric tracking
- Artifact storage

### Weights & Biases Integration
- Real-time training visualization
- Model performance comparison
- Hyperparameter optimization tracking
- Team collaboration features

### Results Analysis
```python
from experiment_manager import ProductionExperimentManager

# Load experiment results
manager = ProductionExperimentManager(config)
comparison = manager.compare_models(model_results)

print(f"Best model: {comparison['best_model']['name']}")
print(f"Performance summary: {comparison['performance_summary']}")
```

## Data Validation

### Schema Validation
- Required field checking
- Data type validation
- Text length constraints
- Classification label validation

### Quality Metrics
- Missing value detection
- Duplicate identification
- Class balance analysis
- Data distribution statistics

### Drift Detection
- Statistical drift detection
- Feature distribution comparison
- Target drift monitoring
- Automated alerts

```python
from data_validator import DataValidator

validator = DataValidator("data_path.json")
validation_results = validator.validate_training_data(data)

if validation_results['overall_valid']:
    print("Data validation passed")
else:
    print("Data validation failed")
```

## Model Training

### Traditional ML Models
- Logistic Regression
- Random Forest
- Support Vector Machine
- Naive Bayes
- XGBoost

### Deep Learning Models
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)
- Transformer
- Hybrid (CNN + LSTM)

### Adaptive Training Features
- Real-time gradient monitoring
- Automatic learning rate adjustment
- Overfitting detection and prevention
- Architecture optimization

## Performance Monitoring

### Training Metrics
- Loss curves and convergence
- Validation performance
- Gradient statistics
- Resource utilization

### Production Metrics
- Model latency
- Throughput (requests/second)
- Error rates
- Memory usage

### Health Checks
- Model availability
- Data pipeline status
- Resource monitoring
- Automated alerts

## Security Features

### Input Validation
- Text sanitization
- Length limits
- Character encoding validation
- Injection attack prevention

### Rate Limiting
- Request throttling
- API key management
- Usage monitoring
- Abuse prevention

## Model Serving

### BentoML Integration
```python
import bentoml
from bentoml import api, artifacts, env, ver

@ver(major=1, minor=0)
@env(pip_packages=["tensorflow", "scikit-learn"])
@artifacts([ModelArtifact("model")])
class PromptClassifierService(bentoml.BentoService):
    
    @api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        text = parsed_json.get("text")
        prediction = self.artifacts.model.predict([text])
        return {"prediction": prediction[0]}
```

### Deployment Options
- REST API
- gRPC service
- Batch processing
- Real-time inference

## Results and Analysis

### Model Comparison
- Performance metrics comparison
- Statistical significance testing
- Resource usage analysis
- Recommendation generation

### Experiment Insights
- Best performing models
- Hyperparameter sensitivity
- Architecture recommendations
- Data quality insights

## Advanced Features

### Hyperparameter Optimization
```python
# Optuna integration
search_space = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]}
}

best_params = manager.hyperparameter_optimization(
    model_fn=create_model,
    search_space=search_space,
    num_trials=50
)
```

### Distributed Training
```python
# Ray integration
strategy = manager.get_distributed_strategy()
with strategy.scope():
    model = create_model()
    model.fit(X_train, y_train, batch_size=32*strategy.num_replicas_in_sync)
```

## Logging and Monitoring

### Comprehensive Logging
- Experiment lifecycle tracking
- Error handling and recovery
- Performance metrics
- Resource utilization

### Real-time Monitoring
- Training progress
- Model performance
- System resources
- Data quality metrics

## Development and Testing

### Testing Framework
```bash
# Run integration tests
python test_adaptive_system.py

# Test individual components
python -m pytest tests/
```

### Development Workflow
1. Configure experiment parameters
2. Validate data quality
3. Run training pipeline
4. Analyze results
5. Deploy best model

## Best Practices

### Experiment Management
- Use descriptive experiment names
- Tag experiments appropriately
- Document model versions
- Track data lineage

### Data Quality
- Validate data before training
- Monitor for data drift
- Maintain data versioning
- Document data sources

### Model Development
- Start with simple baselines
- Use cross-validation
- Monitor for overfitting
- Test on holdout data

### Production Deployment
- Use model versioning
- Implement monitoring
- Set up alerts
- Plan for model updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the documentation
- Review example notebooks
- Open an issue on GitHub
- Contact the development team

---

Note: This framework is designed for production use and includes enterprise-grade features for experiment tracking, model management, and deployment. Ensure proper security measures and monitoring are in place for production deployments.