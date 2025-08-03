# Production-Grade ML Experiment Framework for Prompt Classification

A comprehensive, enterprise-ready machine learning experiment framework for prompt classification with support for traditional ML, deep learning, and transformer models. Built with production-grade features including model evaluation, A/B testing, deployment pipelines, and comprehensive monitoring.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete production pipeline
python run_pipeline.py          # convenience wrapper
# (equivalent to: python experiment/experiment_runner.py)
```

This single command will:
1. **Load and validate data** with comprehensive quality checks
2. **Train all configured models** across three categories (Traditional ML, Deep Learning, Transformers)
3. **Perform rigorous evaluation** with cross-validation, statistical significance testing, **and an external blind-set validation** (column `f1_external` in `results/results.csv`)
4. **Generate comprehensive visualizations** and insights
5. **Select the best model** using multi-criteria optimization
6. **Prepare deployment artifacts** with model cards and documentation

## 🏗️ Architecture Overview

```
7-Experiment_With_Models/
├── config/                     # Configuration management
│   ├── config.json            # Production ML configuration
│   └── config.py              # Configuration loader
├── data/                      # Data processing pipeline
│   ├── data_loader.py         # Data loading and preprocessing
│   └── validation/            # Data quality validation
├── training/                  # Training modules
│   ├── trainer.py             # Traditional ML (sklearn)
│   ├── adaptive_trainer.py    # Deep Learning (Keras/TensorFlow)
│   └── transformer_trainer.py # Transformers (HuggingFace)
├── models/                    # Model definitions and storage
│   ├── keras_models.py        # Keras model factory
│   └── [model_directories]/   # Trained model artifacts
├── experiment/                # Experiment orchestration
│   └── experiment_runner.py   # Main experiment runner
├── validation_jsons/          # Small blind-set datasets for external validation
│   └── outside_test_data.json # Default 96-prompt test set
├── run_pipeline.py            # Thin wrapper that sets cwd & launches experiment
├── results/                   # Output and analysis
│   ├── visual.py              # Standard visualizations
│   ├── dl_visualizations.py   # Deep learning specific viz
│   └── training_analysis.py   # Training analysis
├── utils/                     # Utility modules
├── logs/                      # Experiment logs
└── requirements.txt           # Dependencies
```

## 📊 Model Categories

### 1. **Traditional Machine Learning**
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble tree-based classification
- **Support Vector Machine**: Kernel-based classification
- **Naive Bayes**: Probabilistic classification
- **XGBoost**: Gradient boosting for classification

### 2. **Deep Learning (Keras/TensorFlow)**
- **CNN**: Convolutional Neural Networks for text classification
- **LSTM**: Long Short-Term Memory networks
- **BiLSTM**: Bidirectional LSTM networks
- **Transformer**: Attention-based transformer models
- **Hybrid**: Combined architectures (CNN + LSTM)

### 3. **Transformers (HuggingFace)**
- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: Distilled version of BERT (40% smaller, 60% faster)
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ALBERT**: A Lite BERT (very lightweight, 45MB)
- **DistilRoBERTa**: Distilled version of RoBERTa

## 🔧 Configuration

### Production-Grade Configuration Features

The framework includes comprehensive configuration for production ML:

#### **Model Evaluation**
```json
{
  "model_evaluation": {
    "enable_cross_validation": true,
    "cv_folds": 4,
    "stratified_sampling": true,
    "metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"],
    "confidence_intervals": true,
    "statistical_significance_test": true,
    "performance_thresholds": {
      "min_accuracy": 0.7,
      "min_f1": 0.6,
      "max_latency_ms": 100
    }
  }
}
```

#### **Model Selection**
```json
{
  "model_selection": {
    "selection_strategy": "multi_criteria",
    "criteria": {
      "performance_weight": 0.6,
      "efficiency_weight": 0.2,
      "interpretability_weight": 0.1,
      "robustness_weight": 0.1
    },
    "ensemble_methods": {
      "enable_voting": true,
      "enable_stacking": true,
      "voting_strategy": "soft"
    }
  }
}
```

#### **A/B Testing Framework**
```json
{
  "a_b_testing": {
    "enable_ab_testing": false,
    "test_duration_days": 14,
    "traffic_split": {"control": 0.5, "treatment": 0.5},
    "success_metrics": ["accuracy", "user_satisfaction", "response_time"],
    "statistical_power": 0.8,
    "significance_level": 0.05
  }
}
```

#### **Model Deployment**
```json
{
  "model_deployment": {
    "deployment_strategy": "blue_green",
    "rollback_threshold": 0.05,
    "model_versioning": {
      "enable_semantic_versioning": true,
      "version_format": "major.minor.patch"
    },
    "containerization": {
      "enable_docker": true,
      "base_image": "python:3.9-slim"
    }
  }
}
```

## 📈 Advanced Features

### **1. Comprehensive Model Evaluation**
- **Cross-validation** with stratified sampling
- **Statistical significance testing** between models
- **Confidence intervals** for performance metrics
- **Baseline comparison** against majority class
- **Performance thresholds** for production readiness

### **2. Multi-Criteria Model Selection**
- **Performance-based selection** (accuracy, F1, precision, recall)
- **Efficiency considerations** (inference time, memory usage)
- **Interpretability scoring** (feature importance, model complexity)
- **Robustness assessment** (cross-validation stability)
- **Ensemble methods** (voting, stacking, blending)

### **3. Production Monitoring**
- **Real-time performance tracking**
- **Data drift detection**
- **Model degradation alerts**
- **Resource utilization monitoring**
- **Error rate tracking**

### **4. Model Fairness & Compliance**
- **Fairness testing** across protected attributes
- **Bias detection** and mitigation
- **GDPR compliance** features
- **Audit trail** for model changes
- **Data privacy** controls

### **5. Error Handling & Resilience**
- **Circuit breaker pattern** for fault tolerance
- **Retry policies** with exponential backoff
- **Fallback strategies** for graceful degradation
- **Comprehensive logging** and error tracking

### **6. Performance Optimization**
- **Model compression** (quantization, pruning)
- **Prediction caching** for improved latency
- **Batch processing** for throughput optimization
- **Resource management** and scaling

## 🎯 Usage Examples

### **Basic Production Run**
```bash
# Run complete pipeline with all models
python experiment/experiment_runner.py
```

### **Custom Configuration**
```python
from experiment.experiment_runner import ProductionExperimentRunner
from config.config import load_config

# Load custom configuration
config = load_config("custom_config.json")

# Initialize and run experiment
runner = ProductionExperimentRunner(config)
results = runner.run_experiment()

# Access results
best_model = results.get_best_model()
performance_metrics = results.get_performance_summary()
```

### **Model-Specific Training**
```json
{
  "traditional_ml": {
    "models": ["logistic_regression", "random_forest"],
    "cv_folds": 10
  },
  "deep_learning": {
    "models": ["cnn", "lstm"],
    "max_epochs": 20
  },
  "transformers": {
    "models": ["distilbert"],
    "max_epochs": 5
  }
}
```

### **Ensemble Training**
```json
{
  "model_selection": {
    "ensemble_methods": {
      "enable_voting": true,
      "enable_stacking": true,
      "voting_strategy": "soft"
    }
  }
}
```

## 📊 Output Interpretation

### **Results Structure**
```
results/
├── results.csv              # Complete experiment results
├── experiment_results.json  # Detailed results with metadata
├── images/                  # Comprehensive visualizations
│   ├── model_comparison.png
│   ├── training_curves.png
│   ├── feature_importance.png
│   └── fairness_analysis.png
└── models/                  # Trained model artifacts
    ├── [model_name]/
    │   ├── model.pkl
    │   ├── config.json
    │   └── performance.json
```

### **Key Metrics to Monitor**
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Per-class performance
- **ROC AUC**: Area under ROC curve
- **Cross-validation stability**: Model robustness
- **Inference latency**: Production readiness
- **Memory usage**: Resource efficiency

### **Model Selection Criteria**
1. **Performance**: F1 score > 0.7, accuracy > 0.8
2. **Efficiency**: Latency < 100ms, memory < 2GB
3. **Stability**: CV std < 0.05
4. **Interpretability**: Feature importance available
5. **Fairness**: Bias score < 0.1

## 🔍 Troubleshooting

### **Common Issues & Solutions**

#### **Memory Errors**
```bash
# Reduce batch size in config
"batch_size": 16  # Instead of 64
"max_length": 256  # Instead of 512
```

#### **Slow Training**
```bash
# Enable GPU acceleration
"enable_gpu": true
"mixed_precision": true

# Reduce model complexity
"layers": 1  # Instead of 3
"units": 64  # Instead of 256
```

#### **Poor Performance**
- Check data quality and class balance
- Verify preprocessing pipeline
- Consider feature engineering
- Try different model architectures

#### **Transformer-Specific Issues**
- Ensure MLflow is uninstalled (causes conflicts)
- Use smaller batch sizes for M1/M2 Macs
- Monitor memory usage during training

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### **Core Dependencies**
- `tensorflow>=2.8.0` - Deep learning framework
- `scikit-learn>=1.0.0` - Traditional ML
- `transformers>=4.20.0` - HuggingFace transformers
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization

### **Production Dependencies**
- `mlflow>=1.20.0` - Experiment tracking
- `shap>=0.40.0` - Model interpretability
- `optuna>=2.10.0` - Hyperparameter optimization
- `prometheus-client>=0.12.0` - Monitoring
- `docker>=6.0.0` - Containerization

## 🚀 Deployment

### **Docker Deployment**
```bash
# Build container
docker build -t prompt-classifier .

# Run container
docker run -p 8080:8080 prompt-classifier
```

### **Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prompt-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prompt-classifier
  template:
    metadata:
      labels:
        app: prompt-classifier
    spec:
      containers:
      - name: classifier
        image: prompt-classifier:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "1000m"
            memory: "2Gi"
```

## 📝 Contributing

### **Development Guidelines**
1. **Modular Architecture**: Keep components loosely coupled
2. **Comprehensive Testing**: Unit tests for all modules
3. **Documentation**: Update README and docstrings
4. **Configuration**: Add new options to config.json
5. **Logging**: Use structured logging throughout

### **Code Quality**
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive error handling
- Write unit tests for new features
- Update configuration documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Acknowledgments

- **HuggingFace** for transformer models and training utilities
- **TensorFlow/Keras** for deep learning framework
- **Scikit-learn** for traditional machine learning
- **MLflow** for experiment tracking and model management

---

# ML Experiment Framework for Prompt Classification

A comprehensive, production-ready machine learning experiment framework for prompt classification with support for traditional ML and deep learning models.

## 🚀 Quick Start

```bash
# Run the complete pipeline
python run_pipeline.py
```

This single command will:
1. Load and validate data
2. Train all configured models (traditional ML + deep learning)
3. Generate comprehensive visualizations
4. Create training analysis reports
5. Generate deep learning specific insights

## 📊 Generated Outputs

After running the experiment, you'll find:

### Visualizations (`results/images/`)
- `model_comparison.png` - Overall model performance comparison
- `radar_chart.png` - Top 5 models radar chart
- `parallel_coordinates.png` - Multi-dimensional model comparison
- `training_analysis.png` - Training progress analysis
- `convergence_analysis.png` - Model convergence patterns

### Deep Learning Visualizations (`results/images/`)
- `dl_training_curves.png` - Detailed DL training analysis (6 subplots)
- `dl_architecture_comparison.png` - Architecture-specific comparisons
- `dl_insights.txt` - DL-specific recommendations and insights

### Training Analysis (`results/images/`)
- `training_analysis.png` - Training curves and convergence analysis
- `training_insights.txt` - Automated training insights and recommendations

### Data Files
- `results.csv` - Complete experiment results
- `experiment.log` - Detailed experiment log

## 🏗️ Architecture

```
7-Experiment_With_Models/
├── config/                 # Configuration management
│   ├── config.json        # Main configuration file
│   └── config.py          # Configuration loader
├── data/                  # Data processing
│   ├── data_loader.py     # Data loading and preprocessing
│   └── validation/        # Data validation modules
├── training/              # Training modules
│   ├── trainer.py         # Traditional ML trainer
│   └── adaptive_trainer.py # Deep learning trainer
├── models/                # Model definitions
│   └── keras_models.py    # Keras model factory
├── experiment/            # Experiment orchestration
│   └── experiment_runner.py # Main experiment runner
├── results/               # Output and analysis
│   ├── visual.py          # Standard visualizations
│   ├── dl_visualizations.py # Deep learning specific viz
│   └── training_analysis.py # Training analysis
├── utils/                 # Utility modules
├── logs/                  # Experiment logs
├── mlruns/                # MLflow tracking
└── run_pipeline.py      # Main entry point
```

## 🔧 Configuration

### Main Configuration (`config/config.json`)

```json
{
  "data": {
    "data_path": "path/to/your/data",
    "test_size": 0.2,
    "random_state": 42,
    "max_features": 10000
  },
  "models": [
    {
      "name": "logistic_regression",
      "type": "sklearn",
      "params": {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"]
      }
    },
    {
      "name": "cnn_default",
      "type": "cnn",
      "params": {
        "filters": [32, 64],
        "kernel_size": [3, 5],
        "dropout": [0.3, 0.5]
      }
    }
  ],
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping": true,
    "patience": 3
  },
  "experiment_tracking": {
    "enable_mlflow": true,
    "enable_wandb": false,
    "experiment_name": "prompt_classification"
  }
}
```

### Deep Learning Models

The framework supports various deep learning architectures:

- **CNN**: Convolutional Neural Networks for text classification
- **LSTM**: Long Short-Term Memory networks
- **BiLSTM**: Bidirectional LSTM networks
- **Transformer**: Attention-based transformer models
- **Hybrid**: Combined architectures (CNN + LSTM)
- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: Distilled version of BERT (40% smaller, 60% faster)
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ALBERT**: A Lite BERT (very lightweight, 45MB)
- **DistilRoBERTa**: Distilled version of RoBERTa

Each architecture can be configured with:
- Layer configurations (filters, units, heads)
- Dropout rates
- Learning rate schedules
- Regularization parameters

## 📈 Deep Learning Visualizations

The framework includes specialized deep learning visualizations:

### Training Curves Analysis
- **Loss Curves**: Training vs validation loss with gradient analysis
- **Accuracy Curves**: Training vs validation accuracy progression
- **Learning Rate**: Learning rate scheduling visualization
- **Gradient Analysis**: Loss change rate and stability
- **Overfitting Detection**: Train-validation gap analysis
- **Stability Metrics**: Rolling loss standard deviation

### Architecture Comparison
- **Performance by Architecture**: Best accuracy comparison
- **Convergence Speed**: Epochs to 90% of best performance
- **Stability Analysis**: Loss variance vs performance
- **Overfitting Analysis**: Train-val gap vs performance

### Automated Insights
The system generates architecture-specific recommendations:
- CNN: Filter/layer optimization suggestions
- LSTM: Bidirectional and unit count recommendations
- Transformer: Attention head and layer suggestions
- BERT: Fine-tuning and learning rate recommendations
- DistilBERT: Performance vs efficiency trade-offs
- RoBERTa: Advanced fine-tuning strategies
- ALBERT: Lightweight model optimization
- DistilRoBERTa: Distillation-specific insights
- General: Learning rate, regularization, and convergence tips

## 🔍 Training Analysis Features

### Automated Insights
- **Convergence Analysis**: Identifies optimal stopping points
- **Overfitting Detection**: Flags models with high train-val gaps
- **Learning Rate Analysis**: Suggests scheduling improvements
- **Stability Assessment**: Identifies models with high variance

### Performance Metrics
- **Training Curves**: Loss and accuracy over epochs
- **Convergence Patterns**: Time to convergence analysis
- **Resource Utilization**: Memory and time tracking
- **Model Comparison**: Cross-model performance analysis

## 🚀 Advanced Features

### Experiment Tracking
- **MLflow Integration**: Comprehensive experiment tracking
- **Weights & Biases**: Optional W&B integration
- **Model Registry**: Versioned model storage
- **Artifact Logging**: Save models, plots, and data

### Data Validation
- **Schema Validation**: Ensures data format consistency
- **Quality Validation**: Checks for data quality issues
- **Automated Warnings**: Flags potential problems

### Resource Monitoring
- **Memory Tracking**: Monitor memory usage during training
- **Time Profiling**: Track training and inference times
- **GPU Utilization**: Monitor GPU usage for deep learning

### Distributed Training
- **Ray Integration**: Optional distributed training support
- **Multi-GPU**: Automatic multi-GPU training
- **Hyperparameter Tuning**: Distributed hyperparameter search

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow>=2.8.0`
- `scikit-learn>=1.0.0`
- `pandas>=1.3.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `mlflow>=1.20.0`
- `numpy>=1.21.0`

## 🎯 Usage Examples

### Basic Experiment
```bash
python run_pipeline.py
```

### Custom Configuration
```python
from experiment.experiment_runner import ProductionExperimentRunner
from config.config import load_config

config = load_config("custom_config.json")
runner = ProductionExperimentRunner(config)
runner.run_experiment()
```

### Deep Learning Focus
```json
{
  "models": [
    {"name": "cnn_best", "type": "cnn"},
    {"name": "lstm_best", "type": "lstm"},
    {"name": "transformer_best", "type": "transformer"}
  ],
  "training": {
    "epochs": 20,
    "early_stopping": true
  }
}
```

## 📊 Output Interpretation

### Standard Visualizations
- **Model Comparison**: Overall performance ranking
- **Radar Chart**: Multi-metric comparison of top models
- **Parallel Coordinates**: Multi-dimensional model analysis

### Deep Learning Insights
- **Training Stability**: Look for smooth, decreasing loss curves
- **Overfitting**: Watch for increasing train-val gaps
- **Convergence**: Identify optimal stopping points
- **Architecture Performance**: Compare different DL architectures

### Recommendations
- **High Variance**: Reduce learning rate or add regularization
- **Overfitting**: Add dropout or implement early stopping
- **Slow Convergence**: Increase learning rate or model capacity
- **Poor Performance**: Consider architecture changes or more data

## 🔧 Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or model complexity
2. **Slow Training**: Enable GPU acceleration or reduce data size
3. **Poor Performance**: Check data quality and model configuration
4. **Visualization Errors**: Ensure matplotlib backend is properly configured

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Contributing

1. Follow the modular architecture
2. Add comprehensive logging
3. Include data validation
4. Update configuration documentation
5. Add tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Transformer Training (HuggingFace)

`training/transformer_trainer.py` handles all HuggingFace models (BERT, DistilBERT, RoBERTa, etc.).  Key features:

* Separate from Keras/DL flow – keeps dependencies light.
* Uses `TrainingArguments` with:
  * Linear LR scheduler with 10% warm-up.
  * Weight decay (0.01) and AdamW optimizer.
  * Gradient clipping (`max_grad_norm=1.0`).
* Automatically saves best checkpoints and logs metrics to `results/results.csv`.

The `experiment_runner` now calls three distinct pipelines:

1. **Traditional ML** (`training/trainer.py`)
2. **Deep-learning / Keras** (`training/adaptive_trainer.py`)
3. **Transformers** (`training/transformer_trainer.py`)

Results from all pipelines are aggregated in the final summary.

## 🔑 Default Configuration & Rationale  
Below is **why** the numbers in `config/config.json` are set the way they are.  These values represent a balance between statistical robustness and wall-time on a single Apple-Silicon laptop with 16 K samples.

| Area | Default | Why it's the sweet-spot |
|------|---------|-------------------------|
| **Traditional ML** | `cv_folds = 10`, timeout = 1200 s | 10-fold CV reduces the standard-error ~30 % vs 5 folds while staying well under 1 h total runtime.<br>Each model runs **GridSearchCV** with hard-coded grids in `training/trainer.py` (LogReg, RF, SVM-linear, NB, XGB). |
| **Deep-Learning** | `max_epochs = 15`, `batch_size = 64`, `early_stopping_patience = 3` | 15 is an upper bound; Early-Stopping usually halts at 7–10 epochs. Batch 64 fits comfortably in M-series RAM and gives GPU/ANE utilisation. |
| **Transformers** | `max_epochs = 8`, `batch_size = 16`, `early_stopping_patience = 3` | Fine-tuning converges in 3–6 epochs; 8 gives head-room. Batch 16 prevents OOM on 16 GB Macs. |
| **Tokenisation** | `max_length = 512` for HF models, 1024 for Keras RNN/CNN | BERT-style models max at 512; CNN/LSTM can benefit from longer sequences. |
| **Hyper-parameter search** | GridSearchCV for classical models **on**, Optuna **off by default** | The built-in grids already cover the typical sweet-spots. Turn on Optuna only when you need deeper sweeps. |
| **Safety & Logging** | `utils/sanitizer.sanitize_text` applied **before** any tokenisation | Guarantees the same text preprocessing in both training and inference and neutralises XSS/RTL/Excel-injection payloads. |

### End-to-End Flow
```text
raw JSON ➜ data_loader (sanitize + label-encode + split)
          ➜ trainer.py              (TF-IDF + GridSearchCV, 10-fold)
          ➜ adaptive_trainer.py     (Keras models, Early-Stopping)
          ➜ transformer_trainer.py  (HF fine-tuning, Early-Stopping)
          ➜ results/results.csv     (one row per model)
          ➜ results/*.py            (visuals + insights)
```
• All three trainers append their metrics to **the same** CSV so downstream plots compare apples-to-apples.  
• Logs land in `logs/` and are parsed by `training_analysis.py` / `dl_visualizations.py` for deeper diagnostics.

### Modifying the defaults
* **Faster iteration** Lower `traditional_ml.cv_folds` to 5 and set `transformers.max_epochs` to 4.  
* **Exhaustive sweep** Turn `hyperparameter_optimization.enable_optuna` to `true` and define search spaces for every model.  
* **Memory-constrained** Drop `batch_size` for DL to 32 and transformer `max_length` to 256.

> ℹ️  You do **not** need to edit Python files for common tweaks—90 % of day-to-day tuning is config-only.

## 🚧 Future Expansion Opportunities
The configuration file already reserves space for capabilities that are **not yet wired into the Python code**.  They are safe to keep (the pipeline simply ignores them) and act as a roadmap for growth:

* **hyperparameter_optimization** – Optuna search spaces for every model.
* **model_evaluation** – statistical tests, confidence intervals, baseline thresholds.
* **model_selection** – multi-criteria scorer and ensemble meta-learning.
* **a_b_testing** – live traffic splitting and on-line significance testing.
* **model_deployment** – BentoML/Docker/Kubernetes hooks for serving.
* **performance_optimization** – quantisation, pruning, caching.
* **monitoring** – Prometheus metrics exporter.
* **error_handling** – retry / circuit-breaker wrappers.
* **compliance** – fairness auditing, GDPR data retention.
* **documentation** – automatic model-cards and Swagger API docs.

These blocks let you enable new features configuration-first once their corresponding modules are added to `utils/`, `experiment/`, or `serving/` packages.

<!--
# ## 🔍 External Blind-Set Validation
#
# The framework supports an optional *blind-set* evaluation step to detect data-leakage or over-fitting.
# Enable it via `config/external_validation` (on by default).  The loader reads any JSON list whose
# objects contain `user_message` and `classification` keys.  Metrics are written to the new
# `f1_external` column in `results/results.csv`.
#
# *Replace* `validation_jsons/outside_test_data.json` with your own curated prompts to see how well
# models generalise beyond the training distribution.
-->