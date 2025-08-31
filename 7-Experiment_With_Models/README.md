# Production-Grade ML Experiment Framework for Prompt Classification

A comprehensive, enterprise-ready machine learning experiment framework for prompt classification with support for traditional ML, deep learning, and transformer models. Built with production-grade features including model evaluation, A/B testing, deployment pipelines, and comprehensive monitoring.

**🔒 Security-Focused Design**: This framework is specifically optimized for malicious prompt detection with custom metrics that heavily penalize misclassifications between benign (class 0) and malicious (class 2) content, ensuring high recall for threats while minimizing false alarms.

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
5. **Select the best model** using multi-criteria optimization with custom security-focused scoring
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
│   ├── metrics.py             # Custom security-focused metrics
│   └── focal_loss.py          # Focal loss implementations
├── logs/                      # Experiment logs
└── requirements.txt           # Dependencies
```

## 📊 Model Categories

### 1. **Traditional Machine Learning**
- **Random Forest**: Ensemble tree-based classification
- **Logistic Regression**: Linear classification with regularization
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
- **DistilXLM-RoBERTa**: Cross-lingual distilled transformer
- **XLM-RoBERTa**: Cross-lingual transformer models
- **XLM-RoBERTa-Large**: Large cross-lingual transformer
- **BERT**: Bidirectional Encoder Representations from Transformers
- **DistilBERT**: Distilled version of BERT (40% smaller, 60% faster)
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ALBERT**: A Lite BERT (very lightweight, 45MB)
- **DistilRoBERTa**: Distilled version of RoBERTa

## 🔒 Security-Focused Custom Metrics

### **Unified Scoring System Across All Pipelines**

All three training pipelines (Traditional ML, Deep Learning, Transformers) use the **same custom scoring system** defined in `utils/metrics.py`. This ensures consistent evaluation and model selection across different architectures.

### **Custom Scoring Formula**

```python
# Base score components (weighted by security importance)
base_score = 0.5 * recall_c2 + 0.3 * precision_c1 + 0.2 * f1_c0

# Critical misclassification penalty (exponential)
critical_error_rate = P(0→2) + P(2→0)  # benign→malicious + malicious→benign
penalty_factor = exp(-5 * critical_error_rate)

# Final custom score
custom_score = base_score * penalty_factor
```

### **Deep Scoring Explanation**

#### **1. Base Score Components (Weighted by Security Priority)**

- **`recall_c2` (50% weight)**: **Malicious threat detection rate**
  - Measures how many actual malicious prompts are caught
  - **Critical for security**: Missing threats is unacceptable
  - Target: >0.9 (90% of malicious content detected)

- **`precision_c1` (30% weight)**: **Suspicious classification precision**
  - Measures how many suspicious predictions are actually suspicious
  - **Important for analyst efficiency**: Reduces false alarms to suspicious
  - Target: >0.7 (70% of suspicious predictions are correct)

- **`f1_c0` (20% weight)**: **Benign classification balance**
  - Harmonic mean of precision and recall for benign content
  - **Maintains user experience**: Ensures legitimate prompts aren't blocked
  - Target: >0.8 (balanced performance on benign content)

#### **2. Critical Error Penalty (Exponential Scaling)**

The exponential penalty `exp(-5 * critical_error_rate)` creates a **catastrophic failure mode** for critical misclassifications:

- **`false_alarm_rate` (0→2)**: Benign prompts classified as malicious
  - **Business impact**: Blocks legitimate users, damages trust
  - **Penalty scaling**: Even 2% false alarms reduce score by ~90%

- **`missed_threat_rate` (2→0)**: Malicious prompts classified as benign
  - **Security impact**: Allows threats through, major security failure
  - **Penalty scaling**: Even 1% missed threats reduce score by ~95%

#### **3. Penalty Mathematics**

```python
# Example penalty calculations:
critical_error_rate = 0.01  # 1% critical errors
penalty_factor = exp(-5 * 0.01) = exp(-0.05) ≈ 0.951
# Score reduced by ~5%

critical_error_rate = 0.05  # 5% critical errors  
penalty_factor = exp(-5 * 0.05) = exp(-0.25) ≈ 0.779
# Score reduced by ~22%

critical_error_rate = 0.10  # 10% critical errors
penalty_factor = exp(-5 * 0.10) = exp(-0.50) ≈ 0.607
# Score reduced by ~39%
```

### **Pipeline-Specific Implementation**

#### **Traditional ML Pipeline (`trainer.py`)**
- **GridSearchCV**: Uses `sklearn_custom_scorer` for hyperparameter optimization
- **Cross-validation**: 4-fold CV with custom scoring on each fold
- **Model selection**: Best model chosen based on custom score, not F1-macro
- **Scoring mechanism**: `sklearn_custom_scorer` wraps our custom function for scikit-learn compatibility

#### **Deep Learning Pipeline (`adaptive_trainer.py`)**
- **Keras training**: Custom metrics computed during training via `CustomMetricsCallback`
- **Early stopping**: Based on `val_custom_score` (security-focused)
- **Learning rate scheduling**: Based on `val_custom_score` improvement
- **Model checkpointing**: Saves best model based on `val_custom_score`
- **Adaptive training**: Learning rate adjustments based on custom score dynamics

#### **Transformer Pipeline (`transformer_trainer.py`)**
- **HuggingFace Trainer**: Custom metrics computed during training via `compute_metrics` function
- **Cross-validation**: 3-fold CV (configurable) with custom scoring
- **Model selection**: Best fold chosen based on custom score
- **Early stopping**: Based on `eval_custom_score` (security-focused)
- **Training optimization**: Monitors custom score during training for early stopping decisions

### **`sklearn_custom_scorer` Deep Dive**

The `sklearn_custom_scorer` is a critical component that bridges our security-focused scoring system with scikit-learn's optimization framework:

```python
def _custom_score_func(y_true, y_pred):
    return compute_custom_metrics(y_true, y_pred)["custom_score"]

sklearn_custom_scorer = make_scorer(_custom_score_func, greater_is_better=True)
```

**How it works:**
1. **Function Wrapper**: `make_scorer()` converts our custom function into a scikit-learn compatible scorer
2. **Optimization Direction**: `greater_is_better=True` tells GridSearchCV that higher scores are better
3. **Cross-Validation Integration**: GridSearchCV calls this scorer on each fold during CV
4. **Hyperparameter Selection**: Best hyperparameters are chosen based on **custom score**, not F1-macro
5. **Consistent Evaluation**: Ensures all traditional ML models are optimized for security objectives

**Benefits:**
- **Security-First Optimization**: Models are tuned specifically for threat detection and false alarm reduction
- **Unified Scoring**: Same scoring system across all model types
- **Business Alignment**: Optimization directly reflects security priorities
- **Critical Error Penalization**: Exponential penalties ensure models avoid catastrophic misclassifications

## 🎯 Adaptive Weight Controller

### **Dynamic Class Weight Adjustment**

The framework includes an **Adaptive Weight Controller** that dynamically adjusts class weights during training based on metric performance. This ensures models continuously adapt to meet security objectives.

### **How It Works**

1. **Monitors Key Metrics**: Tracks recall_c2, precision_c1, f1_c0, false_alarm_rate, and missed_threat_rate
2. **Detects Off-Target Performance**: When metrics fall below targets for a patience period, triggers adaptation
3. **Adjusts Class Weights**: Increases weights for underperforming classes to focus training
4. **Updates Loss Functions**: Modifies focal loss alpha values (Keras) or class weights (HuggingFace) in real-time

### **Configuration**

Configure in `config.json` under `adaptive_objectives`:

```json
"adaptive_objectives": {
  "metric_weights": {
    "recall_c2": 0.50,      // Weight for malicious recall
    "precision_c1": 0.30,   // Weight for suspicious precision
    "f1_c0": 0.20,         // Weight for benign F1
    "critical_penalty": 1.00 // Multiplicative penalty weight
  },
  "targets": {
    "recall_c2": 0.95,      // Target 95% recall for malicious
    "precision_c1": 0.80,   // Target 80% precision for suspicious
    "f1_c0": 0.75,         // Target 75% F1 for benign
    "false_alarm_rate": 0.02,  // Max 2% benign→malicious errors
    "missed_threat_rate": 0.02  // Max 2% malicious→benign errors
  },
  "adaptation": {
    "alpha_step": 0.1,      // Weight adjustment step size
    "alpha_min": 0.5,       // Minimum class weight
    "alpha_max": 3.0,       // Maximum class weight
    "patience": 2           // Epochs before triggering adaptation
  }
}
```

### **Integration Points**

#### **Deep Learning (Keras)**
- **Callback**: `CustomMetricsCallback` computes metrics and updates weights
- **Loss Update**: Recompiles model with new focal loss alphas when weights change
- **Monitoring**: Tracks validation metrics each epoch

#### **Transformers (HuggingFace)**
- **Callback**: `DynamicClassWeightCallback` adjusts trainer weights
- **Loss Update**: Updates `class_weights_tensor` in `WeightedTrainer`
- **Evaluation**: Triggers on evaluation steps during training

### **Output Files**

Each model saves its weight adaptation history:
- **Location**: `results/alpha_trajectory_{model_name}.json`
- **Contents**: Initial weights, final weights, change history, metric trends
- **CSV Column**: `final_alpha` in `results.csv` shows final adapted weights

### **Example Adaptation Scenario**

```
Epoch 1: recall_c2=0.85 (below target 0.95)
Epoch 2: recall_c2=0.87 (still below target)
→ Patience exceeded, increasing alpha[2] from 1.0 to 1.1
Epoch 3: recall_c2=0.92 (improving)
Epoch 4: recall_c2=0.96 (target achieved!)
```

## 📊 Results and Metrics

### **Results CSV Structure**
```
results/results.csv columns:
├── model                    # Model name
├── precision_macro          # Overall precision
├── recall_macro            # Overall recall  
├── f1_macro               # Overall F1 score
├── roc_auc                # ROC AUC score
├── best_params            # Best hyperparameters
├── status                 # Training status
├── f1_external           # External blind-set F1
├── precision_c1          # Precision for suspicious class
├── recall_c2             # Recall for malicious class
├── f1_c0                 # F1 for benign class
├── custom_score          # Security-focused composite score
├── false_alarm_rate      # Rate of benign→malicious errors
├── missed_threat_rate    # Rate of malicious→benign errors
├── final_alpha           # Final adapted class weights [c0, c1, c2]
└── per_class_f1          # Per-class F1 scores
```

### **Key Metrics to Monitor**
- **`custom_score`**: Primary metric for model selection (higher is better)
- **`recall_c2`**: Malicious threat detection rate (target: >0.9)
- **`false_alarm_rate`**: Benign→malicious errors (target: <0.01)
- **`missed_threat_rate`**: Malicious→benign errors (target: <0.05)
- **`precision_c1`**: Suspicious classification precision (target: >0.7)
- **`f1_external`**: Performance on unseen blind-set (target: >0.6)

### **Model Selection Criteria**
1. **Security Performance**: `custom_score` > 0.6, `recall_c2` > 0.85
2. **False Alarm Control**: `false_alarm_rate` < 0.02
3. **Threat Detection**: `missed_threat_rate` < 0.1
4. **Efficiency**: Latency < 100ms, memory < 2GB
5. **Stability**: Cross-validation std < 0.05

## 🔧 Configuration

### **Current Multi-Pipeline Configuration**

#### **Traditional ML Models**
```json
{
  "traditional_ml": {
    "models": [
      "random_forest",      
      "logistic_regression",
      "svm",
      "naive_bayes",
      "xgboost"
    ],
    "cv_folds": 4,
    "custom_metric": true
  }
}
```

#### **Deep Learning Models**
```json
{
  "deep_learning": {
    "models": [
      "cnn",
      "lstm",
      "bilstm", 
      "transformer",
      "hybrid"
    ],
    "max_epochs": 25,
    "loss_function": "focal",
    "focal_gamma": 3.0,
    "focal_alpha": [1, 1, 1.5]
  }
}
```

#### **Transformer Models**
```json
{
  "transformers": {
    "models": [
      "distilxlm_roberta",
      "xlm_roberta", 
      "xlm_roberta_large"
    ],
    "max_epochs": 15,
    "cv_folds": 3,
    "loss_function": "focal",
    "focal_gamma": 3.0,
    "focal_alpha": [1, 1, 1.5]
  }
}
```

### **Security-Focused Configuration**

#### **Class Weights and Loss Functions**
```json
{
  "use_class_weights": true,
  "class_weights": [1, 1, 1.5],  // [benign, suspicious, malicious]
  "deep_learning": {
    "loss_function": "focal",
    "focal_gamma": 3.0,
    "focal_alpha": [1, 1, 1.5]
  },
  "transformers": {
    "loss_function": "focal", 
    "focal_gamma": 3.0,
    "focal_alpha": [1, 1, 1.5]
  }
}
```

#### **Custom Metrics Configuration**
```json
{
  "custom_metric": true,  // Enable custom scoring for model selection
  "external_validation": {
    "enable": true,
    "data_path": "validation_jsons/outside_test_data.json"
  }
}
```

### **Production-Grade Configuration Features**

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

## 🚀 Usage Examples

### **Basic Usage**
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
├── results.csv              # Complete experiment results with custom metrics
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

#### **Poor Security Performance**
- Check class balance in training data
- Verify external validation set quality
- Consider adjusting class weights or focal loss parameters
- Review false alarm vs missed threat rates separately

#### **Transformer-Specific Issues**
- Ensure MLflow is uninstalled (causes conflicts)
- Use smaller batch sizes for M1/M2 Macs
- Monitor memory usage during training
- **Large checkpoints (e.g. `xlm_roberta_large`, `roberta_large`) may be killed by the OS if RAM/VRAM is insufficient.**  Symptoms: training starts, then the shell prints `zsh: killed` (macOS) or the process exits with no Python traceback.  Mitigations:
  ```jsonc
  // In config.json → "transformers"
  "batch_size": 2,              // 1-4 on CPU, 8-16 on GPU
  "max_length": 256,            // shorter sequences cut memory >40 %
  "fp16": true,                // only on GPU/CUDA
  "extra_args": {
    "gradient_checkpointing": true  // enables activation checkpointing
  }
  ```
  If you are CPU-only, expect epoch times of several hours even after the above tweaks; consider switching to the `xlm_roberta` *base* model instead.

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
6. **Security Focus**: Maintain custom metrics and penalty system

### **Code Quality**
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive error handling
- Write unit tests for new features
- Update configuration documentation
- Ensure custom metrics are computed for all model types

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.