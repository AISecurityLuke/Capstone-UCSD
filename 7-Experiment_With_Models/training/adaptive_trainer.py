#!/usr/bin/env python3
"""
Adaptive deep learning trainer with real-time monitoring and automatic adaptation
"""

import os
import json
import logging
import numpy as np
import sys
import time
import csv
from typing import Dict, List, Any, Tuple

# ------------------------------------------------------------
# Optional detailed logging controlled by env var
# ------------------------------------------------------------
if os.environ.get("DETAILED_LOGGING", "false").lower() == "true":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("[adaptive_trainer] Detailed logging enabled.")

# ------------------------------------------------------------------
# Early Config load for environment policy (GPU / mixed precision)
# ------------------------------------------------------------------
from config.config import load_config
CONFIG_GLOBAL = load_config()
# Disable GPU if requested BEFORE tensorflow import
if not CONFIG_GLOBAL.get('deep_learning', {}).get('enable_gpu', True):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Only now import heavy libs (tensorflow, torch…)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Mixed precision
if CONFIG_GLOBAL.get('deep_learning', {}).get('mixed_precision', False):
    try:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('mixed_float16')
        logging.info("Mixed precision policy 'mixed_float16' enabled.")
    except Exception as _e:
        logging.warning(f"Could not enable mixed_precision: {_e}")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from models.utils.model_manager import model_manager
from models.keras_models import create_model
from utils.monitor import monitor
import pickle

# Ensure report dir
os.makedirs(os.path.join('results', 'reports'), exist_ok=True)

# ------------------------------------------------------------------
# Mixed-precision & GPU policy (read config after load_config)
# ------------------------------------------------------------------

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        logging.warning(f"GPU configuration failed: {e}")
else:
    logging.info("Using CPU for training")

class AdaptiveTrainingController:
    """Controller for adaptive training with real-time issue detection and recovery"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_history = []
        self.adaptation_history = []
        self.learning_rate = config.get('deep_learning', {}).get('learning_rate', 0.001)
        self.original_lr = self.learning_rate

        # Thresholds for issue detection – overridable via config['adaptive_thresholds']
        default_thresh = {
            'loss_jump': 1.0,          # gradient explosion
            'loss_trend': 0.001,       # gradient vanishing
            'train_val_gap': 0.5,      # overfitting
            'loss_high': 10.0,         # loss explosion
            'loss_std': 0.01           # stagnation
        }
        self.thresholds = {**default_thresh, **config.get('adaptive_thresholds', {})}
        
    def analyze_training_state(self) -> List[str]:
        """Analyze current training state and detect issues"""
        if len(self.training_history) < 2:
            return ['insufficient_data']
        
        issues = []
        recent_losses = [h['loss'] for h in self.training_history[-5:]]
        recent_val_losses = [h['val_loss'] for h in self.training_history[-5:]]
        
        # Check for gradient explosion
        if len(recent_losses) >= 2:
            loss_change = abs(recent_losses[-1] - recent_losses[-2])
            if loss_change > self.thresholds['loss_jump']:
                issues.append('gradient_explosion')
        
        # Check for gradient vanishing
        if len(recent_losses) >= 3:
            loss_trend = np.mean(np.diff(recent_losses[-3:]))
            if abs(loss_trend) < self.thresholds['loss_trend']:
                issues.append('gradient_vanishing')
        
        # Check for overfitting
        if len(recent_val_losses) >= 3:
            train_val_gap = np.mean(recent_losses[-3:]) - np.mean(recent_val_losses[-3:])
            if train_val_gap > self.thresholds['train_val_gap']:
                issues.append('overfitting')
        
        # Check for loss explosion
        if recent_losses and recent_losses[-1] > self.thresholds['loss_high']:
            issues.append('loss_explosion')
        
        # Check for loss stagnation
        if len(recent_losses) >= 5:
            loss_std = np.std(recent_losses[-5:])
            if loss_std < self.thresholds['loss_std']:
                issues.append('loss_stagnation')
        
        return issues
    
    def adapt_training(self, issues: List[str]) -> Dict[str, Any]:
        """Apply adaptive changes based on detected issues"""
        changes = {}
        
        for issue in issues:
            if issue == 'gradient_explosion':
                changes['gradient_clip'] = 1.0
                changes['learning_rate'] = self.learning_rate * 0.5
                
            elif issue == 'gradient_vanishing':
                changes['learning_rate'] = self.learning_rate * 2.0
                changes['batch_norm'] = True
                
            elif issue == 'overfitting':
                changes['dropout'] = 0.5
                changes['regularization'] = 0.01
                
            elif issue == 'loss_explosion':
                changes['learning_rate'] = self.learning_rate * 0.1
                changes['gradient_clip'] = 0.5
                
            elif issue == 'loss_stagnation':
                changes['learning_rate'] = self.learning_rate * 1.5
        
        if changes:
            self.adaptation_history.append({
                'epoch': len(self.training_history),
                'issues': issues,
                'changes': changes.copy()
            })
            
            if 'learning_rate' in changes:
                self.learning_rate = changes['learning_rate']
                
        return changes

class AdaptiveCallback(keras.callbacks.Callback):
    """Keras callback for adaptive training"""
    
    def __init__(self, controller: AdaptiveTrainingController):
        super().__init__()
        self.controller = controller
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        if logs is None:
            logs = {}
            
        # Record training state
        self.controller.training_history.append({
            'epoch': epoch,
            'loss': logs.get('loss', 0),
            'val_loss': logs.get('val_loss', 0),
            'accuracy': logs.get('accuracy', 0),
            'val_accuracy': logs.get('val_accuracy', 0)
        })
        
        # Analyze and adapt
        issues = self.controller.analyze_training_state()
        if issues and issues != ['insufficient_data']:
            changes = self.controller.adapt_training(issues)
            
            if changes:
                logging.info(f"Adaptive training applied: {', '.join(issues)}")
                
                # Apply learning rate change
                if 'learning_rate' in changes:
                    new_lr = changes['learning_rate']
                    keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    logging.info(f"Learning rate adjusted to: {new_lr}")

def prepare_text_data(X_train, X_test, y_train, config, vocab_size=10000, default_max_length=100):
    """Prepare text data for deep learning models (returns integer token sequences)"""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Determine max_length from config if provided
    max_length = config.get('preprocessing', {}).get('tokenization', {}).get('max_length', default_max_length)
    
    # Get global embedding config if available
    embedding_config = config.get('preprocessing', {}).get('embeddings', {})
    global_vocab_size = embedding_config.get('vocab_size', vocab_size)
    global_embedding_dim = embedding_config.get('embedding_dim', 128)
    
    # Use global vocab_size if larger than the passed parameter
    effective_vocab_size = max(vocab_size, global_vocab_size)
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=effective_vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences (final shape: (batch, seq_len))
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Prepare data
    use_custom_weights = config.get('use_class_weights', False)
    custom_weights = config.get('class_weights', [])
    num_classes = len(np.unique(y_train))

    if use_custom_weights:
        if len(custom_weights) != num_classes:
            raise ValueError("Length of class_weights does not match number of classes")
        class_weights = np.array(custom_weights)
        logging.info(f"[AdaptiveTrainer] Using custom class weights from config: {class_weights}")
    else:
        class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train)
        logging.info(f"[AdaptiveTrainer] Using automatically computed class weights: {class_weights}")

    class_weight_dict = {cls: class_weights[idx] for idx, cls in enumerate(np.arange(num_classes))}
    
    return X_train_padded, X_test_padded, tokenizer, class_weight_dict, max_length, effective_vocab_size, global_embedding_dim

def train_adaptive_model(model, X_train, y_train, X_test, y_test, model_name, config, arch_config=None, ext_data=None):
    """Train a Keras model with adaptive training and return results"""
    logging.info(f"Training adaptive model: {model_name}")
    
    vocab_size_param = arch_config.get('vocab_size', 10000) if arch_config else 10000
    # Prepare data
    X_train_padded, X_test_padded, tokenizer, class_weight_dict, seq_len, effective_vocab_size, global_embedding_dim = prepare_text_data(
        X_train, X_test, y_train, config, vocab_size=vocab_size_param)
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train)
    y_test_cat = tf.keras.utils.to_categorical(y_test)
    
    # Setup adaptive training controller
    controller = AdaptiveTrainingController(config)
    
    # --------------------------------------------------------------
    # Optimizer / learning-rate handling
    # --------------------------------------------------------------
    # Priority order: (1) architecture-specific LR, (2) global deep_learning LR, (3) 1e-3 default
    model_config = config.get('model_architectures', {}).get(model_name.split('_')[0], {}).get('default_config', {})
    arch_lr = model_config.get('learning_rate')
    global_lr = config.get('deep_learning', {}).get('learning_rate')
    learning_rate = arch_lr if arch_lr is not None else (global_lr if global_lr is not None else 0.001)

    # Compile with plain string 'adam' for maximum compatibility, then set lr & clipnorm directly
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Override optimizer hyperparams after compile
    opt_obj = model.optimizer
    keras.backend.set_value(opt_obj.learning_rate, learning_rate)
    opt_obj.clipnorm = 1.0

    # Ensure the adaptive controller starts with the same LR baseline
    controller.learning_rate = learning_rate

    # Setup callbacks
    callbacks = []

    # --------------------------------------------------------------
    # F1 metric callback so that 'val_f1_macro' exists for monitoring
    # --------------------------------------------------------------
    class F1MetricsCallback(keras.callbacks.Callback):
        """Compute macro-F1 on validation set at each epoch end."""
        def __init__(self, X_val, y_val):
            super().__init__()
            self.X_val = X_val
            self.y_val = y_val

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
            f1_macro = f1_score(self.y_val, y_pred, average='macro')
            logs['val_f1_macro'] = f1_macro
            print(f"Epoch {epoch+1}: val_f1_macro={f1_macro:.4f}")

    callbacks.append(F1MetricsCallback(X_test_padded, y_test))

    # Early stopping (now sees val_f1_macro)
    early_stopping_config = config.get('training_callbacks', {}).get('early_stopping', {})
    callbacks.append(EarlyStopping(
        patience=early_stopping_config.get('patience', 5),
        restore_best_weights=early_stopping_config.get('restore_best_weights', True),
        monitor=early_stopping_config.get('monitor', 'val_f1_macro'),
        mode='max',
        min_delta=early_stopping_config.get('min_delta', 0.0)
    ))

    # Learning rate scheduler
    lr_config = config.get('training_callbacks', {}).get('learning_rate_scheduler', {})
    callbacks.append(ReduceLROnPlateau(
        monitor=lr_config.get('monitor', 'val_f1_macro'),
        mode='max',
        factor=lr_config.get('factor', 0.5),
        patience=lr_config.get('patience', 3),
        min_lr=lr_config.get('min_lr', 1e-7)
    ))
    
    # Model checkpoint
    checkpoint_config = config.get('training_callbacks', {}).get('model_checkpoint', {})
    callbacks.append(ModelCheckpoint(
        f'models/{model_name}/best_model.keras',
        save_best_only=checkpoint_config.get('save_best_only', True),
        monitor=checkpoint_config.get('monitor', 'val_f1_macro'),
        mode='max'
    ))
    
    # Adaptive callback
    callbacks.append(AdaptiveCallback(controller))
    
    # Train model – optionally switch to tf.data for very large datasets to save RAM
    max_epochs = config.get('deep_learning', {}).get('max_epochs', 3)
    batch_size = config.get('deep_learning', {}).get('batch_size', 32)

    threshold = config.get('deep_learning', {}).get('tfdata_threshold', 100000)
    use_tfdata = len(X_train_padded) >= threshold

    if use_tfdata:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_cat)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds   = tf.data.Dataset.from_tensor_slices((X_test_padded,  y_test_cat)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict if class_weight_dict else None,
            verbose=1
        )
    else:
        history = model.fit(
            X_train_padded, y_train_cat,
            validation_data=(X_test_padded, y_test_cat),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
    
    # Evaluate model on hold-out
    y_pred_proba = model.predict(X_test_padded)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    # Debug: Log the discrepancy between training val_f1_macro and final evaluation
    if 'val_f1_macro' in history.history:
        best_val_f1 = max(history.history['val_f1_macro'])
        logging.info(f"[{model_name}] Training best val_f1_macro: {best_val_f1:.4f}")
        logging.info(f"[{model_name}] Final evaluation f1_macro: {f1:.4f}")
        if abs(best_val_f1 - f1) > 0.1:
            logging.warning(f"[{model_name}] Large discrepancy detected: {best_val_f1:.4f} vs {f1:.4f}")
    # ROC-AUC (macro, OVR)
    try:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        probabilities = softmax(y_pred_proba, axis=1)
        roc_auc = roc_auc_score(y_test_binarized, probabilities, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = float('nan')
    accuracy = accuracy_score(y_test, y_pred)
    
    # External / blind-set evaluation
    f1_ext = float('nan')
    if ext_data:
        try:
            # preprocess ext texts with same tokenizer and padding
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seqs_ext = tokenizer.texts_to_sequences(ext_data['X_ext'])
            X_ext_pad = pad_sequences(seqs_ext, maxlen=seq_len, padding='post', truncating='post')
            y_ext_pred = np.argmax(model.predict(X_ext_pad, verbose=0), axis=1)
            _, _, f1_ext, _ = precision_recall_fscore_support(ext_data['y_ext'], y_ext_pred, average='macro')
        except Exception as e:
            logging.warning(f"{model_name}: external validation failed – {e}")
    
    # ----------------------
    # Save per-class report
    # ----------------------
    os.makedirs(os.path.join('results', 'reports'), exist_ok=True)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join('results', 'reports', f"{model_name}.json"), 'w') as fp:
        json.dump(report, fp, indent=2)
    
    # Save tokenizer
    tokenizer_path = os.path.join('models', model_name, 'tokenizer.pkl')
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1_ext': f1_ext,
        'history': history.history,
        'adaptation_history': controller.adaptation_history,
        'architecture_config': arch_config
    }

def train_and_evaluate_deep_models(data_dict, config, ext_data=None):
    """Train and evaluate all deep learning models (Keras)"""
    logging.info("Starting adaptive Keras models...")
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    num_classes = data_dict['num_classes']
    # sequence length used for Keras input shape
    seq_len_global = config.get('preprocessing', {}).get('tokenization', {}).get('max_length', 100)
    
    # Get global embedding configuration
    embedding_config = config.get('preprocessing', {}).get('embeddings', {})
    effective_vocab_size = embedding_config.get('vocab_size', 20000)
    global_embedding_dim = embedding_config.get('embedding_dim', 256)
    
    # Get models to train
    models_to_train = config.get('deep_learning', {}).get('models', [])
    
    successful_models = []
    failed_models = []
    
    # Results file path
    results_path = os.path.join('results', 'results.csv')
    
    transformer_types = {'bert', 'distilbert', 'roberta', 'albert', 'distilroberta'}

    for model_type in models_to_train:
        if model_type in transformer_types:
            try:
                result = train_hf_transformer_model(model_type, X_train, y_train, X_test, y_test, num_classes, config, data_dict)
                with open(results_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        model_type,
                        f"{result['precision']:.4f}",
                        f"{result['recall']:.4f}",
                        f"{result['f1']:.4f}",
                        f"{result['roc_auc']:.4f}",
                        'hf',
                        'success',
                        f"{result['f1_ext']:.4f}"
                    ])
                successful_models.append(model_type)
                logging.info(f"{model_type} completed - F1: {result['f1']:.4f}")
            except Exception as e:
                logging.error(f"{model_type} failed: {e}")
                failed_models.append(model_type)
            continue  # Skip keras path for transformer models

        try:
            logging.info(f"Starting adaptive Keras model: {model_type}")
            
            # Get architecture configurations
            arch_configs = config.get('model_architectures', {}).get(model_type, {})
            default_config = arch_configs.get('default_config', {})
            search_space = arch_configs.get('search_space', {})
            
            if not config.get('architecture_search', True):
                search_space = {}
            
            # ------------------------------------------------------------------
            # Optuna hyper-parameter optimisation (joint search)
            # ------------------------------------------------------------------
            enable_optuna = config.get('hyperparameter_optimization', {}).get('enable_optuna', False)
            if enable_optuna and search_space:
                try:
                    import optuna
                except ImportError:
                    logging.warning("Optuna not installed – falling back to simple architecture sweep")
                    enable_optuna = False

            if enable_optuna and search_space:
                n_trials = config.get('hyperparameter_optimization', {}).get('num_trials', 30)
                timeout  = config.get('hyperparameter_optimization', {}).get('timeout', 1800)

                input_shape = (seq_len_global,)

                def objective(trial):
                    trial_config = default_config.copy()
                    # Sample each parameter from its candidate list
                    for pname, pvalues in search_space.items():
                        trial_config[pname] = trial.suggest_categorical(pname, pvalues)
                    
                    # Add embedding parameters from global config
                    trial_config['vocab_size'] = effective_vocab_size
                    trial_config['embedding_dim'] = global_embedding_dim

                    model_trial = create_model(model_type, input_shape, num_classes, trial_config)
                    try:
                        result_trial = train_adaptive_model(
                            model_trial, X_train, y_train, X_test, y_test,
                            f"{model_type}_optuna_trial", config, trial_config, ext_data
                        )
                        # Persist full result for later retrieval
                        trial.set_user_attr("result", result_trial)
                        # Use best val_f1_macro from training for consistency with checkpoint selection
                        if 'val_f1_macro' in result_trial.get('history', {}):
                            return max(result_trial['history']['val_f1_macro'])
                        else:
                            # Fallback to final holdout F1 if val_f1_macro not available
                            return result_trial['f1']
                    except Exception as exc:
                        logging.error(f"{model_type} Optuna trial failed: {exc}")
                        return 0.0  # poor score so Optuna discards it

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

                best_trial = study.best_trial
                best_result = best_trial.user_attrs.get('result') if best_trial else None

                if best_result:
                    # Write CSV row
                    with open(results_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            f"{model_type}_best",
                            f"{best_result['precision']:.4f}",
                            f"{best_result['recall']:.4f}",
                            f"{best_result['f1']:.4f}",
                            f"{best_result['roc_auc']:.4f}",
                            str(best_trial.params),
                            'success',
                            f"{best_result['f1_ext']:.4f}"
                        ])

                    # Save model
                    model_dir = os.path.join('models', f"{model_type}_best")
                    os.makedirs(model_dir, exist_ok=True)
                    best_result['model'].save(os.path.join(model_dir, 'model.keras'))

                    successful_models.append(f"{model_type}_best")
                    logging.info(f"{model_type} Optuna best F1={best_result['f1']:.4f} params={best_trial.params}")
                else:
                    logging.error(f"Optuna optimisation for {model_type} produced no valid result")
                    failed_models.append(model_type)

                continue  # Skip legacy one-parameter sweep when Optuna is used

            logging.info(f"Testing {len(search_space) + 1} different architectures for {model_type}")
            
            best_result = None
            best_f1 = 0
            
            # Test default configuration first
            try:
                input_shape = (seq_len_global,)
                # Add embedding parameters from global config
                default_config_with_embeddings = default_config.copy()
                default_config_with_embeddings['vocab_size'] = effective_vocab_size
                default_config_with_embeddings['embedding_dim'] = global_embedding_dim
                
                model = create_model(model_type, input_shape, num_classes, default_config_with_embeddings)
                
                result = train_adaptive_model(
                    model, X_train, y_train, X_test, y_test,
                    f"{model_type}_default", config, default_config, ext_data
                )
                
                if result and result['f1'] > best_f1:
                    best_result = result
                    best_f1 = result['f1']
                    
            except Exception as e:
                logging.error(f"{model_type} default config failed: {e}")
            
            # Test search space configurations
            for i, (param_name, param_values) in enumerate(search_space.items()):
                if not config.get('architecture_search', True):
                    break
                try:
                    logging.info(f"Testing architecture {i+1}/{len(search_space)} for {model_type}")
                    
                    # Create config with this parameter
                    test_config = default_config.copy()
                    test_config[param_name] = param_values[0] if isinstance(param_values, list) else param_values
                    
                    # Add embedding parameters from global config
                    test_config['vocab_size'] = effective_vocab_size
                    test_config['embedding_dim'] = global_embedding_dim
                    
                    input_shape = (seq_len_global,)
                    model = create_model(model_type, input_shape, num_classes, test_config)
                    
                    result = train_adaptive_model(
                        model, X_train, y_train, X_test, y_test,
                        f"{model_type}_arch_{i+1}", config, test_config, ext_data
                    )
                    
                    if result and result['f1'] > best_f1:
                        best_result = result
                        best_f1 = result['f1']
                        
                except Exception as e:
                    logging.error(f"{model_type} architecture {i+1} failed: {e}")
                    continue
            
            if best_result:
                # Save best model results
                with open(results_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{model_type}_best",
                        f"{best_result['precision']:.4f}",
                        f"{best_result['recall']:.4f}",
                        f"{best_result['f1']:.4f}",
                        f"{best_result['roc_auc']:.4f}",
                        str(best_result.get('architecture_config', 'keras_adaptive')),
                        'success',
                        f"{best_result['f1_ext']:.4f}"
                    ])
                
                # Save best model
                model_dir = os.path.join('models', f"{model_type}_best")
                os.makedirs(model_dir, exist_ok=True)
                best_result['model'].save(os.path.join(model_dir, 'model.keras'))
                
                successful_models.append(f"{model_type}_best")
                logging.info(f"Best architecture: {best_result.get('architecture_config', 'N/A')}")
                
            else:
                logging.error(f"All architectures failed for {model_type}")
                failed_models.append(model_type)
                
        except Exception as e:
            logging.error(f"{model_type} failed: {e}")
            failed_models.append(model_type)
    
    # Summary
    logging.info(f"Deep Learning Summary: {len(successful_models)} successful, {len(failed_models)} failed")
    
    return successful_models, failed_models 

class TextClassificationDataset(Dataset):
    """Simple torch dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    """Compute metrics function for HuggingFace Trainer (uses macro F1)"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"precision": precision, "recall": recall, "f1": f1}


def train_hf_transformer_model(model_key, X_train, y_train, X_test, y_test, num_classes, config, data_dict=None):
    """Train a HuggingFace transformer model for sequence classification."""
    logging.info(f"Training HuggingFace model: {model_key}")
    
    # Disable integrations to prevent conflicts
    import os
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name, tokenizer_cls, model_cls = model_manager.get_model_components(model_key)
    max_length = config.get('model_architectures', {}).get(model_key, {}).get('default_config', {}).get('max_length', 256)

    # DEBUG: Print dataset info with filename
    print("[adaptive_trainer.py] X_train len:", len(X_train))
    print("[adaptive_trainer.py] Unique classes in y_train:", set(y_train))
    print("[adaptive_trainer.py] Class counts:", {c: list(y_train).count(c) for c in set(y_train)})
    print("[adaptive_trainer.py] First 10 y_train:", list(y_train)[:10])
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_length)
    print(f"[adaptive_trainer.py] {model_key} train dataset size: {len(train_dataset)} samples (should match len(X_train))")
    logging.info(f"{model_key} train dataset size: {len(train_dataset)} samples (should match len(X_train))")
    eval_dataset = TextClassificationDataset(X_test, y_test, tokenizer, max_length)

    model = model_cls.from_pretrained(model_name, num_labels=num_classes)

    # Determine class weights (custom or balanced)
    use_custom_weights = config.get('use_class_weights', False)
    custom_weights = config.get('class_weights', [])
    unique_classes = np.unique(y_train)

    if use_custom_weights and custom_weights and len(custom_weights) == len(unique_classes):
        class_weights = np.array(custom_weights)
        logging.info(f"[HFTrainer] Using custom class weights from config: {class_weights}")
    else:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        logging.info(f"[HFTrainer] Using automatically computed class weights: {class_weights}")

    class_weight_dict = dict(zip(unique_classes, class_weights))
    logging.info(f"Class weights for {model_key}: {class_weight_dict}")

    # ------------------------------------------------------------------
    # Optuna hyper-parameter optimisation (if enabled in config)
    # ------------------------------------------------------------------
    opt_cfg = config.get('hyperparameter_optimization', {})
    enable_optuna_tf = opt_cfg.get('enable_optuna', False)

    if enable_optuna_tf:
        try:
            import optuna
        except ImportError:
            logging.warning("Optuna not installed – skipping transformer HPO")
            enable_optuna_tf = False

    if enable_optuna_tf:
        n_trials = opt_cfg.get('num_trials', 20)
        timeout   = opt_cfg.get('timeout', 7200)

        # Convert weights tensor once for reuse
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        def objective(trial):
            # ---- sample hyper-parameters ----
            lr          = trial.suggest_float('learning_rate', 1e-5, 8e-5, log=True)
            weight_decay= trial.suggest_float('weight_decay', 0.0, 0.3)
            warmup_ratio= trial.suggest_float('warmup_ratio', 0.0, 0.2)
            dropout_val = trial.suggest_float('dropout', 0.0, 0.3)
            batch_size  = trial.suggest_categorical('batch_size', [8, 16, 32])
            epochs      = trial.suggest_int('epochs', 3, 5)

            # Model init function so Trainer creates fresh model each trial
            def model_init():
                return model_cls.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    hidden_dropout_prob=dropout_val,
                    attention_probs_dropout_prob=dropout_val,
                )

            args = TrainingArguments(
                output_dir=f"models/{model_key}_optuna_trial",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=min(batch_size, 32),
                evaluation_strategy="epoch",
                logging_strategy="no",
                save_strategy="no",
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                report_to=[],
                disable_tqdm=True,
                dataloader_pin_memory=False,
            )

            trainer_trial = WeightedTrainer(
                model_init=model_init,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                class_weights_tensor=class_weights_tensor,
            )

            # Early stopping (reuse patience from config)
            patience_tf = config.get('transformers', {}).get('early_stopping_patience', 3)
            trainer_trial.add_callback(EarlyStoppingCallback(early_stopping_patience=patience_tf))

            trainer_trial.train()
            eval_res = trainer_trial.evaluate()
            f1_score_val = eval_res.get('eval_f1') or eval_res.get('f1') or 0.0
            # Store trainer path for later retrieval if needed
            trial.set_user_attr('best_metrics', eval_res)
            return f1_score_val

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best_params = study.best_trial.params
        logging.info(f"[Optuna] {model_key} best params: {best_params} – F1={study.best_value:.4f}")

        # -----------------------------
        # Retrain best model to save it
        # -----------------------------
        final_bs   = best_params.get('batch_size')
        final_args = TrainingArguments(
            output_dir=f"models/{model_key}_hf_best",
            num_train_epochs=best_params.get('epochs'),
            per_device_train_batch_size=final_bs,
            per_device_eval_batch_size=min(final_bs, 32),
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            learning_rate=best_params.get('learning_rate'),
            weight_decay=best_params.get('weight_decay'),
            warmup_ratio=best_params.get('warmup_ratio'),
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
        )

        def best_model_init():
            return model_cls.from_pretrained(
                model_name,
                num_labels=num_classes,
                hidden_dropout_prob=best_params.get('dropout'),
                attention_probs_dropout_prob=best_params.get('dropout'),
            )

        best_trainer = WeightedTrainer(
            model_init=best_model_init,
            args=final_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            class_weights_tensor=class_weights_tensor,
        )

        best_trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))
        best_trainer.train()

        preds_best = best_trainer.predict(eval_dataset)
        y_pred_best = preds_best.predictions.argmax(axis=1)

        precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(y_test, y_pred_best, average='macro')
        try:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            probs_b = softmax(preds_best.predictions, axis=1)
            roc_auc_b = roc_auc_score(y_test_bin, probs_b, multi_class='ovr', average='macro')
        except Exception:
            roc_auc_b = float('nan')

        # Save final model & tokenizer
        model_dir = os.path.join('models', f"{model_key}_hf_best")
        os.makedirs(model_dir, exist_ok=True)
        best_trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        # Classification report
        from sklearn.metrics import classification_report
        os.makedirs(os.path.join('results', 'reports'), exist_ok=True)
        rpt = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
        with open(os.path.join('results', 'reports', f"{model_key}.json"), 'w') as fp:
            json.dump(rpt, fp, indent=2)

        return {
            'precision': precision_b,
            'recall': recall_b,
            'f1': f1_b,
            'roc_auc': roc_auc_b,
            'accuracy': accuracy_score(y_test, y_pred_best)
        }

    # ------------------------------------------------------------------
    # Fallback: single training run (legacy behaviour)
    # ------------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=f"models/{model_key}_hf",
        num_train_epochs=config.get('deep_learning', {}).get('max_epochs', 3),
        per_device_train_batch_size=config.get('deep_learning', {}).get('batch_size', 8),
        per_device_eval_batch_size=config.get('deep_learning', {}).get('batch_size', 8),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=config.get('model_architectures', {}).get(model_key, {}).get('default_config', {}).get('learning_rate', 2e-5),
        push_to_hub=False,
        report_to=[],
        dataloader_pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Trainer with optional class-weighted loss
    # ------------------------------------------------------------------
    class WeightedTrainer(Trainer):
        """HuggingFace Trainer that supports class-weighted loss"""

        def __init__(self, *args, class_weights_tensor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights_tensor = class_weights_tensor

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights_tensor is not None:
                if logits.device.type == "mps":
                    loss_fct = nn.CrossEntropyLoss()
                else:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights_tensor.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    # Convert weights to tensor on correct device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights_tensor=class_weights_tensor.to(model.device),
    )

    # Early stopping on F1
    patience = config.get('transformers', {}).get('early_stopping_patience', 3)
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=patience))

    # DEBUG: Print effective batch size, dataset length, and steps per epoch
    print(f"[DEBUG] Effective batch size: {training_args.per_device_train_batch_size}")
    print(f"[DEBUG] Length of train_dataset: {len(train_dataset)}")
    print(f"[DEBUG] Trainer's train_batch_size: {trainer.args.train_batch_size}")
    steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    print(f"[DEBUG] Calculated steps per epoch: {steps_per_epoch}")

    trainer.train()

    preds = trainer.predict(eval_dataset)
    y_pred = preds.predictions.argmax(axis=1)

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    # ROC-AUC (macro, OVR)
    try:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        probabilities = softmax(preds.predictions, axis=1)
        roc_auc = roc_auc_score(y_test_binarized, probabilities, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = float('nan')
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate per-class F1 scores
    class_f1_scores = precision_recall_fscore_support(y_test, y_pred, average=None)[2]
    class_f1_dict = {i: f1 for i, f1 in enumerate(class_f1_scores)}
    
    # Add class F1 scores to the results
    result['class_f1'] = class_f1_dict
    
    # ------------------------------------------------------------
    # Log misclassified examples (max 50 per class per model)
    # ------------------------------------------------------------
    misclassified_examples = {}
    for true_class in range(3):
        for pred_class in range(3):
            if true_class != pred_class:
                mask = (y_test == true_class) & (y_pred == pred_class)
                if mask.any():
                    misclassified_texts = [X_test[i] for i in np.where(mask)[0][:50]]  # Max 50 per confusion
                    key = f"true_{true_class}_predicted_{pred_class}"
                    misclassified_examples[key] = misclassified_texts
    
    # Save misclassified examples
    misclassified_path = os.path.join('results', 'results_misclassification.json')
    try:
        if os.path.exists(misclassified_path):
            with open(misclassified_path, 'r') as f:
                all_misclassified = json.load(f)
        else:
            all_misclassified = {}
        
        all_misclassified[model_name] = {
            "model_type": "deep_learning",
            "misclassified_examples": misclassified_examples
        }
        
        with open(misclassified_path, 'w') as f:
            json.dump(all_misclassified, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save misclassified examples for {model_name}: {e}")

    # Detailed classification report
    from sklearn.metrics import classification_report
    os.makedirs(os.path.join('results', 'reports'), exist_ok=True)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(os.path.join('results', 'reports', f"{model_key}.json"), 'w') as fp:
        json.dump(report, fp, indent=2)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1_ext': f1_ext
    } 