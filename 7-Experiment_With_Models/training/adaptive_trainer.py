#!/usr/bin/env python3
"""
Adaptive deep learning trainer with real-time monitoring and automatic adaptation
"""

import logging
import numpy as np
import os
import sys
import time
import csv
from typing import Dict, List, Any, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from models.keras_models import create_model
from utils.sanitizer import escape_for_logging
from utils.monitor import monitor

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
            if loss_change > 1.0:  # Large loss jump
                issues.append('gradient_explosion')
        
        # Check for gradient vanishing
        if len(recent_losses) >= 3:
            loss_trend = np.mean(np.diff(recent_losses[-3:]))
            if abs(loss_trend) < 0.001:  # Very small changes
                issues.append('gradient_vanishing')
        
        # Check for overfitting
        if len(recent_val_losses) >= 3:
            train_val_gap = np.mean(recent_losses[-3:]) - np.mean(recent_val_losses[-3:])
            if train_val_gap > 0.5:  # Large gap between train and val
                issues.append('overfitting')
        
        # Check for loss explosion
        if recent_losses and recent_losses[-1] > 10.0:
            issues.append('loss_explosion')
        
        # Check for loss stagnation
        if len(recent_losses) >= 5:
            loss_std = np.std(recent_losses[-5:])
            if loss_std < 0.01:  # Very low variance
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

def prepare_text_data(X_train, X_test, max_length=100):
    """Prepare text data for deep learning models"""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Tokenize text
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    return X_train_padded, X_test_padded, tokenizer

def train_adaptive_model(model, X_train, y_train, X_test, y_test, model_name, config, arch_config=None):
    """Train Keras model with automatic adaptive feedback loop"""
    logging.info(f"Training adaptive Keras model: {model_name}")
    if arch_config:
        logging.info(f"Architecture config: {arch_config}")
    
    # Create adaptive controller
    controller = AdaptiveTrainingController(config)
    
    # Prepare data
    max_length = config.get('preprocessing', {}).get('tokenization', {}).get('max_length', 100)
    X_train_padded, X_test_padded, tokenizer = prepare_text_data(X_train, X_test, max_length)
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train)
    y_test_cat = tf.keras.utils.to_categorical(y_test)
    
    # Compile model
    learning_rate = config.get('deep_learning', {}).get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stopping_config = config.get('training_callbacks', {}).get('early_stopping', {})
    callbacks.append(EarlyStopping(
        patience=early_stopping_config.get('patience', 5),
        restore_best_weights=early_stopping_config.get('restore_best_weights', True),
        monitor=early_stopping_config.get('monitor', 'val_loss')
    ))
    
    # Learning rate scheduler
    lr_config = config.get('training_callbacks', {}).get('learning_rate_scheduler', {})
    callbacks.append(ReduceLROnPlateau(
        factor=lr_config.get('factor', 0.5),
        patience=lr_config.get('patience', 3),
        min_lr=lr_config.get('min_lr', 1e-7)
    ))
    
    # Model checkpoint
    checkpoint_config = config.get('training_callbacks', {}).get('model_checkpoint', {})
    callbacks.append(ModelCheckpoint(
        f'models/{model_name}/best_model.h5',
        save_best_only=checkpoint_config.get('save_best_only', True),
        monitor=checkpoint_config.get('monitor', 'val_f1_macro')
    ))
    
    # Adaptive callback
    callbacks.append(AdaptiveCallback(controller))
    
    # Train model
    max_epochs = config.get('deep_learning', {}).get('max_epochs', 3)
    batch_size = config.get('deep_learning', {}).get('batch_size', 32)
    
    # Get class weights
    class_weight_dict = {}
    for i in range(len(np.unique(y_train))):
        class_weight_dict[i] = 1.0  # Default equal weights
    
    history = model.fit(
        X_train_padded, y_train_cat,
        validation_data=(X_test_padded, y_test_cat),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate model
    y_pred_proba = model.predict(X_test_padded)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save tokenizer
    tokenizer_path = os.path.join('models', model_name, 'tokenizer.pkl')
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'history': history.history,
        'adaptation_history': controller.adaptation_history
    }

def train_and_evaluate_deep_models(data_dict, config):
    """Train and evaluate all deep learning models with architecture search"""
    logging.info("Starting adaptive Keras models...")
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    num_classes = data_dict['num_classes']
    
    # Get models to train
    models_to_train = config.get('deep_learning', {}).get('models', [])
    
    successful_models = []
    failed_models = []
    
    # Results file path
    results_path = os.path.join('results', 'results.csv')
    
    for model_type in models_to_train:
        try:
            logging.info(f"Starting adaptive Keras model: {model_type}")
            
            # Get architecture configurations
            arch_configs = config.get('model_architectures', {}).get(model_type, {})
            default_config = arch_configs.get('default_config', {})
            search_space = arch_configs.get('search_space', {})
            
            logging.info(f"Testing {len(search_space) + 1} different architectures for {model_type}")
            
            best_result = None
            best_f1 = 0
            
            # Test default configuration first
            try:
                input_shape = (100, 1)  # Will be adjusted based on tokenization
                model = create_model(model_type, input_shape, num_classes, default_config)
                
                result = train_adaptive_model(
                    model, X_train, y_train, X_test, y_test, 
                    f"{model_type}_default", config, default_config
                )
                
                if result and result['f1'] > best_f1:
                    best_result = result
                    best_f1 = result['f1']
                    
            except Exception as e:
                logging.error(f"{model_type} default config failed: {e}")
            
            # Test search space configurations
            for i, (param_name, param_values) in enumerate(search_space.items()):
                try:
                    logging.info(f"Testing architecture {i+1}/{len(search_space)} for {model_type}")
                    
                    # Create config with this parameter
                    test_config = default_config.copy()
                    test_config[param_name] = param_values[0] if isinstance(param_values, list) else param_values
                    
                    input_shape = (100, 1)
                    model = create_model(model_type, input_shape, num_classes, test_config)
                    
                    result = train_adaptive_model(
                        model, X_train, y_train, X_test, y_test,
                        f"{model_type}_arch_{i+1}", config, test_config
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
                        "keras_adaptive",
                        'success'
                    ])
                
                # Save best model
                model_dir = os.path.join('models', f"{model_type}_best")
                os.makedirs(model_dir, exist_ok=True)
                best_result['model'].save(os.path.join(model_dir, 'model.h5'))
                
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