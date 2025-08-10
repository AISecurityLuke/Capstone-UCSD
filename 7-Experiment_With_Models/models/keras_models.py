"""
Keras model architectures for text classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, Tuple

def create_cnn_model(input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create CNN model for text classification with token embeddings"""
    # ------------------------------------------------------------------
    # New: Embedding hyper-parameters
    # ------------------------------------------------------------------
    filters = config.get('filters', 128)
    kernel_size = config.get('kernel_size', 5)
    dropout = config.get('dropout', 0.3)
    layers_count = config.get('layers', 2)
    vocab_size = config.get('vocab_size', 10000)
    embed_dim = config.get('embedding_dim', 128)

    seq_len = input_shape[0]  # ignore any singleton channel dim

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, embed_dim, input_length=seq_len)(inputs)
    x = layers.Dropout(dropout)(x)

    for _ in range(layers_count):
        x = layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(dropout)(x)
        filters = filters // 2  # Reduce filters in deeper layers

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_lstm_model(input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create LSTM model for text classification with token embeddings"""
    units = config.get('units', 128)
    layers_count = config.get('layers', 2)
    dropout = config.get('dropout', 0.2)
    bidirectional = config.get('bidirectional', False)
    vocab_size = config.get('vocab_size', 10000)
    embed_dim = config.get('embedding_dim', 128)

    seq_len = input_shape[0]

    inputs = keras.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, embed_dim, input_length=seq_len)(inputs)
    x = layers.Dropout(dropout)(x)

    for i in range(layers_count):
        return_seq = i < layers_count - 1
        lstm_layer = layers.LSTM(units, return_sequences=return_seq, dropout=dropout)
        if bidirectional:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
        x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_bilstm_model(input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create Bidirectional LSTM model for text classification with embeddings"""
    units = config.get('units', 128)
    layers_count = config.get('layers', 2)
    dropout = config.get('dropout', 0.3)
    bidirectional = config.get('bidirectional', True)
    vocab_size = config.get('vocab_size', 10000)
    embed_dim = config.get('embedding_dim', 128)

    seq_len = input_shape[0]

    inputs = keras.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, embed_dim, input_length=seq_len)(inputs)
    x = layers.Dropout(dropout)(x)

    for i in range(layers_count):
        return_seq = i < layers_count - 1
        lstm_layer = layers.LSTM(units, return_sequences=return_seq, dropout=dropout)
        if bidirectional:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)
        x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_transformer_model(input_shape: Tuple[int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create lightweight Transformer encoder for text classification.

    Expected input_shape = (sequence_length,). Tokens are integer indices. The
    function embeds tokens, adds sinusoidal positional encoding, passes them
    through `num_layers` Transformer encoder blocks, and pools with
    GlobalAveragePooling1D.  This replaces the previous implementation that
    incorrectly expected an (seq_len, d_model) tensor from the caller.
    """

    # Hyper-parameters
    seq_len   = input_shape[0]
    embed_dim = config.get('embedding_dim', 128)
    vocab_sz  = config.get('vocab_size', 10000)
    num_heads = config.get('num_heads', 8)
    ff_dim    = config.get('ff_dim', 512)
    num_layers= config.get('num_layers', 2)
    dropout   = config.get('dropout', 0.1)

    # Inputs are token ids
    inputs = keras.Input(shape=(seq_len,), dtype='int32')

    # Token embeddings + positional encoding
    x = layers.Embedding(vocab_sz, embed_dim, input_length=seq_len)(inputs)
    x = layers.Dropout(dropout)(x)
    x += positional_encoding(seq_len, embed_dim)

    # Transformer encoder blocks
    for _ in range(num_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

        ffn_out = layers.Dense(ff_dim, activation='relu')(x)
        ffn_out = layers.Dense(embed_dim)(ffn_out)
        ffn_out = layers.Dropout(dropout)(ffn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)

    # Pool & classify
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def create_hybrid_model(input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create Hybrid CNN+LSTM model (embedding → CNN & LSTM branches)"""
    cnn_filters = config.get('cnn_filters', 128)
    cnn_kernel_size = config.get('cnn_kernel_size', 5)
    lstm_units = config.get('lstm_units', 128)
    dropout = config.get('dropout', 0.3)
    vocab_size = config.get('vocab_size', 10000)
    embed_dim = config.get('embedding_dim', 128)

    seq_len = input_shape[0]

    inputs = keras.Input(shape=(seq_len,), dtype='int32')
    embedded = layers.Embedding(vocab_size, embed_dim, input_length=seq_len)(inputs)
    embedded = layers.Dropout(dropout)(embedded)

    # CNN branch
    cnn_branch = layers.Conv1D(cnn_filters, cnn_kernel_size, activation='relu', padding='same')(embedded)
    cnn_branch = layers.BatchNormalization()(cnn_branch)
    cnn_branch = layers.MaxPooling1D(2)(cnn_branch)
    cnn_branch = layers.Dropout(dropout)(cnn_branch)
    cnn_branch = layers.GlobalMaxPooling1D()(cnn_branch)

    # LSTM branch
    lstm_branch = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=dropout))(embedded)
    lstm_branch = layers.BatchNormalization()(lstm_branch)
    lstm_branch = layers.Dropout(dropout)(lstm_branch)

    # Combine branches
    combined = layers.Concatenate()([cnn_branch, lstm_branch])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def positional_encoding(position, d_model):
    """Create positional encoding for transformer"""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    """Helper function for positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

# Model factory
MODEL_FACTORY = {
    'cnn': create_cnn_model,
    'lstm': create_lstm_model,
    'bilstm': create_bilstm_model,
    'transformer': create_transformer_model,
    'hybrid': create_hybrid_model
}

def create_model(model_type: str, input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]) -> keras.Model:
    """Create model by type"""
    if model_type not in MODEL_FACTORY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return MODEL_FACTORY[model_type](input_shape, num_classes, config) 