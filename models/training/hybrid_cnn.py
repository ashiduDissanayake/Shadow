"""
Hybrid CNN Model Architecture

This module implements the Hybrid CNN (H-CNN) architecture for stress detection
using BVP signals and HRV features.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HybridCNN(keras.Model):
    """
    Hybrid CNN model for stress detection.
    
    Architecture:
    - Input 1: BVP signal segment (3840 samples)
    - CNN Branch: Convolutional layers for signal processing
    - Input 2: HRV features (19 features)
    - Dense Branch: Feature processing
    - Concatenation: Combine both branches
    - Output: Stress classification
    """
    
    def __init__(self, 
                 input_shape_bvp: Tuple[int, int] = (3840, 1),
                 input_shape_features: Tuple[int] = (19,),
                 num_classes: int = 4,
                 cnn_filters: list = [16, 32, 16],
                 cnn_kernel_sizes: list = [8, 16, 8],
                 cnn_pool_sizes: list = [4, 4, 4],
                 dense_units: list = [64, 32],
                 combined_units: list = [128, 64],
                 dropout_rate: float = 0.3,
                 name: str = "hybrid_cnn"):
        """
        Initialize Hybrid CNN model.
        
        Args:
            input_shape_bvp: Shape of BVP input (samples, channels)
            input_shape_features: Shape of HRV features input
            num_classes: Number of output classes
            cnn_filters: Number of filters for each CNN layer
            cnn_kernel_sizes: Kernel sizes for CNN layers
            cnn_pool_sizes: Pooling sizes for CNN layers
            dense_units: Units for dense layers in feature branch
            combined_units: Units for combined layers
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(HybridCNN, self).__init__(name=name)
        
        self.input_shape_bvp = input_shape_bvp
        self.input_shape_features = input_shape_features
        self.num_classes = num_classes
        
        # Build model components
        self._build_cnn_branch(cnn_filters, cnn_kernel_sizes, cnn_pool_sizes, dropout_rate)
        self._build_dense_branch(dense_units, dropout_rate)
        self._build_combined_layers(combined_units, dropout_rate)
        self._build_output_layer()
        
    def _build_cnn_branch(self, filters: list, kernel_sizes: list, pool_sizes: list, dropout_rate: float):
        """Build CNN branch for BVP signal processing."""
        self.cnn_layers = []
        
        for i, (filters_count, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            # Convolutional layer
            conv_layer = layers.Conv1D(
                filters=filters_count,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )
            self.cnn_layers.append(conv_layer)
            
            # Batch normalization
            bn_layer = layers.BatchNormalization(name=f'bn_conv_{i+1}')
            self.cnn_layers.append(bn_layer)
            
            # Max pooling
            pool_layer = layers.MaxPooling1D(
                pool_size=pool_size,
                padding='same',
                name=f'maxpool_{i+1}'
            )
            self.cnn_layers.append(pool_layer)
            
            # Dropout
            dropout_layer = layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}')
            self.cnn_layers.append(dropout_layer)
            
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D(name='global_pool')
        
    def _build_dense_branch(self, units: list, dropout_rate: float):
        """Build dense branch for HRV features processing."""
        self.dense_layers = []
        
        for i, unit_count in enumerate(units):
            dense_layer = layers.Dense(
                units=unit_count,
                activation='relu',
                name=f'dense_features_{i+1}'
            )
            self.dense_layers.append(dense_layer)
            
            # Batch normalization
            bn_layer = layers.BatchNormalization(name=f'bn_dense_{i+1}')
            self.dense_layers.append(bn_layer)
            
            # Dropout
            dropout_layer = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')
            self.dense_layers.append(dropout_layer)
            
    def _build_combined_layers(self, units: list, dropout_rate: float):
        """Build layers after concatenation."""
        self.combined_layers = []
        
        for i, unit_count in enumerate(units):
            dense_layer = layers.Dense(
                units=unit_count,
                activation='relu',
                name=f'dense_combined_{i+1}'
            )
            self.combined_layers.append(dense_layer)
            
            # Batch normalization
            bn_layer = layers.BatchNormalization(name=f'bn_combined_{i+1}')
            self.combined_layers.append(bn_layer)
            
            # Dropout
            dropout_layer = layers.Dropout(dropout_rate, name=f'dropout_combined_{i+1}')
            self.combined_layers.append(dropout_layer)
            
    def _build_output_layer(self):
        """Build output classification layer."""
        self.output_layer = layers.Dense(
            units=self.num_classes,
            activation='softmax',
            name='output'
        )
        
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary with 'bvp' and 'features' keys
            training: Whether in training mode
            
        Returns:
            Model predictions
        """
        bvp_input = inputs['bvp']
        features_input = inputs['features']
        
        # CNN branch for BVP processing
        x_bvp = bvp_input
        for layer in self.cnn_layers:
            x_bvp = layer(x_bvp, training=training)
        x_bvp = self.global_pool(x_bvp)
        
        # Dense branch for features processing
        x_features = features_input
        for layer in self.dense_layers:
            x_features = layer(x_features, training=training)
            
        # Concatenate both branches
        x_combined = layers.Concatenate(name='concatenate')([x_bvp, x_features])
        
        # Combined layers
        for layer in self.combined_layers:
            x_combined = layer(x_combined, training=training)
            
        # Output layer
        output = self.output_layer(x_combined)
        
        return output
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        # Create a temporary model to get summary
        bvp_input = keras.Input(shape=self.input_shape_bvp, name='bvp')
        features_input = keras.Input(shape=self.input_shape_features, name='features')
        
        temp_model = keras.Model(
            inputs=[bvp_input, features_input],
            outputs=self.call({'bvp': bvp_input, 'features': features_input})
        )
        
        # Capture summary
        summary_list = []
        temp_model.summary(print_fn=lambda x: summary_list.append(x))
        
        return '\n'.join(summary_list)
    
    def get_model_info(self) -> Dict:
        """Get model information and parameters."""
        # Count parameters
        total_params = self.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'model_name': self.name,
            'input_shape_bvp': self.input_shape_bvp,
            'input_shape_features': self.input_shape_features,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'cnn_layers': len([l for l in self.cnn_layers if isinstance(l, layers.Conv1D)]),
            'dense_layers': len([l for l in self.dense_layers if isinstance(l, layers.Dense)]),
            'combined_layers': len([l for l in self.combined_layers if isinstance(l, layers.Dense)])
        }

def create_hybrid_cnn(config: Dict) -> HybridCNN:
    """
    Create Hybrid CNN model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured HybridCNN model
    """
    model_config = config['model']
    
    model = HybridCNN(
        input_shape_bvp=tuple(model_config['input_shape_bvp']),
        input_shape_features=tuple(model_config['input_shape_features']),
        num_classes=model_config['num_classes'],
        cnn_filters=model_config['cnn_branch']['filters'],
        cnn_kernel_sizes=model_config['cnn_branch']['kernel_sizes'],
        cnn_pool_sizes=model_config['cnn_branch']['pool_sizes'],
        dense_units=model_config['dense_branch']['units'],
        combined_units=model_config['combined']['units'],
        dropout_rate=model_config['cnn_branch']['dropout_rate']
    )
    
    logger.info(f"Created Hybrid CNN model: {model.get_model_info()}")
    return model
