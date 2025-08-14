"""
Shadow CNN Model Architecture

This module implements the ShadowCNN architecture optimized for stress detection
using BVP signals. The model is specifically designed for TinyML deployment
with Quantization-Aware Training (QAT) support for ESP32-S3 deployment.

Features:
- Hybrid CNN architecture for BVP signal and HRV features
- Quantization-Aware Training (QAT) ready
- Optimized for ESP32-S3 constraints
- Configurable architecture parameters
- Memory-efficient design for edge deployment
- Real-time inference capabilities

Architecture:
- Input 1: BVP signal segments (3840 samples at 64Hz)
- CNN Branch: 1D convolutions for signal feature extraction
- Input 2: HRV features (time, frequency, non-linear metrics)
- Dense Branch: Feature processing and combination
- Output: Multi-class stress state classification

Author: Shadow AI Team
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ShadowCNN model architecture."""
    # Input shapes
    bvp_input_shape: Tuple[int, int] = (3840, 1)  # 60 seconds at 64Hz
    hrv_input_shape: Tuple[int,] = (20,)          # HRV features
    num_classes: int = 4                          # baseline, stress, amusement, meditation
    
    # CNN branch configuration
    cnn_filters: List[int] = None
    cnn_kernels: List[int] = None
    cnn_strides: List[int] = None
    cnn_pools: List[int] = None
    
    # Dense branch configuration
    dense_units: List[int] = None
    
    # Combined layers configuration
    combined_units: List[int] = None
    
    # Regularization
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    def __post_init__(self):
        """Set default values for configuration."""
        if self.cnn_filters is None:
            self.cnn_filters = [16, 32, 64, 32]
        if self.cnn_kernels is None:
            self.cnn_kernels = [16, 8, 4, 4]
        if self.cnn_strides is None:
            self.cnn_strides = [1, 1, 1, 1]
        if self.cnn_pools is None:
            self.cnn_pools = [4, 4, 4, 2]
        if self.dense_units is None:
            self.dense_units = [32, 16]
        if self.combined_units is None:
            self.combined_units = [64, 32]

class ShadowCNN:
    """
    Shadow CNN for stress detection using BVP signals and HRV features.
    
    This implementation uses a framework-agnostic approach to support multiple
    backends (TensorFlow, PyTorch) and is optimized for TinyML deployment.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the ShadowCNN model.
        
        Args:
            config: Model configuration. Uses default if None.
        """
        self.config = config or ModelConfig()
        self.model = None
        self.is_compiled = False
        self.is_quantized = False
        
        # Model architecture components
        self.cnn_layers = []
        self.dense_layers = []
        self.combined_layers = []
        self.output_layer = None
        
        # Training history
        self.training_history = {}
        
        # Performance metrics
        self.metrics = {}
        
        logger.info(f"ShadowCNN initialized with config: {self.config}")
    
    def build_model(self, framework: str = 'tensorflow') -> object:
        """
        Build the model using the specified framework.
        
        Args:
            framework: Framework to use ('tensorflow', 'pytorch')
            
        Returns:
            Built model object
        """
        if framework.lower() == 'tensorflow':
            return self._build_tensorflow_model()
        elif framework.lower() == 'pytorch':
            return self._build_pytorch_model()
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _build_tensorflow_model(self):
        """Build model using TensorFlow/Keras."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers
        except ImportError:
            logger.error("TensorFlow not available")
            raise
        
        # Input layers
        bvp_input = keras.Input(shape=self.config.bvp_input_shape, name='bvp_input')
        hrv_input = keras.Input(shape=self.config.hrv_input_shape, name='hrv_input')
        
        # CNN Branch for BVP signal processing
        x = bvp_input
        
        for i, (filters, kernel, stride, pool) in enumerate(zip(
            self.config.cnn_filters, 
            self.config.cnn_kernels,
            self.config.cnn_strides,
            self.config.cnn_pools
        )):
            # Convolutional layer
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'conv1d_{i+1}'
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name=f'bn_conv_{i+1}')(x)
            
            # Max pooling
            x = layers.MaxPooling1D(
                pool_size=pool,
                padding='same',
                name=f'pool_{i+1}'
            )(x)
            
            # Dropout for regularization
            if i < len(self.config.cnn_filters) - 1:  # No dropout on last layer
                x = layers.Dropout(self.config.dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        # Global average pooling to reduce parameters
        cnn_output = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense branch for HRV features
        y = hrv_input
        
        for i, units in enumerate(self.config.dense_units):
            y = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_hrv_{i+1}'
            )(y)
            
            y = layers.BatchNormalization(name=f'bn_dense_{i+1}')(y)
            y = layers.Dropout(self.config.dropout_rate, name=f'dropout_dense_{i+1}')(y)
        
        # Combine CNN and Dense branches
        combined = layers.Concatenate(name='concatenate')([cnn_output, y])
        
        # Combined processing layers
        z = combined
        for i, units in enumerate(self.config.combined_units):
            z = layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.config.l2_reg),
                name=f'dense_combined_{i+1}'
            )(z)
            
            z = layers.BatchNormalization(name=f'bn_combined_{i+1}')(z)
            z = layers.Dropout(self.config.dropout_rate, name=f'dropout_combined_{i+1}')(z)
        
        # Output layer
        output = layers.Dense(
            units=self.config.num_classes,
            activation='softmax',
            name='output'
        )(z)
        
        # Create model
        model = keras.Model(
            inputs=[bvp_input, hrv_input],
            outputs=output,
            name='ShadowCNN'
        )
        
        self.model = model
        logger.info("TensorFlow model built successfully")
        
        return model
    
    def _build_pytorch_model(self):
        """Build model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            logger.error("PyTorch not available")
            raise
        
        class ShadowCNNPyTorch(nn.Module):
            def __init__(self, config):
                super(ShadowCNNPyTorch, self).__init__()
                self.config = config
                
                # CNN layers for BVP
                self.conv_layers = nn.ModuleList()
                self.bn_conv_layers = nn.ModuleList()
                self.pool_layers = nn.ModuleList()
                
                in_channels = 1
                for i, (filters, kernel, stride, pool) in enumerate(zip(
                    config.cnn_filters, config.cnn_kernels, 
                    config.cnn_strides, config.cnn_pools
                )):
                    self.conv_layers.append(
                        nn.Conv1d(in_channels, filters, kernel, stride, padding=kernel//2)
                    )
                    self.bn_conv_layers.append(nn.BatchNorm1d(filters))
                    self.pool_layers.append(nn.MaxPool1d(pool))
                    in_channels = filters
                
                # Dense layers for HRV features
                self.dense_layers = nn.ModuleList()
                self.bn_dense_layers = nn.ModuleList()
                
                in_features = config.hrv_input_shape[0]
                for units in config.dense_units:
                    self.dense_layers.append(nn.Linear(in_features, units))
                    self.bn_dense_layers.append(nn.BatchNorm1d(units))
                    in_features = units
                
                # Combined layers
                combined_input_size = config.cnn_filters[-1] + config.dense_units[-1]
                self.combined_layers = nn.ModuleList()
                self.bn_combined_layers = nn.ModuleList()
                
                for units in config.combined_units:
                    self.combined_layers.append(nn.Linear(combined_input_size, units))
                    self.bn_combined_layers.append(nn.BatchNorm1d(units))
                    combined_input_size = units
                
                # Output layer
                self.output_layer = nn.Linear(config.combined_units[-1], config.num_classes)
                self.dropout = nn.Dropout(config.dropout_rate)
            
            def forward(self, bvp_input, hrv_input):
                # CNN branch
                x = bvp_input.transpose(1, 2)  # (batch, channels, sequence)
                
                for i, (conv, bn, pool) in enumerate(zip(
                    self.conv_layers, self.bn_conv_layers, self.pool_layers
                )):
                    x = conv(x)
                    x = bn(x)
                    x = F.relu(x)
                    x = pool(x)
                    if i < len(self.conv_layers) - 1:
                        x = self.dropout(x)
                
                # Global average pooling
                cnn_output = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
                
                # Dense branch
                y = hrv_input
                for dense, bn in zip(self.dense_layers, self.bn_dense_layers):
                    y = dense(y)
                    y = bn(y)
                    y = F.relu(y)
                    y = self.dropout(y)
                
                # Combine branches
                combined = torch.cat([cnn_output, y], dim=1)
                
                # Combined processing
                z = combined
                for dense, bn in zip(self.combined_layers, self.bn_combined_layers):
                    z = dense(z)
                    z = bn(z)
                    z = F.relu(z)
                    z = self.dropout(z)
                
                # Output
                output = self.output_layer(z)
                return F.softmax(output, dim=1)
        
        model = ShadowCNNPyTorch(self.config)
        self.model = model
        logger.info("PyTorch model built successfully")
        
        return model
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     loss: str = 'categorical_crossentropy',
                     metrics: List[str] = None):
        """
        Compile the model for training.
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if metrics is None:
            metrics = ['accuracy']
        
        try:
            import tensorflow as tf
            
            if isinstance(self.model, tf.keras.Model):
                self.model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics
                )
                self.is_compiled = True
                logger.info("TensorFlow model compiled successfully")
            
        except ImportError:
            pass
        
        try:
            import torch
            
            if isinstance(self.model, torch.nn.Module):
                # PyTorch compilation would be handled in training loop
                self.is_compiled = True
                logger.info("PyTorch model ready for training")
                
        except ImportError:
            pass
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            String representation of model summary
        """
        if self.model is None:
            return "Model not built yet"
        
        try:
            import tensorflow as tf
            
            if isinstance(self.model, tf.keras.Model):
                summary_list = []
                self.model.summary(print_fn=lambda x: summary_list.append(x))
                return '\\n'.join(summary_list)
                
        except ImportError:
            pass
        
        try:
            import torch
            
            if isinstance(self.model, torch.nn.Module):
                return str(self.model)
                
        except ImportError:
            pass
        
        return "Summary not available"
    
    def get_model_info(self) -> Dict:
        """
        Get detailed information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'config': self.config.__dict__,
            'is_built': self.model is not None,
            'is_compiled': self.is_compiled,
            'is_quantized': self.is_quantized,
            'framework': self._get_framework(),
        }
        
        if self.model is not None:
            info.update(self._get_parameter_count())
        
        return info
    
    def estimate_memory_usage(self) -> Dict:
        """
        Estimate memory usage for deployment.
        
        Returns:
            Dictionary with memory estimates
        """
        if self.model is None:
            return {'error': 'Model not built'}
        
        param_count = self._get_parameter_count()
        
        # Rough estimates for different data types
        memory_estimates = {
            'float32_mb': param_count['total_parameters'] * 4 / (1024**2),
            'float16_mb': param_count['total_parameters'] * 2 / (1024**2),
            'int8_mb': param_count['total_parameters'] / (1024**2),
            'esp32_s3_limit_mb': 8,  # ESP32-S3 memory limit
        }
        
        memory_estimates['fits_esp32_s3'] = memory_estimates['int8_mb'] < memory_estimates['esp32_s3_limit_mb']
        
        return memory_estimates
    
    def prepare_for_quantization(self) -> Dict:
        """
        Prepare model for quantization-aware training.
        
        Returns:
            Dictionary with preparation status and recommendations
        """
        if self.model is None:
            return {'error': 'Model not built'}
        
        recommendations = {
            'qat_ready': False,
            'recommendations': [],
            'layer_analysis': {}
        }
        
        try:
            import tensorflow as tf
            
            if isinstance(self.model, tf.keras.Model):
                # Analyze layers for QAT compatibility
                compatible_layers = 0
                total_layers = 0
                
                for layer in self.model.layers:
                    total_layers += 1
                    if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Dense)):
                        compatible_layers += 1
                        recommendations['layer_analysis'][layer.name] = 'QAT compatible'
                    elif isinstance(layer, (tf.keras.layers.BatchNormalization, 
                                          tf.keras.layers.Dropout)):
                        recommendations['layer_analysis'][layer.name] = 'QAT compatible (helper)'
                    else:
                        recommendations['layer_analysis'][layer.name] = 'Check compatibility'
                
                compatibility_ratio = compatible_layers / max(total_layers, 1)
                recommendations['qat_ready'] = compatibility_ratio > 0.7
                
                if not recommendations['qat_ready']:
                    recommendations['recommendations'].append(
                        'Consider replacing incompatible layers for better QAT support'
                    )
                
                recommendations['compatibility_ratio'] = compatibility_ratio
                
        except ImportError:
            recommendations['recommendations'].append('TensorFlow required for QAT analysis')
        
        return recommendations
    
    def export_for_deployment(self, format: str = 'tflite') -> Dict:
        """
        Export model for deployment.
        
        Args:
            format: Export format ('tflite', 'onnx', 'tensorrt')
            
        Returns:
            Dictionary with export status and file paths
        """
        if self.model is None:
            return {'error': 'Model not built'}
        
        export_info = {
            'format': format,
            'success': False,
            'file_path': None,
            'size_bytes': 0,
            'optimization_applied': []
        }
        
        if format.lower() == 'tflite':
            export_info = self._export_tflite()
        elif format.lower() == 'onnx':
            export_info = self._export_onnx()
        else:
            export_info['error'] = f'Unsupported export format: {format}'
        
        return export_info
    
    def _export_tflite(self) -> Dict:
        """Export model to TensorFlow Lite format."""
        try:
            import tensorflow as tf
            
            if not isinstance(self.model, tf.keras.Model):
                return {'error': 'TFLite export requires TensorFlow model'}
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Apply optimizations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if self.is_quantized:
                converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save to file
            tflite_path = 'shadow_cnn_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            return {
                'format': 'tflite',
                'success': True,
                'file_path': tflite_path,
                'size_bytes': len(tflite_model),
                'optimization_applied': ['DEFAULT']
            }
            
        except Exception as e:
            return {'error': f'TFLite export failed: {str(e)}'}
    
    def _export_onnx(self) -> Dict:
        """Export model to ONNX format."""
        return {'error': 'ONNX export not implemented yet'}
    
    def _get_framework(self) -> str:
        """Determine which framework is being used."""
        if self.model is None:
            return 'none'
        
        try:
            import tensorflow as tf
            if isinstance(self.model, tf.keras.Model):
                return 'tensorflow'
        except ImportError:
            pass
        
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                return 'pytorch'
        except ImportError:
            pass
        
        return 'unknown'
    
    def _get_parameter_count(self) -> Dict:
        """Get parameter count information."""
        if self.model is None:
            return {'total_parameters': 0, 'trainable_parameters': 0}
        
        try:
            import tensorflow as tf
            
            if isinstance(self.model, tf.keras.Model):
                total_params = self.model.count_params()
                trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
                
                return {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'non_trainable_parameters': total_params - trainable_params
                }
                
        except ImportError:
            pass
        
        try:
            import torch
            
            if isinstance(self.model, torch.nn.Module):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                return {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'non_trainable_parameters': total_params - trainable_params
                }
                
        except ImportError:
            pass
        
        return {'total_parameters': 0, 'trainable_parameters': 0}

def create_shadow_cnn(config: Optional[ModelConfig] = None, 
                     framework: str = 'tensorflow') -> ShadowCNN:
    """
    Create and build a ShadowCNN model.
    
    Args:
        config: Model configuration
        framework: Framework to use
        
    Returns:
        Built ShadowCNN model
    """
    model = ShadowCNN(config)
    model.build_model(framework)
    return model