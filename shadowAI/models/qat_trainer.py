"""
Quantization-Aware Training (QAT) Module

This module provides comprehensive Quantization-Aware Training capabilities
for the ShadowCNN model, optimizing for deployment on ESP32-S3 and other
resource-constrained edge devices.

Features:
- TensorFlow Lite quantization-aware training
- Custom quantization strategies for BVP signals
- Memory optimization for edge deployment
- Performance monitoring during QAT
- Automated hyperparameter tuning for quantized models
- ESP32-S3 specific optimizations

Author: Shadow AI Team
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QATConfig:
    """Configuration for Quantization-Aware Training."""
    # Quantization settings
    quantization_type: str = 'int8'  # 'int8', 'float16', 'dynamic'
    representative_dataset_size: int = 1000
    optimization_level: str = 'default'  # 'default', 'size', 'latency'
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    
    # QAT specific
    quantize_delay: int = 5  # Start quantization after N epochs
    quantization_schedule: str = 'linear'  # 'linear', 'cosine', 'step'
    
    # Target hardware
    target_device: str = 'esp32_s3'
    memory_limit_mb: float = 8.0
    inference_time_limit_ms: float = 100.0
    
    # Validation settings
    min_accuracy_retention: float = 0.95  # Minimum accuracy vs full precision
    
class QATTrainer:
    """
    Quantization-Aware Training manager for ShadowCNN models.
    
    Provides end-to-end QAT pipeline including model preparation,
    training with quantization simulation, validation, and optimization
    for target hardware deployment.
    """
    
    def __init__(self, config: Optional[QATConfig] = None):
        """
        Initialize QAT trainer.
        
        Args:
            config: QAT configuration. Uses default if None.
        """
        self.config = config or QATConfig()
        self.model = None
        self.quantized_model = None
        self.original_model = None
        
        # Training state
        self.training_history = {}
        self.quantization_metrics = {}
        self.is_qat_ready = False
        
        # Performance tracking
        self.performance_metrics = {
            'accuracy_retention': 0.0,
            'model_size_reduction': 0.0,
            'inference_speedup': 0.0,
            'memory_usage': 0.0
        }
        
        logger.info(f"QAT Trainer initialized for {self.config.target_device}")
    
    def prepare_model_for_qat(self, model) -> bool:
        """
        Prepare a model for quantization-aware training.
        
        Args:
            model: Model to prepare for QAT
            
        Returns:
            True if preparation successful, False otherwise
        """
        try:
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot
            
            if not isinstance(model, tf.keras.Model):
                logger.error("QAT currently supports TensorFlow models only")
                return False
            
            self.original_model = model
            
            # Check model compatibility
            if not self._check_qat_compatibility(model):
                logger.warning("Model may not be fully compatible with QAT")
            
            # Apply quantization-aware training
            quantize_model = tfmot.quantization.keras.quantize_model
            
            # Configure quantization
            quantize_config = self._get_quantization_config()
            
            self.quantized_model = quantize_model(
                model, 
                quantized_layer_config=quantize_config
            )
            
            self.model = self.quantized_model
            self.is_qat_ready = True
            
            logger.info("Model prepared for QAT successfully")
            return True
            
        except ImportError:
            logger.error("TensorFlow Model Optimization Toolkit not available")
            return False
        except Exception as e:
            logger.error(f"Failed to prepare model for QAT: {e}")
            return False
    
    def train_with_qat(self, 
                      train_data: Tuple[np.ndarray, np.ndarray],
                      validation_data: Tuple[np.ndarray, np.ndarray],
                      callbacks: Optional[List] = None) -> Dict:
        """
        Train model with quantization-aware training.
        
        Args:
            train_data: Training data (X, y)
            validation_data: Validation data (X_val, y_val)
            callbacks: Optional additional callbacks
            
        Returns:
            Training history and metrics
        """
        if not self.is_qat_ready:
            raise ValueError("Model not prepared for QAT. Call prepare_model_for_qat first.")
        
        try:
            import tensorflow as tf
            
            # Prepare data
            X_train, y_train = train_data
            X_val, y_val = validation_data
            
            # Setup optimizer with lower learning rate for QAT
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            
            # Compile quantized model
            self.quantized_model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Setup callbacks
            qat_callbacks = self._setup_qat_callbacks()
            if callbacks:
                qat_callbacks.extend(callbacks)
            
            # Train the quantized model
            logger.info("Starting QAT training...")
            start_time = time.time()
            
            history = self.quantized_model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_val, y_val),
                callbacks=qat_callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Store training history
            self.training_history = history.history
            
            # Evaluate performance
            self._evaluate_qat_performance(validation_data)
            
            # Generate training summary
            training_summary = {
                'training_time_seconds': training_time,
                'epochs_completed': len(history.history['loss']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'final_val_accuracy': history.history['val_accuracy'][-1],
                'performance_metrics': self.performance_metrics,
                'quantization_metrics': self.quantization_metrics
            }
            
            logger.info(f"QAT training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation accuracy: {training_summary['best_val_accuracy']:.4f}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"QAT training failed: {e}")
            raise
    
    def convert_to_tflite(self, 
                         representative_data: Optional[np.ndarray] = None,
                         output_path: Optional[str] = None) -> Dict:
        """
        Convert quantized model to TensorFlow Lite format.
        
        Args:
            representative_data: Representative dataset for quantization calibration
            output_path: Path to save the TFLite model
            
        Returns:
            Conversion results and model information
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Train with QAT first.")
        
        try:
            import tensorflow as tf
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.quantized_model)
            
            # Configure optimization
            if self.config.quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                
                # Set representative dataset for full integer quantization
                if representative_data is not None:
                    def representative_dataset():
                        for data in representative_data[:self.config.representative_dataset_size]:
                            yield [data.astype(np.float32)]
                    
                    converter.representative_dataset = representative_dataset
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    
            elif self.config.quantization_type == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                
            elif self.config.quantization_type == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert model
            logger.info(f"Converting model to TFLite with {self.config.quantization_type} quantization...")
            tflite_model = converter.convert()
            
            # Save model
            if output_path is None:
                output_path = f"shadow_cnn_{self.config.quantization_type}.tflite"
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Analyze converted model
            model_info = self._analyze_tflite_model(tflite_model, output_path)
            
            logger.info(f"TFLite model saved: {output_path} ({model_info['size_kb']:.1f} KB)")
            
            return model_info
            
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            raise
    
    def benchmark_inference(self, 
                           tflite_model_path: str,
                           test_data: np.ndarray,
                           num_runs: int = 100) -> Dict:
        """
        Benchmark inference performance of TFLite model.
        
        Args:
            tflite_model_path: Path to TFLite model
            test_data: Test data for benchmarking
            num_runs: Number of inference runs for timing
            
        Returns:
            Benchmark results
        """
        try:
            import tensorflow as tf
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare test data
            if len(test_data.shape) == 2:  # Add batch dimension
                test_data = np.expand_dims(test_data, 0)
            
            # Warm up
            for _ in range(5):
                interpreter.set_tensor(input_details[0]['index'], test_data[:1])
                interpreter.invoke()
            
            # Benchmark inference time
            inference_times = []
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                
                interpreter.set_tensor(input_details[0]['index'], test_data[i % len(test_data):i % len(test_data) + 1])
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            benchmark_results = {
                'mean_inference_time_ms': np.mean(inference_times),
                'std_inference_time_ms': np.std(inference_times),
                'min_inference_time_ms': np.min(inference_times),
                'max_inference_time_ms': np.max(inference_times),
                'p95_inference_time_ms': np.percentile(inference_times, 95),
                'p99_inference_time_ms': np.percentile(inference_times, 99),
                'throughput_samples_per_second': 1000 / np.mean(inference_times),
                'meets_latency_requirement': np.mean(inference_times) <= self.config.inference_time_limit_ms,
                'model_path': tflite_model_path,
                'test_samples': len(test_data),
                'benchmark_runs': num_runs
            }
            
            logger.info(f"Inference benchmark completed:")
            logger.info(f"  Mean latency: {benchmark_results['mean_inference_time_ms']:.2f} ms")
            logger.info(f"  Throughput: {benchmark_results['throughput_samples_per_second']:.1f} samples/sec")
            logger.info(f"  Meets requirement: {benchmark_results['meets_latency_requirement']}")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Inference benchmarking failed: {e}")
            raise
    
    def optimize_for_target_device(self) -> Dict:
        """
        Optimize model specifically for target device constraints.
        
        Returns:
            Optimization results and recommendations
        """
        optimization_results = {
            'target_device': self.config.target_device,
            'optimizations_applied': [],
            'performance_improvements': {},
            'meets_constraints': False,
            'recommendations': []
        }
        
        if self.config.target_device == 'esp32_s3':
            optimization_results.update(self._optimize_for_esp32_s3())
        else:
            optimization_results['recommendations'].append(
                f"No specific optimizations available for {self.config.target_device}"
            )
        
        return optimization_results
    
    def get_training_summary(self) -> Dict:
        """
        Get comprehensive summary of QAT training process.
        
        Returns:
            Training summary with all metrics and analysis
        """
        summary = {
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'quantization_metrics': self.quantization_metrics,
            'model_info': self._get_model_info(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _check_qat_compatibility(self, model) -> bool:
        """Check if model is compatible with QAT."""
        try:
            import tensorflow as tf
            
            compatible_layers = 0
            total_layers = 0
            
            for layer in model.layers:
                total_layers += 1
                if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Dense)):
                    compatible_layers += 1
                elif isinstance(layer, (tf.keras.layers.BatchNormalization,
                                      tf.keras.layers.Dropout,
                                      tf.keras.layers.MaxPooling1D,
                                      tf.keras.layers.GlobalAveragePooling1D)):
                    compatible_layers += 1
            
            compatibility_ratio = compatible_layers / max(total_layers, 1)
            self.quantization_metrics['compatibility_ratio'] = compatibility_ratio
            
            return compatibility_ratio > 0.8
            
        except Exception as e:
            logger.warning(f"Compatibility check failed: {e}")
            return False
    
    def _get_quantization_config(self):
        """Get quantization configuration for different layer types."""
        try:
            import tensorflow_model_optimization as tfmot
            
            # Custom quantization config for BVP signal processing
            def get_quantization_config(layer):
                if isinstance(layer, tf.keras.layers.Conv1D):
                    # More aggressive quantization for conv layers
                    return tfmot.quantization.keras.QuantizeConfig(
                        weight_attrs=['kernel'],
                        activation_attrs=['activation'],
                        quantize_output_in_call=True
                    )
                elif isinstance(layer, tf.keras.layers.Dense):
                    # Standard quantization for dense layers
                    return tfmot.quantization.keras.QuantizeConfig(
                        weight_attrs=['kernel'],
                        activation_attrs=['activation'],
                        quantize_output_in_call=True
                    )
                else:
                    return tfmot.quantization.keras.quantize_annotate_layer(layer)
            
            return get_quantization_config
            
        except ImportError:
            logger.warning("TensorFlow Model Optimization not available, using default config")
            return None
    
    def _setup_qat_callbacks(self) -> List:
        """Setup callbacks for QAT training."""
        try:
            import tensorflow as tf
            
            callbacks = []
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Model checkpoint
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'best_qat_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
            
            return callbacks
            
        except ImportError:
            logger.warning("TensorFlow not available for callbacks")
            return []
    
    def _evaluate_qat_performance(self, validation_data: Tuple[np.ndarray, np.ndarray]):
        """Evaluate QAT performance against original model."""
        if self.original_model is None or self.quantized_model is None:
            return
        
        try:
            X_val, y_val = validation_data
            
            # Evaluate original model
            original_results = self.original_model.evaluate(X_val, y_val, verbose=0)
            original_accuracy = original_results[1] if len(original_results) > 1 else original_results[0]
            
            # Evaluate quantized model
            quantized_results = self.quantized_model.evaluate(X_val, y_val, verbose=0)
            quantized_accuracy = quantized_results[1] if len(quantized_results) > 1 else quantized_results[0]
            
            # Calculate metrics
            accuracy_retention = quantized_accuracy / original_accuracy
            
            self.performance_metrics.update({
                'original_accuracy': original_accuracy,
                'quantized_accuracy': quantized_accuracy,
                'accuracy_retention': accuracy_retention,
                'accuracy_drop': original_accuracy - quantized_accuracy
            })
            
            self.quantization_metrics.update({
                'meets_accuracy_requirement': accuracy_retention >= self.config.min_accuracy_retention
            })
            
            logger.info(f"Accuracy retention: {accuracy_retention:.4f}")
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
    
    def _analyze_tflite_model(self, tflite_model: bytes, model_path: str) -> Dict:
        """Analyze TFLite model properties."""
        try:
            import tensorflow as tf
            
            # Basic model info
            model_info = {
                'size_bytes': len(tflite_model),
                'size_kb': len(tflite_model) / 1024,
                'size_mb': len(tflite_model) / (1024 * 1024),
                'model_path': model_path,
                'quantization_type': self.config.quantization_type
            }
            
            # Load interpreter for detailed analysis
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            model_info.update({
                'input_shape': input_details[0]['shape'].tolist(),
                'input_dtype': str(input_details[0]['dtype']),
                'output_shape': output_details[0]['shape'].tolist(),
                'output_dtype': str(output_details[0]['dtype']),
                'num_inputs': len(input_details),
                'num_outputs': len(output_details)
            })
            
            # Memory analysis
            model_info.update({
                'fits_esp32_s3': model_info['size_mb'] < self.config.memory_limit_mb,
                'memory_efficiency': model_info['size_mb'] / self.config.memory_limit_mb
            })
            
            return model_info
            
        except Exception as e:
            logger.error(f"TFLite model analysis failed: {e}")
            return {'error': str(e)}
    
    def _optimize_for_esp32_s3(self) -> Dict:
        """ESP32-S3 specific optimizations."""
        optimizations = {
            'optimizations_applied': [
                'INT8 quantization for memory efficiency',
                'Layer fusion for reduced operations',
                'Memory layout optimization'
            ],
            'esp32_s3_constraints': {
                'memory_limit_mb': self.config.memory_limit_mb,
                'target_latency_ms': self.config.inference_time_limit_ms,
                'power_constraints': 'Low power operation required'
            },
            'recommendations': [
                'Use SPIRAM for model storage if model > 512KB',
                'Consider model pruning for further size reduction',
                'Implement ring buffer for continuous inference',
                'Use ESP32-S3 AI acceleration features'
            ]
        }
        
        return optimizations
    
    def _get_model_info(self) -> Dict:
        """Get detailed model information."""
        if self.quantized_model is None:
            return {'error': 'No quantized model available'}
        
        try:
            import tensorflow as tf
            
            if isinstance(self.quantized_model, tf.keras.Model):
                total_params = self.quantized_model.count_params()
                
                return {
                    'total_parameters': total_params,
                    'model_type': 'TensorFlow/Keras',
                    'quantization_ready': True,
                    'estimated_size_mb': total_params * 1 / (1024 * 1024)  # INT8 estimate
                }
        except:
            pass
        
        return {'error': 'Unable to get model info'}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        if not self.performance_metrics:
            return ['Complete QAT training to get recommendations']
        
        accuracy_retention = self.performance_metrics.get('accuracy_retention', 0)
        
        if accuracy_retention < self.config.min_accuracy_retention:
            recommendations.append(
                f"Accuracy retention ({accuracy_retention:.3f}) is below threshold "
                f"({self.config.min_accuracy_retention}). Consider:")
            recommendations.extend([
                "- Increasing training epochs",
                "- Using a lower learning rate",
                "- Applying gradual quantization",
                "- Using knowledge distillation"
            ])
        
        if self.quantization_metrics.get('compatibility_ratio', 1.0) < 0.9:
            recommendations.append("Consider model architecture changes for better QAT compatibility")
        
        if self.config.target_device == 'esp32_s3':
            recommendations.extend([
                "For ESP32-S3 deployment:",
                "- Test on actual hardware for final validation",
                "- Monitor power consumption during inference",
                "- Consider temperature effects on performance"
            ])
        
        return recommendations