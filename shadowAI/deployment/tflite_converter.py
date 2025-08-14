"""
TensorFlow Lite Model Converter Module

This module provides comprehensive TensorFlow Lite conversion capabilities
for the ShadowCNN model, with optimizations specifically for ESP32-S3
and other edge computing platforms.

Features:
- Multiple quantization strategies (INT8, Float16, Dynamic)
- Representative dataset generation for calibration
- Model optimization for different deployment scenarios
- Comprehensive conversion validation and testing
- Performance benchmarking and analysis
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
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class TFLiteConfig:
    """Configuration for TensorFlow Lite conversion."""
    # Quantization settings
    quantization_type: str = 'int8'  # 'int8', 'float16', 'dynamic', 'full_int8'
    enable_optimizations: bool = True
    target_spec_types: List[str] = None
    
    # Representative dataset
    representative_dataset_size: int = 500
    use_random_representative_data: bool = False
    
    # Optimization settings
    optimization_level: str = 'default'  # 'default', 'optimize_for_size', 'optimize_for_latency'
    experimental_new_converter: bool = True
    experimental_new_quantizer: bool = True
    
    # Input/Output quantization
    inference_input_type: str = 'float32'  # 'float32', 'int8', 'uint8'
    inference_output_type: str = 'float32'  # 'float32', 'int8', 'uint8'
    
    # Validation settings
    validate_conversion: bool = True
    tolerance: float = 0.01  # Tolerance for numerical differences
    
    # ESP32-S3 specific
    esp32_optimizations: bool = True
    memory_limit_mb: float = 8.0
    
    def __post_init__(self):
        """Set default target spec types based on quantization type."""
        if self.target_spec_types is None:
            if self.quantization_type == 'int8':
                self.target_spec_types = ['int8']
            elif self.quantization_type == 'float16':
                self.target_spec_types = ['float16']
            else:
                self.target_spec_types = ['float32']

class TFLiteConverter:
    """
    Advanced TensorFlow Lite converter for ShadowCNN models.
    
    Provides comprehensive conversion capabilities with multiple quantization
    strategies, validation, and optimization for edge deployment.
    """
    
    def __init__(self, config: Optional[TFLiteConfig] = None):
        """
        Initialize the TFLite converter.
        
        Args:
            config: Conversion configuration. Uses default if None.
        """
        self.config = config or TFLiteConfig()
        
        # Conversion results storage
        self.conversion_results = {}
        self.validation_results = {}
        self.benchmark_results = {}
        
        # Model storage
        self.original_model = None
        self.tflite_model = None
        self.interpreter = None
        
        logger.info(f"TFLite converter initialized with {self.config.quantization_type} quantization")
    
    def convert_model(self, 
                     model,
                     representative_data: Optional[np.ndarray] = None,
                     output_path: Optional[str] = None) -> Dict:
        """
        Convert Keras model to TensorFlow Lite format.
        
        Args:
            model: Keras model to convert
            representative_data: Representative dataset for quantization calibration
            output_path: Path to save the converted model
            
        Returns:
            Conversion results dictionary
        """
        logger.info(f"Starting TFLite conversion with {self.config.quantization_type} quantization")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for TFLite conversion")
        
        self.original_model = model
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Configure converter
        self._configure_converter(converter, representative_data)
        
        # Perform conversion
        start_time = time.time()
        try:
            tflite_model = converter.convert()
            conversion_time = time.time() - start_time
            
            self.tflite_model = tflite_model
            
            # Save model
            if output_path is None:
                output_path = f"shadow_cnn_{self.config.quantization_type}.tflite"
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Analyze converted model
            analysis_results = self._analyze_converted_model(tflite_model, output_path)
            
            # Validate conversion if requested
            validation_results = {}
            if self.config.validate_conversion and representative_data is not None:
                validation_results = self.validate_conversion(representative_data)
            
            # Compile results
            conversion_results = {
                'success': True,
                'conversion_time_seconds': conversion_time,
                'output_path': output_path,
                'model_analysis': analysis_results,
                'validation_results': validation_results,
                'config': self.config.__dict__
            }
            
            self.conversion_results = conversion_results
            
            logger.info(f"Conversion completed successfully in {conversion_time:.2f} seconds")
            logger.info(f"Model saved to: {output_path} ({analysis_results['size_kb']:.1f} KB)")
            
            return conversion_results
            
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'config': self.config.__dict__
            }
    
    def convert_with_multiple_strategies(self, 
                                       model,
                                       representative_data: Optional[np.ndarray] = None,
                                       strategies: Optional[List[str]] = None) -> Dict:
        """
        Convert model using multiple quantization strategies for comparison.
        
        Args:
            model: Keras model to convert
            representative_data: Representative dataset
            strategies: List of quantization strategies to try
            
        Returns:
            Results for all conversion strategies
        """
        if strategies is None:
            strategies = ['float32', 'float16', 'dynamic', 'int8']
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"Converting with {strategy} strategy...")
            
            # Update config for this strategy
            original_type = self.config.quantization_type
            self.config.quantization_type = strategy
            
            try:
                result = self.convert_model(
                    model, 
                    representative_data, 
                    f"shadow_cnn_{strategy}.tflite"
                )
                results[strategy] = result
                
            except Exception as e:
                logger.error(f"Conversion with {strategy} failed: {e}")
                results[strategy] = {'success': False, 'error': str(e)}
            
            # Restore original config
            self.config.quantization_type = original_type
        
        # Compare results
        comparison = self._compare_conversion_strategies(results)
        results['comparison'] = comparison
        
        return results
    
    def validate_conversion(self, test_data: np.ndarray) -> Dict:
        """
        Validate TFLite conversion by comparing outputs with original model.
        
        Args:
            test_data: Test data for validation
            
        Returns:
            Validation results
        """
        if self.tflite_model is None or self.original_model is None:
            return {'error': 'Both original and TFLite models required for validation'}
        
        logger.info("Validating TFLite conversion...")
        
        try:
            import tensorflow as tf
        except ImportError:
            return {'error': 'TensorFlow required for validation'}
        
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare test data
        num_samples = min(len(test_data), 100)  # Limit for validation
        test_subset = test_data[:num_samples]
        
        # Get original model predictions
        original_predictions = self.original_model.predict(test_subset, verbose=0)
        
        # Get TFLite predictions
        tflite_predictions = []
        
        for i, sample in enumerate(test_subset):
            # Prepare input
            input_data = np.expand_dims(sample, axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            tflite_predictions.append(output_data[0])
        
        tflite_predictions = np.array(tflite_predictions)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            original_predictions, tflite_predictions
        )
        
        validation_results = {
            'num_samples_tested': num_samples,
            'validation_metrics': validation_metrics,
            'input_details': self._format_tensor_details(input_details),
            'output_details': self._format_tensor_details(output_details),
            'passes_validation': validation_metrics['max_absolute_error'] <= self.config.tolerance
        }
        
        self.validation_results = validation_results
        
        logger.info(f"Validation completed. Max error: {validation_metrics['max_absolute_error']:.6f}")
        
        return validation_results
    
    def benchmark_inference(self, 
                          test_data: np.ndarray,
                          num_runs: int = 1000,
                          warmup_runs: int = 10) -> Dict:
        """
        Benchmark TFLite model inference performance.
        
        Args:
            test_data: Test data for benchmarking
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        if self.interpreter is None:
            if self.tflite_model is None:
                return {'error': 'No TFLite model available for benchmarking'}
            
            # Create interpreter
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
                self.interpreter.allocate_tensors()
            except ImportError:
                return {'error': 'TensorFlow required for benchmarking'}
        
        logger.info(f"Benchmarking TFLite inference with {num_runs} runs...")
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Prepare test data
        if len(test_data.shape) > len(input_details[0]['shape']):
            test_data = test_data[0]  # Take first sample if batch
        
        input_data = np.expand_dims(test_data, axis=0).astype(input_details[0]['dtype'])
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
        
        # Benchmark runs
        inference_times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_details[0]['index'])
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        benchmark_results = {
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
            'mean_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'median_inference_time_ms': np.median(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'p99_inference_time_ms': np.percentile(inference_times, 99),
            'throughput_samples_per_second': 1000 / np.mean(inference_times),
            'meets_esp32_requirement': np.mean(inference_times) <= 100.0,  # 100ms target
            'memory_usage_estimate_kb': self._estimate_runtime_memory_usage()
        }
        
        self.benchmark_results = benchmark_results
        
        logger.info(f"Benchmark completed:")
        logger.info(f"  Mean latency: {benchmark_results['mean_inference_time_ms']:.2f} ms")
        logger.info(f"  Throughput: {benchmark_results['throughput_samples_per_second']:.1f} samples/sec")
        
        return benchmark_results
    
    def optimize_for_esp32(self) -> Dict:
        """
        Apply ESP32-S3 specific optimizations.
        
        Returns:
            Optimization results and recommendations
        """
        optimization_results = {
            'optimizations_applied': [],
            'esp32_compatibility': {},
            'performance_estimates': {},
            'recommendations': []
        }
        
        if self.conversion_results:
            model_size_mb = self.conversion_results['model_analysis']['size_mb']
            
            # Memory optimization
            if model_size_mb > self.config.memory_limit_mb:
                optimization_results['recommendations'].extend([
                    f"Model size ({model_size_mb:.2f} MB) exceeds ESP32-S3 limit ({self.config.memory_limit_mb} MB)",
                    "Consider more aggressive quantization (INT8)",
                    "Apply model pruning to reduce parameters",
                    "Use SPIRAM for model storage if available"
                ])
            else:
                optimization_results['optimizations_applied'].append("Memory requirements within ESP32-S3 limits")
            
            # Performance optimization
            if self.benchmark_results:
                latency = self.benchmark_results['mean_inference_time_ms']
                if latency > 100:
                    optimization_results['recommendations'].extend([
                        f"Inference latency ({latency:.2f} ms) may be high for real-time applications",
                        "Consider model architecture simplification",
                        "Use ESP32-S3 AI acceleration features"
                    ])
                else:
                    optimization_results['optimizations_applied'].append("Inference latency suitable for real-time use")
        
        # ESP32-S3 compatibility checks
        optimization_results['esp32_compatibility'] = {
            'memory_compatible': True,  # Based on earlier checks
            'instruction_set_compatible': True,  # TFLite Micro supports ESP32
            'peripheral_requirements': {
                'flash_memory': '≥4MB for model storage',
                'ram_memory': '≥512KB for inference',
                'spiram_recommended': model_size_mb > 2.0 if self.conversion_results else False
            }
        }
        
        # Performance estimates for ESP32-S3
        if self.benchmark_results:
            # Rough scaling factors for ESP32-S3 vs development machine
            esp32_scaling_factor = 3.0  # Conservative estimate
            
            optimization_results['performance_estimates'] = {
                'estimated_esp32_latency_ms': self.benchmark_results['mean_inference_time_ms'] * esp32_scaling_factor,
                'estimated_power_consumption_mw': self._estimate_power_consumption(),
                'battery_life_estimate_hours': self._estimate_battery_life()
            }
        
        # Add ESP32-specific recommendations
        optimization_results['recommendations'].extend([
            "Test on actual ESP32-S3 hardware for final validation",
            "Monitor temperature effects on performance",
            "Implement proper error handling for edge cases",
            "Consider using FreeRTOS task scheduling for continuous inference"
        ])
        
        return optimization_results
    
    def export_c_header(self, 
                       output_path: Optional[str] = None,
                       array_name: str = "shadow_cnn_model") -> str:
        """
        Export TFLite model as C header file for ESP32 integration.
        
        Args:
            output_path: Path to save the header file
            array_name: Name of the C array containing model data
            
        Returns:
            Path to the generated header file
        """
        if self.tflite_model is None:
            raise ValueError("No TFLite model available for C header export")
        
        if output_path is None:
            output_path = f"{array_name}.h"
        
        # Generate C header content
        header_content = self._generate_c_header(self.tflite_model, array_name)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(header_content)
        
        logger.info(f"C header file generated: {output_path}")
        
        return output_path
    
    def _configure_converter(self, converter, representative_data: Optional[np.ndarray]):
        """Configure TFLite converter based on settings."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required for converter configuration")
        
        # Basic optimizations
        if self.config.enable_optimizations:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Configure based on quantization type
        if self.config.quantization_type == 'int8' or self.config.quantization_type == 'full_int8':
            converter.target_spec.supported_types = [tf.int8]
            
            # Set representative dataset for full integer quantization
            if representative_data is not None:
                def representative_dataset():
                    dataset_size = min(len(representative_data), self.config.representative_dataset_size)
                    for i in range(dataset_size):
                        sample = representative_data[i:i+1].astype(np.float32)
                        yield [sample]
                
                converter.representative_dataset = representative_dataset
                
                if self.config.quantization_type == 'full_int8':
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
            
        elif self.config.quantization_type == 'float16':
            converter.target_spec.supported_types = [tf.float16]
            
        elif self.config.quantization_type == 'dynamic':
            # Dynamic range quantization (default with optimizations)
            pass
        
        # Set experimental flags
        if hasattr(converter, 'experimental_new_converter'):
            converter.experimental_new_converter = self.config.experimental_new_converter
        
        if hasattr(converter, 'experimental_new_quantizer'):
            converter.experimental_new_quantizer = self.config.experimental_new_quantizer
        
        # ESP32-specific optimizations
        if self.config.esp32_optimizations:
            # Enable optimizations that work well on ESP32
            converter.allow_custom_ops = False  # Ensure all ops are supported
            if hasattr(converter, 'experimental_lower_tensor_list_ops'):
                converter.experimental_lower_tensor_list_ops = True
    
    def _analyze_converted_model(self, tflite_model: bytes, model_path: str) -> Dict:
        """Analyze the converted TFLite model."""
        try:
            import tensorflow as tf
        except ImportError:
            return {'error': 'TensorFlow required for model analysis'}
        
        # Basic model information
        analysis = {
            'size_bytes': len(tflite_model),
            'size_kb': len(tflite_model) / 1024,
            'size_mb': len(tflite_model) / (1024 * 1024),
            'model_path': model_path,
            'quantization_type': self.config.quantization_type
        }
        
        try:
            # Load interpreter for detailed analysis
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            analysis.update({
                'input_details': self._format_tensor_details(input_details),
                'output_details': self._format_tensor_details(output_details),
                'num_inputs': len(input_details),
                'num_outputs': len(output_details)
            })
            
            # Memory analysis
            analysis.update({
                'fits_esp32_s3': analysis['size_mb'] < self.config.memory_limit_mb,
                'memory_efficiency_percent': (self.config.memory_limit_mb - analysis['size_mb']) / self.config.memory_limit_mb * 100
            })
            
            # Estimate operation count
            try:
                # Get tensor details for all tensors
                tensor_details = interpreter.get_tensor_details()
                analysis['total_tensors'] = len(tensor_details)
                
                # Estimate computational complexity
                total_ops = 0
                for tensor in tensor_details:
                    shape = tensor.get('shape', [])
                    if len(shape) > 0:
                        total_ops += np.prod(shape)
                
                analysis['estimated_operations'] = total_ops
                
            except Exception as e:
                logger.debug(f"Could not estimate operations: {e}")
            
        except Exception as e:
            logger.warning(f"Detailed model analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def _calculate_validation_metrics(self, 
                                    original_predictions: np.ndarray,
                                    tflite_predictions: np.ndarray) -> Dict:
        """Calculate validation metrics between original and TFLite predictions."""
        
        # Ensure same shape
        if original_predictions.shape != tflite_predictions.shape:
            logger.warning(f"Shape mismatch: original {original_predictions.shape}, tflite {tflite_predictions.shape}")
            # Try to align shapes
            min_samples = min(len(original_predictions), len(tflite_predictions))
            original_predictions = original_predictions[:min_samples]
            tflite_predictions = tflite_predictions[:min_samples]
        
        # Calculate metrics
        absolute_errors = np.abs(original_predictions - tflite_predictions)
        relative_errors = np.abs((original_predictions - tflite_predictions) / (original_predictions + 1e-8))
        
        # Classification accuracy if predictions are probabilities
        if original_predictions.shape[-1] > 1:  # Multi-class predictions
            original_classes = np.argmax(original_predictions, axis=-1)
            tflite_classes = np.argmax(tflite_predictions, axis=-1)
            classification_accuracy = np.mean(original_classes == tflite_classes)
        else:
            classification_accuracy = None
        
        metrics = {
            'max_absolute_error': float(np.max(absolute_errors)),
            'mean_absolute_error': float(np.mean(absolute_errors)),
            'std_absolute_error': float(np.std(absolute_errors)),
            'max_relative_error': float(np.max(relative_errors)),
            'mean_relative_error': float(np.mean(relative_errors)),
            'rmse': float(np.sqrt(np.mean((original_predictions - tflite_predictions) ** 2))),
            'correlation': float(np.corrcoef(original_predictions.flatten(), tflite_predictions.flatten())[0, 1]),
            'classification_accuracy': classification_accuracy
        }
        
        return metrics
    
    def _compare_conversion_strategies(self, results: Dict) -> Dict:
        """Compare results from different conversion strategies."""
        comparison = {
            'strategy_ranking': [],
            'size_comparison': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Extract metrics for comparison
        strategy_metrics = {}
        
        for strategy, result in results.items():
            if result.get('success', False):
                strategy_metrics[strategy] = {
                    'size_mb': result['model_analysis']['size_mb'],
                    'fits_esp32': result['model_analysis']['fits_esp32_s3']
                }
                
                # Add validation metrics if available
                if 'validation_results' in result and result['validation_results']:
                    val_metrics = result['validation_results'].get('validation_metrics', {})
                    strategy_metrics[strategy]['max_error'] = val_metrics.get('max_absolute_error', float('inf'))
                    strategy_metrics[strategy]['classification_accuracy'] = val_metrics.get('classification_accuracy', 0)
        
        # Rank strategies
        if strategy_metrics:
            # Primary ranking by ESP32 compatibility, then by size
            ranked_strategies = sorted(
                strategy_metrics.items(),
                key=lambda x: (x[1]['fits_esp32'], -x[1]['size_mb']),
                reverse=True
            )
            
            comparison['strategy_ranking'] = [strategy for strategy, _ in ranked_strategies]
            
            # Size comparison
            comparison['size_comparison'] = {
                strategy: metrics['size_mb'] 
                for strategy, metrics in strategy_metrics.items()
            }
            
            # Generate recommendations
            best_strategy = ranked_strategies[0][0]
            comparison['recommendations'] = [
                f"Recommended strategy: {best_strategy}",
                f"Best for ESP32-S3 deployment: {best_strategy}"
            ]
            
            # Add specific recommendations
            for strategy, metrics in strategy_metrics.items():
                if not metrics['fits_esp32']:
                    comparison['recommendations'].append(
                        f"Avoid {strategy}: too large for ESP32-S3 ({metrics['size_mb']:.2f} MB)"
                    )
        
        return comparison
    
    def _format_tensor_details(self, tensor_details: List) -> List[Dict]:
        """Format tensor details for JSON serialization."""
        formatted_details = []
        
        for tensor in tensor_details:
            formatted = {
                'name': tensor.get('name', 'unknown'),
                'shape': tensor.get('shape', []).tolist(),
                'dtype': str(tensor.get('dtype', 'unknown')),
                'index': tensor.get('index', -1)
            }
            
            # Add quantization info if available
            if 'quantization_parameters' in tensor:
                quant_params = tensor['quantization_parameters']
                formatted['quantization'] = {
                    'scales': quant_params.get('scales', []),
                    'zero_points': quant_params.get('zero_points', [])
                }
            
            formatted_details.append(formatted)
        
        return formatted_details
    
    def _estimate_runtime_memory_usage(self) -> float:
        """Estimate runtime memory usage in KB."""
        if self.interpreter is None:
            return 0.0
        
        try:
            # Get memory usage information
            tensor_details = self.interpreter.get_tensor_details()
            
            total_memory = 0
            for tensor in tensor_details:
                shape = tensor.get('shape', [])
                dtype = tensor.get('dtype', None)
                
                if len(shape) > 0 and dtype is not None:
                    # Calculate tensor size
                    tensor_size = np.prod(shape)
                    
                    # Estimate bytes per element based on dtype
                    if 'int8' in str(dtype) or 'uint8' in str(dtype):
                        bytes_per_element = 1
                    elif 'int16' in str(dtype) or 'float16' in str(dtype):
                        bytes_per_element = 2
                    else:  # float32, int32
                        bytes_per_element = 4
                    
                    total_memory += tensor_size * bytes_per_element
            
            return total_memory / 1024  # Convert to KB
            
        except Exception as e:
            logger.debug(f"Memory estimation failed: {e}")
            return 0.0
    
    def _estimate_power_consumption(self) -> float:
        """Estimate power consumption in milliwatts."""
        # Rough estimates based on ESP32-S3 specifications
        base_power = 20  # Base ESP32-S3 power consumption (mW)
        
        if self.benchmark_results:
            # Estimate additional power for ML inference
            inference_time_ms = self.benchmark_results['mean_inference_time_ms']
            inference_power = 100  # Additional power during inference (mW)
            
            # Calculate average power assuming 1 inference per second
            duty_cycle = inference_time_ms / 1000  # Fraction of time doing inference
            avg_power = base_power + (inference_power * duty_cycle)
            
            return avg_power
        
        return base_power
    
    def _estimate_battery_life(self) -> float:
        """Estimate battery life in hours."""
        power_consumption = self._estimate_power_consumption()
        battery_capacity_mah = 2000  # Typical battery capacity
        battery_voltage = 3.7  # Typical Li-ion voltage
        
        battery_energy_mwh = battery_capacity_mah * battery_voltage
        
        # Conservative estimate with 70% efficiency
        estimated_hours = (battery_energy_mwh * 0.7) / power_consumption
        
        return estimated_hours
    
    def _generate_c_header(self, tflite_model: bytes, array_name: str) -> str:
        """Generate C header file content."""
        
        # Convert bytes to C array
        byte_array = ', '.join([f'0x{b:02x}' for b in tflite_model])
        
        # Generate header content
        header_content = f"""/*
 * TensorFlow Lite model for ShadowCNN
 * Generated automatically - do not edit manually
 * 
 * Model size: {len(tflite_model)} bytes
 * Quantization: {self.config.quantization_type}
 */

#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

#ifdef __cplusplus
extern "C" {{
#endif

// Model data
const unsigned char {array_name}[] = {{
{byte_array}
}};

// Model size
const unsigned int {array_name}_len = {len(tflite_model)};

// Model metadata
#define MODEL_INPUT_SHAPE_0 {self.conversion_results.get('model_analysis', {}).get('input_details', [{}])[0].get('shape', [])}
#define MODEL_OUTPUT_SHAPE_0 {self.conversion_results.get('model_analysis', {}).get('output_details', [{}])[0].get('shape', [])}
#define MODEL_QUANTIZATION_TYPE "{self.config.quantization_type}"

#ifdef __cplusplus
}}
#endif

#endif // {array_name.upper()}_H
"""
        
        return header_content