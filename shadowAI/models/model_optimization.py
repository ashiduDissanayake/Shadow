#!/usr/bin/env python3
"""
Model Optimization and TensorFlow Lite Conversion for ESP32-S3 Deployment

This script handles the conversion of trained H-CNN models to TensorFlow Lite format
with quantization for deployment on resource-constrained devices like the ESP32-S3.
"""

import numpy as np
import tensorflow as tf
import os
import json
import logging
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles model optimization and conversion for TinyML deployment."""
    
    def __init__(self, model_path, scaler_path, config_path=None):
        """
        Initialize the model optimizer.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config = self._load_config(config_path)
        
        # Load model and scaler
        self.model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded scaler from {scaler_path}")
    
    def _load_config(self, config_path):
        """Load optimization configuration."""
        default_config = {
            'optimization': {
                'enable_quantization': True,
                'quantization_type': 'int8',  # 'int8' or 'float16'
                'representative_samples': 1000,
                'optimize_for_size': True
            },
            'output': {
                'models_dir': 'models',
                'tflite_dir': 'models/tflite',
                'esp32_dir': 'models/esp32'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def generate_representative_dataset(self, num_samples=1000):
        """
        Generate representative dataset for quantization.
        
        Args:
            num_samples: Number of representative samples
            
        Returns:
            Generator function for representative dataset
        """
        # Generate synthetic representative data
        # In a real scenario, this would use actual validation data
        
        def representative_dataset_gen():
            for i in range(num_samples):
                # Generate synthetic BVP segment
                segment_length = 3840  # 60 seconds * 64 Hz
                bvp_segment = np.random.randn(segment_length).astype(np.float32)
                bvp_segment = np.expand_dims(bvp_segment, axis=0)  # Add batch dimension
                bvp_segment = np.expand_dims(bvp_segment, axis=-1)  # Add channel dimension
                
                # Generate synthetic features
                num_features = 9  # Based on our feature extraction
                features = np.random.randn(1, num_features).astype(np.float32)
                
                yield [bvp_segment, features]
        
        return representative_dataset_gen
    
    def convert_to_tflite(self, output_path=None, quantize=True):
        """
        Convert the model to TensorFlow Lite format.
        
        Args:
            output_path: Output path for TFLite model
            quantize: Whether to apply quantization
            
        Returns:
            Path to the converted model
        """
        logger.info("Converting model to TensorFlow Lite format...")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config['output']['tflite_dir']}/hcnn_model_{timestamp}.tflite"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize converter
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        
        if quantize and self.config['optimization']['enable_quantization']:
            logger.info("Applying quantization...")
            
            # Set optimization flags
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set representative dataset
            representative_dataset = self.generate_representative_dataset(
                self.config['optimization']['representative_samples']
            )
            converter.representative_dataset = representative_dataset
            
            # Set target specs
            if self.config['optimization']['quantization_type'] == 'int8':
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:  # float16
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                converter.inference_input_type = tf.float16
                converter.inference_output_type = tf.float16
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        logger.info(f"TFLite model saved to {output_path}")
        logger.info(f"Model size: {model_size:.2f} MB")
        
        return output_path, model_size
    
    def validate_tflite_model(self, tflite_path):
        """
        Validate the TFLite model by running inference.
        
        Args:
            tflite_path: Path to TFLite model
            
        Returns:
            Validation results
        """
        logger.info("Validating TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info("Input details:")
        for i, detail in enumerate(input_details):
            logger.info(f"  Input {i}: {detail}")
        
        logger.info("Output details:")
        for i, detail in enumerate(output_details):
            logger.info(f"  Output {i}: {detail}")
        
        # Test inference with sample data
        test_segment = np.random.randn(1, 3840, 1).astype(np.float32)
        test_features = np.random.randn(1, 9).astype(np.float32)
        
        # Set input tensors
        interpreter.set_tensor(input_details[0]['index'], test_segment)
        interpreter.set_tensor(input_details[1]['index'], test_features)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"Test inference successful. Output shape: {output.shape}")
        logger.info(f"Output values: {output}")
        
        return {
            'input_details': input_details,
            'output_details': output_details,
            'test_output': output
        }
    
    def create_esp32_header(self, tflite_path, output_path=None):
        """
        Create C header file for ESP32 deployment.
        
        Args:
            tflite_path: Path to TFLite model
            output_path: Output path for header file
            
        Returns:
            Path to the header file
        """
        logger.info("Creating ESP32 header file...")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config['output']['esp32_dir']}/model_data_{timestamp}.h"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read TFLite model
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        # Convert to C array
        model_array = []
        for byte in model_data:
            model_array.append(f"0x{byte:02x}")
        
        # Create header file content
        header_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Auto-generated header file for ESP32 deployment
// Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// Model size: {len(model_data)} bytes

const unsigned char model_data[] = {{
    {', '.join(model_array)}
}};

const unsigned int model_data_len = {len(model_data)};

#endif // MODEL_DATA_H
"""
        
        # Write header file
        with open(output_path, 'w') as f:
            f.write(header_content)
        
        logger.info(f"ESP32 header file saved to {output_path}")
        return output_path
    
    def create_model_config(self, tflite_path, output_path=None):
        """
        Create model configuration file for deployment.
        
        Args:
            tflite_path: Path to TFLite model
            output_path: Output path for config file
            
        Returns:
            Path to the config file
        """
        logger.info("Creating model configuration file...")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.config['output']['esp32_dir']}/model_config_{timestamp}.json"
        
        # Get model info
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create config
        config = {
            'model_info': {
                'name': 'H-CNN Stress Detection Model',
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'model_path': tflite_path,
                'model_size_bytes': os.path.getsize(tflite_path)
            },
            'inputs': [
                {
                    'name': detail['name'],
                    'shape': detail['shape'].tolist(),
                    'dtype': str(detail['dtype']),
                    'index': detail['index']
                }
                for detail in input_details
            ],
            'outputs': [
                {
                    'name': detail['name'],
                    'shape': detail['shape'].tolist(),
                    'dtype': str(detail['dtype']),
                    'index': detail['index']
                }
                for detail in output_details
            ],
            'preprocessing': {
                'segment_length': 3840,  # 60 seconds * 64 Hz
                'feature_count': 9,
                'sampling_rate': 64,
                'filter_lowcut': 0.7,
                'filter_highcut': 3.7
            },
            'classes': ['Baseline', 'Stress', 'Amusement']
        }
        
        # Write config file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model configuration saved to {output_path}")
        return output_path
    
    def optimize_for_esp32(self):
        """
        Complete optimization pipeline for ESP32 deployment.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting ESP32 optimization pipeline...")
        
        results = {}
        
        # 1. Convert to TFLite with quantization
        tflite_path, model_size = self.convert_to_tflite(quantize=True)
        results['tflite_path'] = tflite_path
        results['model_size_mb'] = model_size
        
        # 2. Validate TFLite model
        validation_results = self.validate_tflite_model(tflite_path)
        results['validation'] = validation_results
        
        # 3. Create ESP32 header file
        header_path = self.create_esp32_header(tflite_path)
        results['header_path'] = header_path
        
        # 4. Create model configuration
        config_path = self.create_model_config(tflite_path)
        results['config_path'] = config_path
        
        logger.info("ESP32 optimization completed successfully!")
        
        return results


def main():
    """Main function to run the optimization pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize H-CNN model for ESP32 deployment')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--scaler_path', required=True, help='Path to feature scaler')
    parser.add_argument('--config_path', help='Path to optimization config')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.model_path, args.scaler_path, args.config_path)
    
    # Run optimization
    results = optimizer.optimize_for_esp32()
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"TFLite model: {results['tflite_path']}")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"ESP32 header: {results['header_path']}")
    print(f"Model config: {results['config_path']}")
    print("="*50)


if __name__ == "__main__":
    main()
