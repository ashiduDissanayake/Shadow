import tensorflow as tf
import numpy as np
import os

def representative_dataset_gen(data_path=\""): # data_path is a placeholder, in real scenario it would load actual data
    # This function should yield concrete input examples for quantization.
    # For demonstration, we'll use dummy data. In a real scenario, you would load
    # a subset of your training data here.
    print("Generating representative dataset for quantization...")
    # Assuming X_seg_train and X_feat_train are available from your preprocessing
    # For this script, we'll create dummy data that matches the expected input shape
    # from the H-CNN model.
    segment_length_samples = 60 * 64 # 60 seconds * 64 Hz
    num_features = 7 # Based on the dummy features in model_development.ipynb
    num_representative_samples = 100 # Number of samples to use for quantization

    for _ in range(num_representative_samples):
        dummy_segment = np.random.rand(1, segment_length_samples, 1).astype(np.float32)
        dummy_feature = np.random.rand(1, num_features).astype(np.float32)
        yield [dummy_segment, dummy_feature]

def quantize_model(saved_model_path, output_path, representative_data_loader=None):
    """
    Quantizes a saved TensorFlow Keras model to TensorFlow Lite integer format.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_data_loader:
        converter.representative_dataset = representative_data_loader
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFL_OPS, tf.lite.OpsSet.TFL_OPS_V2]
        converter.inference_input_type = tf.int8  # Input type for inference
        converter.inference_output_type = tf.int8 # Output type for inference
    else:
        print("Warning: No representative dataset provided. Performing dynamic range quantization.")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFL_OPS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_quant_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized TFLite model saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    SAVED_MODEL_PATH = 'models/saved_hcnn_model'
    OUTPUT_QUANT_TFLITE_PATH = 'models/hcnn_model_quant.tflite'

    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Saved model not found at {SAVED_MODEL_PATH}. Please run model_development.ipynb first.")
    else:
        quantize_model(SAVED_MODEL_PATH, OUTPUT_QUANT_TFLITE_PATH, representative_dataset_gen)


