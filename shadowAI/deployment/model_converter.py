import tensorflow as tf
import os

def convert_model_to_tflite(saved_model_path, output_path):
    """
    Converts a saved TensorFlow Keras model to TensorFlow Lite format.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a saved Keras model at this path
    SAVED_MODEL_PATH = 'models/saved_hcnn_model'
    OUTPUT_TFLITE_PATH = 'models/hcnn_model.tflite'

    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Saved model not found at {SAVED_MODEL_PATH}. Please run model_development.ipynb first.")
    else:
        convert_model_to_tflite(SAVED_MODEL_PATH, OUTPUT_TFLITE_PATH)


