"""
PyTorch to TFLite INT8 Converter using AI Edge Torch
=====================================================

Google's Official PyTorch → TFLite Conversion Tool
Colab: https://colab.research.google.com/

This is the RECOMMENDED approach for converting PyTorch models to TFLite!
Much simpler than ONNX → TensorFlow → TFLite pipeline.

References:
- https://github.com/google-ai-edge/ai-edge-torch
- https://ai.google.dev/edge/litert/models/pytorch_to_tflite

Installation:
    !pip install torch ai-edge-torch tensorflow

Usage in Colab:
    1. Upload best.pth
    2. Run this script
    3. Download stress_model_quant.tflite
"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

print("=" * 80)
print("PyTorch → TFLite INT8 using AI Edge Torch")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
print(f"TensorFlow: {tf.__version__}")

try:
    import ai_edge_torch
    print(f"AI Edge Torch: {ai_edge_torch.__version__}")
except ImportError:
    print("\n❌ AI Edge Torch not installed!")
    print("   Run: pip install ai-edge-torch")
    exit(1)


# ============================================================================
# Model Architecture
# ============================================================================

class StressDetectionCNN(nn.Module):
    def __init__(self):
        super(StressDetectionCNN, self).__init__()
        
        self.shared_conv = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=10, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=10, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        self.universal_private = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.shared_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        x = self.universal_private(x)
        return x


# ============================================================================
# Load PyTorch Model
# ============================================================================

def load_model(model_path='best.pth'):
    print("\n" + "=" * 80)
    print("Loading PyTorch Model")
    print("=" * 80)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = StressDetectionCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ============================================================================
# Convert to TFLite with Quantization
# ============================================================================

def convert_to_tflite(pytorch_model, output_path='stress_model_quant.tflite'):
    print("\n" + "=" * 80)
    print("Converting PyTorch → TFLite INT8")
    print("=" * 80)
    
    # Prepare sample input
    sample_input = torch.randn(1, 4, 240)
    
    print("   Input shape:", tuple(sample_input.shape))
    print("   Converting...")
    
    # Convert to TFLite with ai-edge-torch
    edge_model = ai_edge_torch.convert(
        pytorch_model,
        (sample_input,)
    )
    
    print("✅ Conversion complete (FLOAT32)")
    
    # Now quantize to INT8
    print("\n   Applying INT8 quantization...")
    
    # Create representative dataset
    def representative_dataset():
        for _ in range(100):
            yield [np.random.randn(1, 4, 240).astype(np.float32)]
    
    # Load the float model and quantize
    converter = tf.lite.TFLiteConverter.from_saved_model(
        edge_model.export_saved_model()
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Quantized model saved: {output_path}")
    print(f"   Size: {len(tflite_model) / 1024:.2f} KB")
    
    return tflite_model


# ============================================================================
# Validate
# ============================================================================

def validate(model_path='stress_model_quant.tflite'):
    print("\n" + "=" * 80)
    print("Validating TFLite Model")
    print("=" * 80)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"   Input: {input_details['shape']} {input_details['dtype']}")
    print(f"   Output: {output_details['shape']} {output_details['dtype']}")
    print("✅ Model is ready for ESP32-S3!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    model = load_model('best.pth')
    convert_to_tflite(model)
    validate()
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print("\nNext: Download stress_model_quant.tflite")
    print("Then: xxd -i stress_model_quant.tflite > stress_model_data.c")
