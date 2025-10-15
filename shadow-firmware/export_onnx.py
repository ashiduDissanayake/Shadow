#!/usr/bin/env python3
"""
Export PyTorch Model to ONNX Format
===================================

Simple script to export the trained PyTorch model to ONNX format.
ONNX can then be converted to TFLite using online tools or alternative converters.

Usage:
    python3 export_onnx.py
"""

import torch
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "best.pth"
OUTPUT_DIR = Path(__file__).parent / "model_output"
ONNX_PATH = OUTPUT_DIR / "stress_model.onnx"

OUTPUT_DIR.mkdir(exist_ok=True)


class StressDetectionCNN(torch.nn.Module):
    def __init__(self):
        super(StressDetectionCNN, self).__init__()
        
        self.shared_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=4, out_channels=64, kernel_size=10, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool1d(kernel_size=2),
        )
        
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        self.shared_fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        
        self.universal_private = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.shared_conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        x = self.universal_private(x)
        return x


print("Loading PyTorch model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = StressDetectionCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())} parameters)")

# Test inference
sample_input = torch.randn(1, 4, 240)
with torch.no_grad():
    output = model(sample_input)
print(f"✅ Test inference: input {list(sample_input.shape)} → output {list(output.shape)}")
print(f"   Output value: {output[0, 0].item():.6f}")

# Export to ONNX
print(f"\nExporting to ONNX: {ONNX_PATH}")
torch.onnx.export(
    model,
    sample_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    }
)

print(f"✅ ONNX export complete: {ONNX_PATH.stat().st_size / 1024:.2f} KB")
print("\nNext steps:")
print("1. Use ONNX model directly with ESP-NN or TensorFlow Lite Micro")
print("2. Convert ONNX to TFLite using: https://netron.app or onnx-tensorflow")
print("3. Or use model-serving/convert_to_tflite.py if available")
