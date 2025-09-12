# Shadow ML: Python vs ESP32 Model Comparison Report

Generated: 2025-09-12 06:56:20.620720

## Model Specifications
- **Original Model**: Python sklearn MLPClassifier with isotonic calibration
- **Quantized Model**: ESP32 C implementation (Float32 → Int16)
- **Test Dataset**: 1095 samples, 30 features
- **Ground Truth Distribution**: 458 No-Stress, 637 Stress

## Performance Metrics Comparison

| Metric    | Python   | ESP32    | Difference | Relative Error |
|-----------|----------|----------|------------|----------------|
| Accuracy  | 0.990868 | 0.990868 | 0.000000 | 0.0000% |
| Precision | 0.993701 | 0.993701 | 0.000000 | 0.0000% |
| Recall    | 0.990581 | 0.990581 | 0.000000 | 0.0000% |
| F1 Score  | 0.992138 | 0.992138 | 0.000000 | 0.0000% |

## Prediction Analysis
- **Prediction Agreement**: 100.00%
- **Exact Matches**: 1095/1095
- **Disagreements**: 0/1095

## Probability Analysis
- **Correlation Coefficient**: 0.99999997
- **Maximum Probability Difference**: 0.00049700
- **Mean Probability Difference**: 0.00003702
- **Samples with Exact Probability Match**: 60/1095

## Confusion Matrices

### Python Model
```
              Predicted
           No-Stress  Stress
Actual No-Stress   454        4
       Stress        6      631
```

### ESP32 Model
```
              Predicted
           No-Stress  Stress
Actual No-Stress   454        4
       Stress        6      631
```

## Quantization Quality Assessment

### ✅ **EXCELLENT QUANTIZATION RESULTS**

The quantized ESP32 model demonstrates:
- **Zero prediction disagreements** (100.0% agreement)
- **Negligible probability differences** (max: 0.000497)
- **Identical performance metrics** (differences < 0.000001)
- **Perfect correlation** (r = 0.99999997)

### Conclusion
The ESP32 quantized model is **production-ready** and maintains identical performance to the original Python model. This represents an optimal quantization result with no measurable accuracy loss.

### Deployment Recommendations
1. ✅ **Deploy to ESP32** - quantization preserves full accuracy
2. ✅ **Use in production** - identical predictions to original model
3. ✅ **Real-time inference** - suitable for edge deployment
4. ✅ **Memory efficient** - reduced footprint with same performance
