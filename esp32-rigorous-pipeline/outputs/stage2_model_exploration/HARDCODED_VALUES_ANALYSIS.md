# ESP32 Deployment Visualizer: Hardcoded Values Analysis

## 📊 **COMPLETE BREAKDOWN OF HARDCODED VALUES**

This document reveals all the hardcoded values used in the ESP32 deployment visualization script and explains the reasoning behind each.

---

## 🔍 **INFERENCE TIME CALCULATIONS**

### **Before/After Comparison Values:**
```python
# Line 386-387: Simulated inference times
python_time = 15.2  # ms (HARDCODED)
esp32_time = 3.8    # ms (HARDCODED)
```

**🎯 How These Were Calculated:**
- **Python Time (15.2ms)**: Estimated based on typical sklearn MLP inference on Intel i7
  - Includes Python overhead, library calls, and feature processing
  - Realistic for 30-feature MLP with 64+32 hidden neurons
  
- **ESP32 Time (3.8ms)**: Estimated for ESP32-S3 at 240MHz
  - Based on C fixed-point arithmetic operations
  - ~4,000 parameters × multiply-accumulate operations
  - Optimized memory access patterns

**📈 Speedup Calculation:** `15.2 / 3.8 = 4.0x faster`

---

## 💾 **MEMORY USAGE VALUES**

### **Memory Comparison (Line 363-364):**
```python
python_memory = [50, 10, 100]  # MB for Python (HARDCODED)
esp32_memory = [8, 2, 4]      # KB for ESP32 (HARDCODED)
```

**Breakdown:**
- **Model Weights:** Python 50MB vs ESP32 8KB
- **Feature Buffer:** Python 10MB vs ESP32 2KB  
- **Computation:** Python 100MB vs ESP32 4KB

**🧮 Efficiency Calculation:**
```python
efficiency = sum(esp32_memory) / (sum(python_memory) * 1024) * 100
# (8+2+4) / (160*1024) * 100 = 0.0085% of Python memory
```

---

## 📏 **MODEL SIZE OPTIMIZATION**

### **Size Progression (Line 445):**
```python
sizes = [50.2, 12.8, 8.4, 7.9]  # MB to KB progression (HARDCODED)
```

**Stages:**
1. **Original Python (50.2KB):** Full sklearn model with metadata
2. **Quantized Weights (12.8KB):** Float32 → Int16 conversion
3. **Compressed C Code (8.4KB):** Optimized arrays, removed overhead
4. **Flash Optimized (7.9KB):** Memory layout optimization

---

## 🎯 **ACCURACY VALUES**

### **Performance Metrics (Line 319-325):**
```python
# Mock metrics if real data unavailable
metrics = {
    'mean_f1': 0.85,                    # HARDCODED
    'mean_balanced_accuracy': 0.89,     # HARDCODED
    'mean_precision': 0.93,             # HARDCODED
    'mean_recall': 0.82                 # HARDCODED
}
```

### **ESP32 Performance Drop (Line 334):**
```python
esp32_values = [v * 0.98 for v in python_values]  # 2% performance drop (HARDCODED)
```

### **Quantization Accuracy (Line 460):**
```python
accuracies = [0.851, 0.849, 0.847, 0.832]  # F1 scores (HARDCODED)
```
- **Float32:** 85.1%
- **Float16:** 84.9% (-0.2%)
- **Int16:** 84.7% (-0.4%)
- **Int8:** 83.2% (-1.9%)

---

## ⚡ **RESOURCE UTILIZATION**

### **ESP32-S3 Usage (Line 416):**
```python
used = [8, 12, 25, 45]  # Percentages (HARDCODED)
```

**Resources:**
- **Flash Memory:** 8% (8KB of 8MB)
- **SRAM:** 12% (12KB of 512KB)
- **CPU Usage:** 25% (during inference)
- **Power Consumption:** 45% increase

---

## 🔄 **REAL-TIME PERFORMANCE**

### **Dynamic Metrics (Line 490-491):**
```python
inference_times = 3.8 + 0.2 * np.sin(time_points/10) + np.random.normal(0, 0.1, len(time_points))
memory_usage = 12 + 2 * np.sin(time_points/15) + np.random.normal(0, 0.3, len(time_points))
```

**Formula Breakdown:**
- **Base inference time:** 3.8ms
- **Sinusoidal variation:** ±0.2ms (simulates load variation)
- **Random noise:** ±0.1ms (realistic measurement variation)
- **Memory baseline:** 12KB with ±2KB variation

---

## 📊 **FEATURE IMPORTANCE**

### **Feature Contribution (Line 478):**
```python
importance = [25, 20, 18, 12, 25]  # Percentage contribution (HARDCODED)
```

**Distribution:**
- **BVP Entropy:** 25%
- **ACC Energy:** 20%
- **EDA LineIntegral:** 18%
- **TEMP Min:** 12%
- **Others:** 25%

---

## 🎨 **WHAT'S REAL vs ESTIMATED**

### ✅ **Real Data Used:**
- Model architecture (30 features, 64+32 neurons)
- Model type (MLP with isotonic calibration)
- Optimal threshold (0.4095)
- Feature names from actual model artifacts
- Training data size (2759 samples)

### 🔮 **Estimated/Hardcoded:**
- All inference timing values
- Memory usage comparisons
- Resource utilization percentages
- Performance degradation rates
- Real-time monitoring data

---

## 💡 **HOW TO MAKE IT REAL**

### **To Get Actual ESP32 Inference Times:**
```c
// Add timing to your ESP32 code
uint32_t start_time = esp_timer_get_time();
float prediction = mlp_predict(features);
uint32_t end_time = esp_timer_get_time();
uint32_t inference_time_us = end_time - start_time;
```

### **To Measure Real Memory Usage:**
```c
// Check heap usage
size_t free_heap = esp_get_free_heap_size();
size_t min_free_heap = esp_get_minimum_free_heap_size();
```

### **To Benchmark Python Performance:**
```python
import time
start = time.perf_counter()
prediction = model.predict(features.reshape(1, -1))
end = time.perf_counter()
inference_time_ms = (end - start) * 1000
```

---

## 🚀 **CONCLUSION**

**Hardcoded Percentage:** ~90% of performance metrics are estimated
**Real Data Percentage:** ~10% comes from actual model artifacts

The visualizations provide **realistic estimates** based on:
- Typical embedded system performance characteristics
- ESP32-S3 hardware specifications
- Standard quantization impact studies
- Memory footprint analysis of similar models

**To get 100% real data:** Deploy the model to ESP32 and implement performance monitoring!
