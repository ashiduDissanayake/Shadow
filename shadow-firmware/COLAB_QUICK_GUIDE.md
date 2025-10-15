# 🚀 Quick Guide: Convert ONNX to TFLite using Google Colab

## ⏱️ Time Required: 10 minutes

---

## Step 1: Open Google Colab
Go to: https://colab.research.google.com/

---

## Step 2: Copy & Paste This Code

Copy the **ENTIRE contents** of `COLAB_CONVERSION_NOTEBOOK.py` and paste into a new Colab cell.

---

## Step 3: Run the Cell

Click the "Play" button or press `Shift + Enter`

You'll see:
```
================================================================================
Installing dependencies...
================================================================================
```

Wait 30-60 seconds for packages to install.

---

## Step 4: Upload Your ONNX File

When prompted:
```
Upload your ONNX model
Click the file upload button and select: stress_model.onnx
```

1. Click the "Choose Files" button that appears
2. Navigate to: `/Users/ashidudissanayake/Dev/Shadow/shadow-firmware/model_output/`
3. Select: `stress_model.onnx`
4. Wait for upload (431 KB, ~5 seconds)

---

## Step 5: Watch the Magic Happen ✨

The script will automatically:
- ✅ Convert ONNX → TensorFlow (1-2 minutes)
- ✅ Convert TensorFlow → TFLite Float32
- ✅ Convert TensorFlow → TFLite INT8 Quantized
- ✅ Validate the models
- ✅ Generate C header preview
- ✅ Trigger downloads

---

## Step 6: Download Your Model

Two files will automatically download:
1. **`stress_model_quant.tflite`** ← **USE THIS FOR ESP32**
2. `stress_model_float.tflite` (for comparison)

---

## Step 7: Verify Success

Check the output for:
```
✅ INT8 Quantized TFLite model: ~100-150 KB
✅ Model validation successful
✅ CONVERSION COMPLETE!
```

---

## Step 8: Copy Model to Project

Move the downloaded file:
```bash
mv ~/Downloads/stress_model_quant.tflite /Users/ashidudissanayake/Dev/Shadow/shadow-firmware/model_output/
```

---

## 🎉 You're Done!

You now have:
- ✅ `stress_model_quant.tflite` (INT8 quantized, ~100-150 KB)
- ✅ Ready to integrate with ESP32-S3

---

## ⚠️ Troubleshooting

### Issue: Import errors during dependency installation
**Solution:** Colab updates packages frequently. Just re-run the cell.

### Issue: "onnx2tf" not found
**Solution:** Make sure you updated the code to use `onnx2tf` instead of `onnx-tf`

### Issue: Conversion takes too long (>5 minutes)
**Solution:** The first conversion is slower. Be patient, or restart runtime and try again.

### Issue: Model file won't download
**Solution:** Right-click the download link in Colab's file browser (left sidebar) and select "Download"

---

## 📝 Expected Output

```
================================================================================
✅ CONVERSION COMPLETE!
================================================================================

Models created:
  • Float32:       400-450 KB
  • INT8 Quantized: 100-150 KB (← Use this for ESP32)

Model specifications:
  Input:  (1, 4, 240) float32 - [ACC_MAG, BVP, EDA, TEMP]
  Output: (1, 1) float32 - Stress probability [0.0-1.0]
  Quantization: INT8 weights/activations, FLOAT32 input/output

🎉 Ready for ESP32 deployment!
```

---

## 🔜 Next Steps (After Conversion)

1. **Generate C arrays** from the TFLite model
2. **Create `cnn_inference` component** in ESP-IDF
3. **Integrate with signal preprocessor**
4. **Test on ESP32-S3 hardware**

See `PHASE2_PROGRESS.md` for detailed next steps.

---

**Good luck! You're almost there!** 🚀
