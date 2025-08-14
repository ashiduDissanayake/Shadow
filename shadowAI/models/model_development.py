# TinyML-Ready BVP Stress Detection (Laptop Training → ESP32-S3 Inference)
# -----------------------------------------------------------------------
# This single file provides:
# 1) A compact training pipeline for a small 1D CNN using short BVP windows (8 s @ 64 Hz)
# 2) Causal band-pass filtering + z-normalization consistent with on-device preprocessing
# 3) Full-integer (INT8) TFLite conversion with a representative dataset
# 4) Export of .tflite and a C header array for TFLite Micro
# 5) Export of filter coefficients and normalization stats to a C header
# 6) A minimal ESP-IDF/TFLite Micro C snippet at the bottom of this file (copy to your ESP32 project)
#
# Notes:
# • Focuses on binary classification: Stress vs Not-Stress for v1 (fast + small)
# • Uses synthetic data fallback; plug in real WESAD/BVP data by replacing `load_data()`
# • Keep ops TFLM-friendly (Conv1D, AvgPool1D, GAP, Dense, ReLU)
# • Window = 8 s, stride = 1 s (decision every 1 s with 8 s context)
# • Model typically < ~100 KB once INT8 quantized (varies with filters/units)

import os
import json
import struct
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
CFG = {
    "fs": 64,                    # Sampling rate (Hz)
    "win_sec": 8,                # Window length (s)
    "stride_sec": 1,             # Stride (s)
    "bp_low": 0.7,               # Bandpass low (Hz)
    "bp_high": 3.5,              # Bandpass high (Hz)
    "bp_order": 3,               # IIR order
    "batch_size": 128,
    "epochs": 40,
    "val_split": 0.2,
    "results_dir": "tinyml_results",
    "num_classes": 2,            # Binary: 0=NotStress, 1=Stress
    "seed": 42
}
np.random.seed(CFG["seed"])

ios = CFG["fs"]
WIN = CFG["win_sec"] * CFG["fs"]
STRIDE = CFG["stride_sec"] * CFG["fs"]

os.makedirs(CFG["results_dir"], exist_ok=True)

# ---------------------------
# Data Loading (replace with real)
# ---------------------------
def load_data():
    """Return raw_bvp (1D array) and raw_labels (per-sample labels 0/1). Replace with real loader.
    For demo: synthesize ~30 minutes with a stress block.
    """
    dur_sec = 30 * 60
    n = dur_sec * CFG["fs"]
    t = np.arange(n) / CFG["fs"]

    # Base heart rate ~ 70 bpm with mild variability
    hr_bpm = 70 + 8*np.sin(2*np.pi*t/60)
    sig = 0.8*np.sin(2*np.pi*(hr_bpm/60)*t)
    sig += 0.15*np.sin(2*np.pi*2*(hr_bpm/60)*t)
    sig += 0.05*np.random.randn(n)

    labels = np.zeros(n, dtype=np.uint8)
    # Stress between [8, 18) minutes
    labels[8*60*CFG["fs"]:18*60*CFG["fs"]] = 1

    return sig.astype(np.float32), labels

# ---------------------------
# Windowing
# ---------------------------
def make_windows(x, y, win=WIN, stride=STRIDE):
    xs, ys = [], []
    for start in range(0, len(x) - win + 1, stride):
        w = x[start:start+win]
        lab = np.argmax(np.bincount(y[start:start+win]))
        xs.append(w)
        ys.append(lab)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)

# ---------------------------
# Causal IIR (sos) + z-norm (stats learned on train only)
# ---------------------------

def design_bandpass(fs, low, high, order):
    nyq = 0.5*fs
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos.astype(np.float32)


def apply_filter(sos, x):
    # Causal filtering (one pass)
    return sosfilt(sos, x).astype(np.float32)

# ---------------------------
# Model (single-input tiny 1D CNN)
# ---------------------------

def build_model(input_len=WIN, num_classes=CFG["num_classes"]):
    inp = layers.Input(shape=(input_len, 1), name="bvp")
    x = layers.Conv1D(8, 9, strides=2, padding='same', activation='relu')(inp)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Conv1D(16, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Conv1D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# TFLite INT8 conversion
# ---------------------------

def rep_ds_gen(x_train):
    # Yield int8 calibration samples in float domain; converter quantizes
    for i in range(min(500, len(x_train))):
        yield [x_train[i:i+1]]


def to_tflite_int8(model, x_train, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: rep_ds_gen(x_train)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(tfl)
    return tfl

# ---------------------------
# Export helpers (C arrays)
# ---------------------------

def bytes_to_c_array(name, b):
    # Produce a C array named `name`
    hex_list = ", ".join(f"0x{bt:02x}" for bt in b)
    return f"const unsigned char {name}[] = {{ {hex_list} }};\nconst unsigned int {name}_len = {len(b)};\n"


def floats_to_c_array(name, arr):
    # 32-bit floats
    body = ", ".join(f"{v:.9g}f" for v in arr.flatten())
    return f"const float {name}[{arr.size}] = {{ {body} }};\n"

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    raw, lab = load_data()

    # Windowing first (so filter has <=8s buffers on-device later)
    X, y = make_windows(raw, lab)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CFG["seed"], stratify=y
    )

    # Design filter once
    sos = design_bandpass(CFG["fs"], CFG["bp_low"], CFG["bp_high"], CFG["bp_order"])

    # Filter
    X_train_f = np.stack([apply_filter(sos, xi) for xi in X_train])
    X_test_f  = np.stack([apply_filter(sos, xi) for xi in X_test])

    # z-norm from TRAIN only
    mean = X_train_f.mean()
    std = X_train_f.std() + 1e-6
    X_train_z = (X_train_f - mean) / std
    X_test_z  = (X_test_f - mean) / std

    # Add channel dim
    X_train_z = X_train_z[..., None]
    X_test_z  = X_test_z[..., None]

    # Build & train
    model = build_model()
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)]
    history = model.fit(
        X_train_z, y_train,
        validation_split=CFG["val_split"],
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        callbacks=cb,
        verbose=2
    )

    # Eval
    test_loss, test_acc = model.evaluate(X_test_z, y_test, verbose=0)
    print(f"Test acc: {test_acc:.4f}")

    # Export paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(CFG["results_dir"], f"bvp_tiny_{ts}")
    os.makedirs(base, exist_ok=True)

    # Save tflite (INT8)
    tfl_path = os.path.join(base, "model_int8.tflite")
    tfl_bytes = to_tflite_int8(model, X_train_z, tfl_path)
    print("Saved:", tfl_path)

    # Export TFLM array header
    h_model = bytes_to_c_array("g_bvp_model", tfl_bytes)
    with open(os.path.join(base, "model_data.h"), "w") as f:
        f.write("#pragma once\n\n")
        f.write(h_model)

    # Export filter + norm constants to header
    with open(os.path.join(base, "preprocess_constants.h"), "w") as f:
        f.write("#pragma once\n\n")
        f.write(f"#define BVP_FS {CFG['fs']}\n")
        f.write(f"#define BVP_WIN {WIN}\n")
        f.write(f"#define BVP_STRIDE {STRIDE}\n")
        f.write(f"static const float BVP_MEAN = {float(mean):.9g}f;\n")
        f.write(f"static const float BVP_STD  = {float(std):.9g}f;\n\n")
        # SOS has shape (sections, 6): [b0, b1, b2, a0, a1, a2] with a0 = 1 in SciPy
        f.write(f"#define BVP_SOS_SECTIONS {sos.shape[0]}\n")
        f.write(floats_to_c_array("BVP_SOS", sos))

    # Also persist JSON metadata
    meta = {
        "cfg": CFG,
        "test_acc": float(test_acc),
        "mean": float(mean),
        "std": float(std),
    }
    with open(os.path.join(base, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Artifacts written to:", base)

# -----------------------------------------------------------------------------
# ESP32-S3 (ESP-IDF + TFLite Micro) – Minimal Inference Snippet (C/C++)
# -----------------------------------------------------------------------------
# Copy the following into your ESP-IDF project (e.g., main/bvp_infer.c). Ensure you
# add the TFLM component (or use ESP-DSP/TFLM port). Include the headers generated
# above: model_data.h and preprocess_constants.h
#
# NOTE: This is a reference snippet; integrate with your sensor acquisition loop.
#
# /* bvp_infer.c (reference) */
# #include <math.h>
# #include <string.h>
# #include "model_data.h"            // g_bvp_model, g_bvp_model_len
# #include "preprocess_constants.h"  // BVP_MEAN, BVP_STD, BVP_SOS, etc.
# #include "tensorflow/lite/micro/all_ops_resolver.h"
# #include "tensorflow/lite/micro/micro_error_reporter.h"
# #include "tensorflow/lite/micro/micro_interpreter.h"
# #include "tensorflow/lite/schema/schema_generated.h"
# #include "tensorflow/lite/version.h"
#
# // Simple SOS IIR (causal) for one block of samples
# static inline float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
#
# typedef struct { float z1, z2; } BiquadState;
#
# // SciPy SOS layout: [b0, b1, b2, a0(=1), a1, a2]
# static void sosfilt_block(const float* sos, int sections, const float* x, float* y, int n) {
#     static BiquadState st[BVP_SOS_SECTIONS];
#     for (int i = 0; i < n; ++i) {
#         float v = x[i];
#         for (int s = 0; s < sections; ++s) {
#             const float b0 = sos[s*6 + 0];
#             const float b1 = sos[s*6 + 1];
#             const float b2 = sos[s*6 + 2];
#             /* a0 = 1 */
#             const float a1 = sos[s*6 + 4];
#             const float a2 = sos[s*6 + 5];
#             float w = v - a1*st[s].z1 - a2*st[s].z2;
#             v = b0*w + b1*st[s].z1 + b2*st[s].z2;
#             st[s].z2 = st[s].z1;
#             st[s].z1 = w;
#         }
#         y[i] = v;
#     }
# }
#
# // Arena size: tune after seeing interpreter->arena_used_bytes()
# constexpr int kArenaSize = 120 * 1024;
# static uint8_t *tensor_arena;  // allocate from heap/PSRAM if needed
#
# extern "C" void bvp_infer_init() {
#   static tflite::MicroErrorReporter micro_error_reporter;
#   const tflite::Model* model = tflite::GetModel(g_bvp_model);
#   static tflite::AllOpsResolver resolver; // Or a reduced resolver for smaller binary
#   tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#   static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize, &micro_error_reporter);
#   interpreter.AllocateTensors();
# }
#
# extern "C" int bvp_infer_run(const float* raw_window, int win_len /*=BVP_WIN*/) {
#   // 1) Filter
#   static float filt[BVP_WIN];
#   sosfilt_block(BVP_SOS, BVP_SOS_SECTIONS, raw_window, filt, win_len);
#   // 2) z-norm
#   for (int i=0;i<win_len;++i) filt[i] = (filt[i] - BVP_MEAN) / BVP_STD;
#   // 3) Quantize to int8 using input scale/zero-point from the model tensor
#   //    (read from interpreter->input(0)->params.scale/zero_point)
#   //    Here we assume scale ~ S and zp ~ Z; replace with actual values.
#   extern float input_scale; extern int input_zero_point; // set during init
#   static int8_t in_q[BVP_WIN];
#   for (int i=0;i<win_len;++i) {
#     const int32_t q = (int32_t)roundf(filt[i]/input_scale) + input_zero_point;
#     in_q[i] = (int8_t)clampf(q, -128, 127);
#   }
#   // 4) Run inference
#   //   Set input shape [1, WIN, 1]
#   //   Read output logits and argmax to class {0,1}
#   //   Return predicted class.
#   return 0; // placeholder
# }
#
# // In your acquisition loop:
# // - Maintain a rolling buffer of BVP samples at 64 Hz
# // - Every 1 s, call bvp_infer_run(buffer_tail_8s, BVP_WIN)
# // - Debounce with a short EMA over the last few predictions to stabilize output
