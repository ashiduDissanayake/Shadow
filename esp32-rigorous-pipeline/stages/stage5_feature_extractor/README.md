# Stage 5: Feature Extractor

Bridge from 60-second sensor windows to 30 features for the stress detection model.

## 📁 Structure

```
stage5_feature_extractor/
├── feature_extractor.py      # Main feature extraction implementation
├── integration_test.py       # Test with real C model
└── README.md                 # This file
```

## 🎯 Purpose

Converts raw sensor data from 60-second windows into the exact 30 features expected by the C stress detection model.

### Input:
- **BVP**: 3,840 samples (64 Hz × 60s)
- **ACC (x,y,z)**: 1,920 samples each (32 Hz × 60s)
- **EDA**: 240 samples (4 Hz × 60s)
- **TEMP**: 240 samples (4 Hz × 60s)

### Output:
- **30 features** in exact order matching model_data.json

## 🧪 Testing

```bash
python feature_extractor.py
```

## ✅ Current Status

- ✅ Feature extraction implemented
- ✅ All 30 features extracted correctly
- ✅ Synthetic data testing passed
- 🔄 Integration with C model (next step)

## 📊 Extracted Features

1. **BVP Features (4)**:
   - bvp_BVP_perm_entropy
   - bvp_BVP_n_sign_changes  
   - bvp_l2_iqr
   - bvp_l2_peaks

2. **Accelerometer Features (15)**:
   - acc_y_perm_entropy, acc_y_lineintegral, acc_y_peaks
   - acc_z_perm_entropy, acc_z_peaks, acc_z_rms, acc_z_min, acc_z_energy, acc_z_pct_95, acc_z_mean
   - acc_l2_ptp, acc_l2_max, acc_l2_peaks, acc_l2_rms, acc_l2_min

3. **EDA Features (6)**:
   - eda_EDA_lineintegral, eda_EDA_iqr_5_95, eda_EDA_max
   - eda_l2_lineintegral, eda_l2_iqr_5_95, eda_l2_min

4. **Temperature Features (5)**:
   - temp_TEMP_min, temp_TEMP_energy, temp_TEMP_sum
   - temp_l2_min, temp_l2_energy

## 🔗 Integration

Next: Test with Stage 4 C model to ensure full pipeline works correctly.
