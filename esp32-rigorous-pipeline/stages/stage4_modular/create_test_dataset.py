#!/usr/bin/env python3
"""
Create a test dataset with only the 30 features our model uses
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_model_data():
    """Load the model data to get feature names"""
    with open("model_data.json", "r") as f:
        return json.load(f)

def create_test_dataset():
    """Create a dataset with just the features we need"""
    
    # Load model info
    model_data = load_model_data()
    feature_names = model_data["features"]
    print(f"📋 Model expects {len(feature_names)} features")
    
    # Load full dataset
    data_path = "../../model-development/data-input/flirt-wesad-acc-bvp-eda-temp-60-10.parquet"
    print(f"📂 Loading full dataset from {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    df = pd.read_parquet(data_path)
    print(f"📏 Full dataset: {df.shape}")
    print(f"📋 Available columns: {len(df.columns)}")
    
    # Check which features we can find
    available_features = []
    missing_features = []
    
    for feature in feature_names:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"\n🔍 Feature Analysis:")
    print(f"✅ Available features: {len(available_features)}/{len(feature_names)}")
    print(f"❌ Missing features: {len(missing_features)}")
    
    if missing_features:
        print(f"Missing: {missing_features[:5]}...")  # Show first 5
    
    if len(available_features) < 20:  # Need at least most features
        print("❌ Too many features missing. Cannot create test dataset.")
        return False
    
    # Create test dataset
    test_features = df[available_features].copy()
    test_labels = df['label'].copy()
    
    # Handle missing features by filling with zeros or mean
    for missing_feature in missing_features:
        print(f"⚠️  Filling missing feature '{missing_feature}' with zeros")
        test_features[missing_feature] = 0.0
    
    # Reorder columns to match model expectations
    test_features = test_features[feature_names]
    
    # Create final dataset
    test_df = test_features.copy()
    test_df['label'] = test_labels
    test_df['subject'] = df['subject'] if 'subject' in df.columns else 0
    
    # Remove rows with infinite or extremely large values
    print(f"📊 Before cleaning: {len(test_df)} samples")
    
    # Replace infinity with NaN
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN
    test_df = test_df.dropna()
    
    # Remove outliers (values > 1e6 or < -1e6)
    numeric_cols = feature_names
    for col in numeric_cols:
        test_df = test_df[(test_df[col] > -1e6) & (test_df[col] < 1e6)]
    
    print(f"📊 After cleaning: {len(test_df)} samples")
    
    if len(test_df) < 100:
        print("❌ Too few samples after cleaning")
        return False
    
    # Save test dataset
    output_path = "test_dataset_30_features.parquet"
    test_df.to_parquet(output_path, index=False)
    
    print(f"✅ Test dataset saved to {output_path}")
    print(f"📏 Final shape: {test_df.shape}")
    print(f"📋 Columns: {list(test_df.columns)}")
    
    # Show sample statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"Label distribution:")
    print(test_df['label'].value_counts())
    
    print(f"\nFeature ranges:")
    for i, feature in enumerate(feature_names[:5]):  # Show first 5
        col_data = test_df[feature]
        print(f"  {feature}: [{col_data.min():.3f}, {col_data.max():.3f}]")
    
    return True

if __name__ == "__main__":
    print("🎯 Creating Test Dataset with 30 Features")
    print("=" * 50)
    
    success = create_test_dataset()
    if success:
        print("\n✅ Test dataset created successfully!")
        print("Now you can run the full metrics validation.")
    else:
        print("\n❌ Failed to create test dataset.")
