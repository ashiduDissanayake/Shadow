#!/usr/bin/env python3
"""
STAGE 1: Feature Analysis & Selection for ESP32-S3 Deployment
Incrementally reduce 73 features → 25-30 features while preserving accuracy

This is the FIRST step in our hybrid conversion approach.
We will build, test, and validate before moving to Stage 2.
"""

import joblib
import numpy as np
import json
from collections import OrderedDict

class FeatureAnalyzer:
    """
    Analyze feature importance and select optimal subset for ESP32-S3
    """
    
    def __init__(self):
        self.model = None
        self.model_with_threshold = None
        self.feature_names = None
        self.feature_importance = None
        self.optimal_threshold = None
        
    def load_model(self):
        """Load the pre-trained ExtraTreesClassifier"""
        
        print("🔄 STAGE 1: LOADING PRE-TRAINED MODEL")
        print("=" * 50)
        
        # Load model with threshold info
        self.model_with_threshold = joblib.load('model/model_with_threshold.joblib')
        self.model = self.model_with_threshold['model']
        self.optimal_threshold = self.model_with_threshold['optimal_threshold']
        self.feature_names = self.model_with_threshold['feature_names']
        
        print(f"✅ Model loaded:")
        print(f"   Trees: {self.model.n_estimators}")
        print(f"   Features: {self.model.n_features_in_}")
        print(f"   Threshold: {self.optimal_threshold:.4f}")
        print(f"   Feature names: {len(self.feature_names)}")
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        return True
    
    def analyze_feature_importance(self):
        """Detailed analysis of feature importance"""
        
        print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        # Sort features by importance
        importance_indices = np.argsort(self.feature_importance)[::-1]
        
        # Group features by sensor type
        sensor_groups = {
            'bvp': [],
            'acc': [],
            'eda': [],
            'temp': []
        }
        
        for i, feat_name in enumerate(self.feature_names):
            importance = self.feature_importance[i]
            for sensor in sensor_groups.keys():
                if feat_name.startswith(sensor):
                    sensor_groups[sensor].append((i, feat_name, importance))
                    break
        
        # Sort each group by importance
        for sensor in sensor_groups:
            sensor_groups[sensor].sort(key=lambda x: x[2], reverse=True)
        
        print(f"📈 Feature distribution by sensor:")
        for sensor, features in sensor_groups.items():
            total_importance = sum(f[2] for f in features)
            print(f"   {sensor.upper()}: {len(features)} features, {total_importance:.1%} total importance")
        
        # Show top features overall
        print(f"\n🏆 TOP 20 MOST IMPORTANT FEATURES:")
        cumulative_importance = 0
        for i, feat_idx in enumerate(importance_indices[:20]):
            importance = self.feature_importance[feat_idx]
            cumulative_importance += importance
            feat_name = self.feature_names[feat_idx]
            print(f"   {i+1:2d}. {feat_name:<25} {importance:.4f} ({importance*100:.1f}%) [cum: {cumulative_importance:.1%}]")
        
        return sensor_groups
    
    def find_optimal_feature_subset(self, target_features=25, min_importance_retention=0.90):
        """
        Find optimal subset of features that retains most importance
        
        Args:
            target_features: Target number of features (25-30 for ESP32-S3)
            min_importance_retention: Minimum % of total importance to retain
        """
        
        print(f"\n🎯 FINDING OPTIMAL FEATURE SUBSET")
        print(f"   Target features: {target_features}")
        print(f"   Min importance retention: {min_importance_retention:.1%}")
        print("=" * 50)
        
        # Sort features by importance
        importance_indices = np.argsort(self.feature_importance)[::-1]
        total_importance = np.sum(self.feature_importance)
        
        # Try different subset sizes
        results = []
        
        for n_features in range(15, 35, 2):  # Test 15, 17, 19, ..., 33 features
            selected_indices = importance_indices[:n_features]
            selected_importance = np.sum(self.feature_importance[selected_indices])
            retention_ratio = selected_importance / total_importance
            
            # Calculate memory reduction
            memory_reduction = (73 - n_features) / 73
            
            # Calculate processing reduction (assuming linear relationship)
            processing_reduction = memory_reduction
            
            result = {
                'n_features': n_features,
                'selected_indices': selected_indices,
                'importance_retention': retention_ratio,
                'memory_reduction': memory_reduction,
                'processing_reduction': processing_reduction,
                'score': retention_ratio * (1 + memory_reduction * 0.3)  # Weight memory reduction slightly
            }
            results.append(result)
            
            print(f"   {n_features:2d} features: {retention_ratio:.1%} importance, "
                  f"{memory_reduction:.1%} memory saved, score: {result['score']:.3f}")
        
        # Find best option that meets criteria
        valid_options = [r for r in results if r['importance_retention'] >= min_importance_retention]
        
        if not valid_options:
            print(f"⚠️  No options meet {min_importance_retention:.1%} importance retention")
            # Use the option closest to target
            best_option = min(results, key=lambda x: abs(x['n_features'] - target_features))
        else:
            # Among valid options, pick the one closest to target features
            best_option = min(valid_options, key=lambda x: abs(x['n_features'] - target_features))
        
        print(f"\n✅ RECOMMENDED FEATURE SUBSET:")
        print(f"   Features: {best_option['n_features']} (from 73)")
        print(f"   Importance retention: {best_option['importance_retention']:.1%}")
        print(f"   Memory reduction: {best_option['memory_reduction']:.1%}")
        print(f"   Processing speed gain: {best_option['processing_reduction']:.1%}")
        
        return best_option
    
    def create_feature_subset(self, selected_option):
        """Create the actual feature subset with details"""
        
        selected_indices = selected_option['selected_indices']
        
        print(f"\n📋 SELECTED FEATURE DETAILS:")
        print("=" * 60)
        
        selected_features = []
        sensor_counts = {'bvp': 0, 'acc': 0, 'eda': 0, 'temp': 0}
        
        for i, feat_idx in enumerate(selected_indices):
            feat_name = self.feature_names[feat_idx]
            importance = self.feature_importance[feat_idx]
            
            # Count by sensor
            for sensor in sensor_counts:
                if feat_name.startswith(sensor):
                    sensor_counts[sensor] += 1
                    break
            
            selected_features.append({
                'index': feat_idx,
                'name': feat_name,
                'importance': importance,
                'rank': i + 1
            })
            
            print(f"   {i+1:2d}. [{feat_idx:2d}] {feat_name:<25} {importance:.4f} ({importance*100:.1f}%)")
        
        print(f"\n📊 SENSOR DISTRIBUTION IN SUBSET:")
        for sensor, count in sensor_counts.items():
            print(f"   {sensor.upper()}: {count} features")
        
        # Save subset configuration
        subset_config = {
            'original_features': 73,
            'selected_features': selected_option['n_features'],
            'importance_retention': selected_option['importance_retention'],
            'memory_reduction': selected_option['memory_reduction'],
            'selected_indices': selected_indices.tolist(),
            'feature_mapping': selected_features,
            'sensor_distribution': sensor_counts,
            'optimal_threshold': self.optimal_threshold
        }
        
        # Save to JSON for next stage
        with open('stage1_feature_subset.json', 'w') as f:
            json.dump(subset_config, f, indent=2)
        
        print(f"\n💾 Saved configuration to: stage1_feature_subset.json")
        
        return subset_config
    
    def validate_subset(self, subset_config):
        """Basic validation of the feature subset"""
        
        print(f"\n✅ STAGE 1 VALIDATION:")
        print("=" * 30)
        
        n_selected = subset_config['selected_features']
        retention = subset_config['importance_retention']
        reduction = subset_config['memory_reduction']
        
        # Validation checks
        checks = [
            ("Features in ESP32-S3 range", 15 <= n_selected <= 35, f"{n_selected} features"),
            ("Importance retention good", retention >= 0.85, f"{retention:.1%}"),
            ("Memory reduction significant", reduction >= 0.50, f"{reduction:.1%}"),
            ("All sensors represented", min(subset_config['sensor_distribution'].values()) > 0, "All 4 sensors"),
        ]
        
        all_passed = True
        for check_name, passed, detail in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check_name}: {status} ({detail})")
            if not passed:
                all_passed = False
        
        print(f"\nSTAGE 1 RESULT: {'✅ READY FOR STAGE 2' if all_passed else '⚠️ NEEDS ADJUSTMENT'}")
        
        return all_passed

def main():
    """Stage 1: Feature analysis and selection"""
    
    print("🚀 ESP32-S3 TFLITE CONVERSION - STAGE 1")
    print("🎯 Goal: Reduce 73 features → 25-30 features")
    print("📊 Method: Importance-based selection")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer()
    
    # Step 1: Load model
    if not analyzer.load_model():
        print("❌ Failed to load model")
        return
    
    # Step 2: Analyze feature importance
    sensor_groups = analyzer.analyze_feature_importance()
    
    # Step 3: Find optimal subset
    optimal_subset = analyzer.find_optimal_feature_subset(target_features=25)
    
    # Step 4: Create detailed subset
    subset_config = analyzer.create_feature_subset(optimal_subset)
    
    # Step 5: Validate
    validation_passed = analyzer.validate_subset(subset_config)
    
    print(f"\n🎯 STAGE 1 COMPLETE!")
    print(f"Next: Stage 2 - Tree optimization and selection")
    
    if validation_passed:
        print(f"✅ Ready to proceed to Stage 2")
    else:
        print(f"⚠️ Consider adjusting parameters before Stage 2")

if __name__ == "__main__":
    main()
