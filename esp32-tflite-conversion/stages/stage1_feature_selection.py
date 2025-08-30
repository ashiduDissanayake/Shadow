#!/usr/bin/env python3
"""
STAGE 1: Feature Analysis & Selection for ESP32-S3
Goal: Reduce 73 features → 25-30 features while preserving 90%+ importance

This is a methodical, step-by-step approach.
Build → Test → Validate → Proceed to next stage.
"""

import joblib
import numpy as np
import json
import os
from collections import OrderedDict

class Stage1FeatureSelector:
    """
    Stage 1: Analyze and select optimal feature subset for ESP32-S3
    """
    
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.model = None
        self.model_with_threshold = None
        self.feature_names = None
        self.feature_importance = None
        self.optimal_threshold = None
        self.results = {}
        
    def load_model(self):
        """Load the pre-trained ExtraTreesClassifier"""
        
        print("🔄 STAGE 1: LOADING PRE-TRAINED MODEL")
        print("=" * 50)
        
        model_path = os.path.join(self.data_dir, 'model', 'model_with_threshold.joblib')
        
        try:
            self.model_with_threshold = joblib.load(model_path)
            self.model = self.model_with_threshold['model']
            self.optimal_threshold = self.model_with_threshold['optimal_threshold']
            self.feature_names = self.model_with_threshold['feature_names']
            
            print(f"✅ Model loaded successfully:")
            print(f"   Trees: {self.model.n_estimators}")
            print(f"   Features: {self.model.n_features_in_}")
            print(f"   Threshold: {self.optimal_threshold:.4f}")
            print(f"   Classes: {self.model.classes_}")
            
            # Get feature importance
            self.feature_importance = self.model.feature_importances_
            
            # Store basic info
            self.results['original_model'] = {
                'n_trees': self.model.n_estimators,
                'n_features': self.model.n_features_in_,
                'optimal_threshold': self.optimal_threshold,
                'max_depth': self.model.max_depth,
                'feature_names': self.feature_names
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def analyze_feature_importance(self):
        """Detailed analysis of feature importance by sensor"""
        
        print(f"\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Group features by sensor type
        sensor_groups = {
            'bvp': [],
            'acc': [],
            'eda': [],
            'temp': []
        }
        
        for i, feat_name in enumerate(self.feature_names):
            importance = self.feature_importance[i]
            
            # Determine sensor type
            sensor_type = None
            for sensor in sensor_groups.keys():
                if feat_name.startswith(sensor):
                    sensor_type = sensor
                    break
            
            if sensor_type:
                sensor_groups[sensor_type].append({
                    'index': i,
                    'name': feat_name,
                    'importance': importance
                })
        
        # Sort each group by importance
        for sensor in sensor_groups:
            sensor_groups[sensor].sort(key=lambda x: x['importance'], reverse=True)
        
        # Display analysis
        print(f"📈 Feature distribution by sensor:")
        total_by_sensor = {}
        for sensor, features in sensor_groups.items():
            total_importance = sum(f['importance'] for f in features)
            total_by_sensor[sensor] = total_importance
            print(f"   {sensor.upper()}: {len(features):2d} features, {total_importance:.1%} total importance")
        
        # Show top features overall
        all_features = []
        for i, name in enumerate(self.feature_names):
            all_features.append({
                'index': i,
                'name': name,
                'importance': self.feature_importance[i]
            })
        
        all_features.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
        cumulative_importance = 0
        for i, feat in enumerate(all_features[:15]):
            cumulative_importance += feat['importance']
            print(f"   {i+1:2d}. {feat['name']:<30} {feat['importance']:.4f} "
                  f"({feat['importance']*100:.1f}%) [cum: {cumulative_importance:.1%}]")
        
        # Store results
        self.results['feature_analysis'] = {
            'sensor_groups': sensor_groups,
            'sensor_totals': total_by_sensor,
            'top_features': all_features[:20]
        }
        
        return sensor_groups
    
    def find_optimal_subset(self, target_features=25, min_importance_retention=0.90):
        """
        Find optimal feature subset for ESP32-S3 constraints
        """
        
        print(f"\n🎯 FINDING OPTIMAL FEATURE SUBSET")
        print(f"   Target features: {target_features}")
        print(f"   Min importance retention: {min_importance_retention:.1%}")
        print("=" * 50)
        
        # Sort features by importance
        importance_indices = np.argsort(self.feature_importance)[::-1]
        total_importance = np.sum(self.feature_importance)
        
        # Test different subset sizes
        candidates = []
        
        for n_features in range(15, 36, 2):  # 15, 17, 19, ..., 35
            selected_indices = importance_indices[:n_features]
            selected_importance = np.sum(self.feature_importance[selected_indices])
            retention_ratio = selected_importance / total_importance
            
            # Calculate resource savings
            feature_reduction = (73 - n_features) / 73
            memory_reduction = feature_reduction  # Approximate
            processing_speedup = 1 / (1 - feature_reduction)  # Approximate
            
            # Score: balance importance retention with resource efficiency
            efficiency_bonus = feature_reduction * 0.2  # 20% weight to efficiency
            score = retention_ratio + efficiency_bonus
            
            candidate = {
                'n_features': n_features,
                'selected_indices': selected_indices.tolist(),
                'importance_retention': float(retention_ratio),
                'feature_reduction': float(feature_reduction),
                'memory_reduction': float(memory_reduction),
                'processing_speedup': float(processing_speedup),
                'score': float(score)
            }
            candidates.append(candidate)
            
            print(f"   {n_features:2d} features: {retention_ratio:.1%} importance, "
                  f"{feature_reduction:.1%} reduction, score: {score:.3f}")
        
        # Find best candidate that meets requirements
        valid_candidates = [c for c in candidates if c['importance_retention'] >= min_importance_retention]
        
        if valid_candidates:
            # Among valid candidates, pick best score
            best_candidate = max(valid_candidates, key=lambda x: x['score'])
            print(f"\n✅ OPTIMAL SUBSET FOUND:")
        else:
            # If no valid candidates, pick closest to target
            best_candidate = min(candidates, key=lambda x: abs(x['n_features'] - target_features))
            print(f"\n⚠️  BEST AVAILABLE SUBSET (below {min_importance_retention:.1%} retention):")
        
        print(f"   Features: {best_candidate['n_features']} (was 73)")
        print(f"   Importance retention: {best_candidate['importance_retention']:.1%}")
        print(f"   Feature reduction: {best_candidate['feature_reduction']:.1%}")
        print(f"   Processing speedup: {best_candidate['processing_speedup']:.1f}x")
        print(f"   Score: {best_candidate['score']:.3f}")
        
        self.results['optimal_subset'] = best_candidate
        return best_candidate
    
    def create_feature_mapping(self, optimal_subset):
        """Create detailed mapping of selected features"""
        
        selected_indices = optimal_subset['selected_indices']
        
        print(f"\n📋 SELECTED FEATURE MAPPING:")
        print("=" * 70)
        
        # Create detailed mapping
        feature_mapping = []
        sensor_counts = {'bvp': 0, 'acc': 0, 'eda': 0, 'temp': 0}
        
        for new_idx, original_idx in enumerate(selected_indices):
            feat_name = self.feature_names[original_idx]
            importance = self.feature_importance[original_idx]
            
            # Count by sensor
            for sensor in sensor_counts:
                if feat_name.startswith(sensor):
                    sensor_counts[sensor] += 1
                    break
            
            mapping = {
                'new_index': new_idx,
                'original_index': int(original_idx),
                'name': feat_name,
                'importance': float(importance),
                'importance_percent': float(importance * 100)
            }
            feature_mapping.append(mapping)
            
            print(f"   {new_idx:2d} ← [{original_idx:2d}] {feat_name:<35} "
                  f"{importance:.4f} ({importance*100:.1f}%)")
        
        print(f"\n📊 SENSOR DISTRIBUTION IN SELECTED SUBSET:")
        for sensor, count in sensor_counts.items():
            percentage = count / len(selected_indices) * 100
            print(f"   {sensor.upper()}: {count:2d} features ({percentage:.1f}%)")
        
        # Store mapping
        self.results['feature_mapping'] = {
            'mapping': feature_mapping,
            'sensor_distribution': sensor_counts,
            'total_selected': len(selected_indices)
        }
        
        return feature_mapping
    
    def validate_subset(self):
        """Validate the selected subset meets ESP32-S3 requirements"""
        
        print(f"\n✅ STAGE 1 VALIDATION")
        print("=" * 40)
        
        subset = self.results['optimal_subset']
        mapping = self.results['feature_mapping']
        
        # Define validation criteria
        validations = [
            {
                'name': 'Feature count in range',
                'condition': 15 <= subset['n_features'] <= 35,
                'value': f"{subset['n_features']} features",
                'target': '15-35 features'
            },
            {
                'name': 'Importance retention good',
                'condition': subset['importance_retention'] >= 0.85,
                'value': f"{subset['importance_retention']:.1%}",
                'target': '≥85%'
            },
            {
                'name': 'All sensors represented',
                'condition': min(mapping['sensor_distribution'].values()) > 0,
                'value': f"{len([s for s in mapping['sensor_distribution'].values() if s > 0])}/4 sensors",
                'target': '4/4 sensors'
            },
            {
                'name': 'Significant size reduction',
                'condition': subset['feature_reduction'] >= 0.50,
                'value': f"{subset['feature_reduction']:.1%}",
                'target': '≥50%'
            }
        ]
        
        # Run validations
        all_passed = True
        for validation in validations:
            passed = validation['condition']
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {validation['name']:<25}: {status} ({validation['value']} vs {validation['target']})")
            
            if not passed:
                all_passed = False
        
        # Overall assessment
        if all_passed:
            print(f"\n🎯 STAGE 1: ✅ SUCCESS - Ready for Stage 2")
            self.results['validation_status'] = 'PASSED'
        else:
            print(f"\n🎯 STAGE 1: ⚠️  NEEDS ADJUSTMENT")
            self.results['validation_status'] = 'NEEDS_ADJUSTMENT'
        
        return all_passed
    
    def save_results(self):
        """Save Stage 1 results for next stage"""
        
        # Create comprehensive output
        output = {
            'stage': 1,
            'description': 'Feature selection for ESP32-S3',
            'timestamp': str(np.datetime64('now')),
            'original_model': self.results['original_model'],
            'feature_analysis': self.results['feature_analysis'],
            'optimal_subset': self.results['optimal_subset'],
            'feature_mapping': self.results['feature_mapping'],
            'validation_status': self.results['validation_status']
        }
        
        # Save to outputs directory
        output_file = '../outputs/stage1_feature_selection.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 RESULTS SAVED: {output_file}")
        
        # Also save just the feature indices for easy use
        selected_indices = self.results['optimal_subset']['selected_indices']
        simple_output = {
            'selected_feature_indices': selected_indices,
            'feature_names': [self.feature_names[i] for i in selected_indices],
            'n_features': len(selected_indices),
            'importance_retention': self.results['optimal_subset']['importance_retention']
        }
        
        simple_file = '../outputs/stage1_selected_features.json'
        with open(simple_file, 'w') as f:
            json.dump(simple_output, f, indent=2)
        
        print(f"💾 SIMPLE MAPPING SAVED: {simple_file}")
        
        return output_file

def main():
    """Execute Stage 1: Feature Selection"""
    
    print("🚀 ESP32-S3 TFLITE CONVERSION PIPELINE")
    print("📊 STAGE 1: FEATURE SELECTION")
    print("🎯 Goal: 73 features → 25-30 features")
    print("=" * 60)
    
    # Initialize Stage 1
    stage1 = Stage1FeatureSelector()
    
    # Execute pipeline
    try:
        # Step 1: Load model
        if not stage1.load_model():
            return False
        
        # Step 2: Analyze features
        stage1.analyze_feature_importance()
        
        # Step 3: Find optimal subset
        stage1.find_optimal_subset(target_features=25, min_importance_retention=0.90)
        
        # Step 4: Create mapping
        stage1.create_feature_mapping(stage1.results['optimal_subset'])
        
        # Step 5: Validate
        validation_passed = stage1.validate_subset()
        
        # Step 6: Save results
        output_file = stage1.save_results()
        
        # Summary
        print(f"\n🎯 STAGE 1 COMPLETE!")
        print(f"   Status: {'✅ SUCCESS' if validation_passed else '⚠️  NEEDS REVIEW'}")
        print(f"   Output: {output_file}")
        
        if validation_passed:
            print(f"\n➡️  READY FOR STAGE 2: Tree Optimization")
        else:
            print(f"\n🔧 RECOMMEND: Review and adjust parameters")
        
        return validation_passed
        
    except Exception as e:
        print(f"❌ STAGE 1 FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
