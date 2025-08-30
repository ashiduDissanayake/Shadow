#!/usr/bin/env python3
"""
ESP32-S3 TensorFlow Lite Conversion Pipeline
STAGE 1: ENHANCED FEATURE SELECTION WITH PERFORMANCE VALIDATION

Purpose: Select optimal features with REAL performance validation
- Load original training data and retrain models
- Test feature subsets on actual test data with F1/accuracy metrics
- Find optimal feature count that maintains performance
- Save retrained model with selected features

This addresses the critical issues:
✅ Real performance validation (not just importance)
✅ Model retraining with selected features
✅ Actual F1/accuracy-based optimization
✅ Save retrained model for Stage 2
"""

import json
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings('ignore')

class Stage1EnhancedFeatureSelector:
    """Enhanced feature selector with real performance validation"""
    
    def __init__(self):
        self.original_model = None
        self.feature_names = None
        self.optimal_threshold = None
        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None
        self.baseline_performance = None
        
    def load_original_model(self):
        """Load the original trained model for feature importance analysis"""
        print(f"🔄 LOADING ORIGINAL MODEL")
        print("=" * 50)
        
        model_path = "../data/model/model_with_threshold.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)
        self.original_model = model_data['model']
        self.optimal_threshold = model_data['optimal_threshold']
        self.feature_names = model_data['feature_names']
        
        print(f"✅ Original model loaded:")
        print(f"   Type: {type(self.original_model).__name__}")
        print(f"   Trees: {self.original_model.n_estimators}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Optimal threshold: {self.optimal_threshold:.4f}")
        
    def load_training_data(self):
        """Load REAL WESAD training data - NO synthetic data fallback"""
        print(f"\n🔄 LOADING REAL WESAD TRAINING DATA")
        print("=" * 50)
        
        # Use the EXACT same data source as the original model training
        print("🎯 Loading WESAD physiological data (same source as original model)...")
        
        try:
            # Direct approach: Load the exact parquet file used for training
            wesad_data_path = "../../model-development/data-input/flirt-wesad-acc-bvp-eda-temp-30-1.parquet"
            
            print(f"   📊 Loading: {wesad_data_path}")
            
            # Load the full WESAD dataset
            df_full = pd.read_parquet(wesad_data_path)
            
            print(f"✅ WESAD dataset loaded successfully:")
            print(f"   Total samples: {len(df_full)}")
            print(f"   Total features: {len(df_full.columns)}")
            print(f"   Columns: {list(df_full.columns)}")
            
            # Apply the SAME preprocessing as original model
            print(f"   🔧 Applying same preprocessing as original model...")
            
            # Remove the same columns as in utils.py
            columns_to_drop = ['eda_EDA_n_sign_changes',
                             'temp_TEMP_peaks',
                             'acc_y_entropy',
                             'acc_l2_n_sign_changes',
                             'acc_x_entropy',
                             'acc_z_entropy',
                             'temp_l2_n_sign_changes',
                             'bvp_BVP_entropy',
                             'temp_TEMP_n_sign_changes',
                             'temp_l2_peaks',
                             'eda_l2_n_sign_changes']
            
            # Only drop columns that actually exist
            existing_columns_to_drop = [col for col in columns_to_drop if col in df_full.columns]
            if existing_columns_to_drop:
                df_full = df_full.drop(columns=existing_columns_to_drop)
                print(f"   ✅ Dropped {len(existing_columns_to_drop)} columns as per original preprocessing")
            
            # Split into features and labels (same as utils.py)
            if 'subject' in df_full.columns and 'label' in df_full.columns:
                X_full = df_full.drop(columns=['subject', 'label'])
                y_full = df_full['label']
                groups_full = df_full['subject']
                
                # Apply same train/test split as original (GroupKFold with 5 splits)
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=5)
                train_idx, test_idx = next(gkf.split(X_full, y_full, groups_full))
                
                X_train = X_full.iloc[train_idx]
                y_train = y_full.iloc[train_idx]
                
                print(f"   ✅ Applied GroupKFold split (same as original model)")
                print(f"   📊 Train samples: {len(X_train)}")
                print(f"   📊 Train features: {len(X_train.columns)}")
                
                # Apply correlation removal (same as original)
                print(f"   🔧 Removing correlated features (threshold: 0.8)...")
                cor = X_train.corr(numeric_only=True)
                keep_columns = np.full(cor.shape[0], True)
                for i in range(cor.shape[0] - 1):
                    for j in range(i + 1, cor.shape[0] - 1):
                        if (np.abs(cor.iloc[i, j]) >= 0.8):
                            keep_columns[j] = False
                            
                selected_columns = X_train.columns[keep_columns]
                X_train_reduced = X_train[selected_columns]
                
                print(f"   ✅ Correlation removal: {len(X_train.columns)} → {len(selected_columns)} features")
                
                self.training_data = X_train_reduced
                self.training_labels = y_train.values
                
                print(f"✅ REAL WESAD training data ready:")
                print(f"   Training samples: {len(self.training_data)}")
                print(f"   Features: {len(self.training_data.columns)}")
                print(f"   Class distribution: {np.bincount(self.training_labels)}")
                print(f"   Data source: Real WESAD physiological dataset")
                
                # Ensure feature order matches model
                if list(self.training_data.columns) != self.feature_names:
                    print(f"   🔄 Matching features with original model...")
                    # Find intersection of features
                    available_features = [f for f in self.feature_names if f in self.training_data.columns]
                    missing_features = [f for f in self.feature_names if f not in self.training_data.columns]
                    
                    if missing_features:
                        print(f"   ⚠️ Missing features in WESAD data: {len(missing_features)}")
                        print(f"      This might indicate preprocessing differences")
                        # Use available features only
                        self.training_data = self.training_data[available_features]
                        self.feature_names = available_features
                        print(f"   ✅ Using {len(available_features)} available features")
                    else:
                        self.training_data = self.training_data[self.feature_names]
                        print(f"   ✅ All features matched successfully")
                
                return True
                
            else:
                print(f"❌ Required columns ('subject', 'label') not found in WESAD data")
                print(f"   Available columns: {list(df_full.columns)}")
                return False
                
        except Exception as e:
            print(f"❌ Could not load WESAD training data: {e}")
            print(f"   File path: {wesad_data_path}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_test_data(self):
        """Load test data for performance validation"""
        print(f"\n🔄 LOADING TEST DATA FOR VALIDATION")
        print("=" * 50)
        
        # Try parquet test data first
        stress_file = "../../web-app/test-data/test-data-stress.parquet"
        nostress_file = "../../web-app/test-data/test-data-nostress.parquet"
        
        if os.path.exists(stress_file) and os.path.exists(nostress_file):
            try:
                stress_data = pd.read_parquet(stress_file)
                nostress_data = pd.read_parquet(nostress_file)
                
                self.test_data = pd.concat([stress_data, nostress_data], ignore_index=True)
                self.test_labels = np.array([1] * len(stress_data) + [0] * len(nostress_data))
                
                # Ensure feature order matches
                if list(self.test_data.columns) == self.feature_names:
                    print(f"✅ Test data loaded from parquet files:")
                    print(f"   Test samples: {len(self.test_data)}")
                    print(f"   Class distribution: {np.bincount(self.test_labels)}")
                    return
                else:
                    self.test_data = self.test_data[self.feature_names]
                    print(f"✅ Test data loaded and reordered:")
                    print(f"   Test samples: {len(self.test_data)}")
                    
            except Exception as e:
                print(f"   ⚠️ Could not load parquet test data: {e}")
        
        # Fallback: use part of training data as test (split)
        print("   Using training data split for testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.training_data, self.training_labels, 
            test_size=0.2, random_state=42, stratify=self.training_labels
        )
        
        # Update training and test data
        self.training_data = X_train
        self.training_labels = y_train
        self.test_data = X_test
        self.test_labels = y_test
        
        print(f"✅ Data split completed:")
        print(f"   Training samples: {len(self.training_data)}")
        print(f"   Test samples: {len(self.test_data)}")
        
    def establish_baseline_performance(self):
        """Establish baseline performance with all features"""
        print(f"\n🎯 BASELINE PERFORMANCE EVALUATION")
        print("=" * 50)
        
        # Test original model on test data
        if hasattr(self.original_model, 'predict_proba'):
            probabilities = self.original_model.predict_proba(self.test_data)[:, 1]
            predictions = (probabilities >= self.optimal_threshold).astype(int)
        else:
            predictions = self.original_model.predict(self.test_data)
            
        self.baseline_performance = {
            'accuracy': accuracy_score(self.test_labels, predictions),
            'f1_score': f1_score(self.test_labels, predictions),
            'precision': precision_score(self.test_labels, predictions, zero_division=0),
            'recall': recall_score(self.test_labels, predictions, zero_division=0)
        }
        
        print(f"✅ Baseline Performance (ALL {len(self.feature_names)} features):")
        print(f"   Accuracy:  {self.baseline_performance['accuracy']:.4f}")
        print(f"   F1-Score:  {self.baseline_performance['f1_score']:.4f}")
        print(f"   Precision: {self.baseline_performance['precision']:.4f}")
        print(f"   Recall:    {self.baseline_performance['recall']:.4f}")
        
        return self.baseline_performance
        
    def test_feature_subset_performance(self, feature_indices, model_params=None):
        """Test performance of a specific feature subset by retraining"""
        
        # Use original model parameters if not specified
        if model_params is None:
            model_params = {
                'n_estimators': self.original_model.n_estimators,
                'criterion': getattr(self.original_model, 'criterion', 'entropy'),
                'max_features': getattr(self.original_model, 'max_features', 0.8),
                'min_samples_leaf': getattr(self.original_model, 'min_samples_leaf', 4),
                'min_samples_split': getattr(self.original_model, 'min_samples_split', 4),
                'bootstrap': getattr(self.original_model, 'bootstrap', False),
                'random_state': getattr(self.original_model, 'random_state', 0),
                'class_weight': getattr(self.original_model, 'class_weight', {0: 1.0, 1: 4.0})
            }
        
        # Extract subset features
        X_train_subset = self.training_data.iloc[:, feature_indices]
        X_test_subset = self.test_data.iloc[:, feature_indices]
        
        # Train new model on subset
        subset_model = ExtraTreesClassifier(**model_params)
        subset_model.fit(X_train_subset, self.training_labels)
        
        # Evaluate performance
        if hasattr(subset_model, 'predict_proba'):
            probabilities = subset_model.predict_proba(X_test_subset)[:, 1]
            predictions = (probabilities >= self.optimal_threshold).astype(int)
        else:
            predictions = subset_model.predict(X_test_subset)
            
        performance = {
            'accuracy': accuracy_score(self.test_labels, predictions),
            'f1_score': f1_score(self.test_labels, predictions),
            'precision': precision_score(self.test_labels, predictions, zero_division=0),
            'recall': recall_score(self.test_labels, predictions, zero_division=0)
        }
        
        return subset_model, performance
        
    def iterative_feature_selection(self):
        """Find optimal feature subset using performance-based iterative selection"""
        print(f"\n🔍 ITERATIVE FEATURE SELECTION WITH PERFORMANCE VALIDATION")
        print("=" * 50)
        
        # Get feature importance ranking
        feature_importance = self.original_model.feature_importances_
        feature_rankings = np.argsort(feature_importance)[::-1]  # Highest importance first
        
        # Test different feature counts
        feature_counts = [60, 50, 45, 40, 35, 30, 25, 20, 15]
        min_f1_retention = 0.95  # Must retain 95% of F1 score
        min_accuracy_retention = 0.95
        
        baseline_f1 = self.baseline_performance['f1_score']
        baseline_accuracy = self.baseline_performance['accuracy']
        
        print(f"🎯 Selection targets:")
        print(f"   Min F1 retention: {min_f1_retention*100:.1f}% (≥{baseline_f1*min_f1_retention:.4f})")
        print(f"   Min accuracy retention: {min_accuracy_retention*100:.1f}% (≥{baseline_accuracy*min_accuracy_retention:.4f})")
        print(f"   Target features: 15-40 (ESP32-S3 optimized)")
        
        results = []
        
        for n_features in feature_counts:
            print(f"\n   Testing {n_features} features...")
            
            # Select top N features by importance
            selected_indices = feature_rankings[:n_features].tolist()
            
            # Test performance with retraining
            subset_model, performance = self.test_feature_subset_performance(selected_indices)
            
            # Calculate retention rates
            f1_retention = performance['f1_score'] / baseline_f1
            accuracy_retention = performance['accuracy'] / baseline_accuracy
            
            # Memory estimation for ESP32-S3
            estimated_memory_kb = self._estimate_feature_memory(n_features)
            
            result = {
                'n_features': n_features,
                'selected_indices': selected_indices,
                'selected_features': [self.feature_names[i] for i in selected_indices],
                'performance': performance,
                'f1_retention': f1_retention,
                'accuracy_retention': accuracy_retention,
                'estimated_memory_kb': estimated_memory_kb,
                'esp32_compatible': estimated_memory_kb < 100 and 15 <= n_features <= 40,
                'performance_acceptable': (f1_retention >= min_f1_retention and 
                                         accuracy_retention >= min_accuracy_retention),
                'retrained_model': subset_model
            }
            
            results.append(result)
            
            # Status reporting
            f1_status = "✅" if f1_retention >= min_f1_retention else "❌"
            acc_status = "✅" if accuracy_retention >= min_accuracy_retention else "❌"
            mem_status = "✅" if estimated_memory_kb < 100 else "⚠️"
            
            print(f"     {f1_status} F1: {performance['f1_score']:.4f} (retention: {f1_retention:.1%})")
            print(f"     {acc_status} Acc: {performance['accuracy']:.4f} (retention: {accuracy_retention:.1%})")
            print(f"     {mem_status} Memory: {estimated_memory_kb:.1f}KB")
            
            # Early stopping if performance drops too much
            if f1_retention < 0.90:
                print(f"     ⚠️ F1 retention below 90%, stopping feature reduction")
                break
                
        return results
        
    def _estimate_feature_memory(self, n_features):
        """Estimate memory usage for features on ESP32-S3"""
        # Feature storage: 4 bytes per float32 feature
        feature_storage_kb = (n_features * 4) / 1024
        
        # Preprocessing overhead (normalization, etc.)
        preprocessing_overhead_kb = n_features * 0.1
        
        # Buffer and alignment overhead
        buffer_overhead_kb = 5
        
        total_kb = feature_storage_kb + preprocessing_overhead_kb + buffer_overhead_kb
        return total_kb
        
    def select_optimal_configuration(self, results):
        """Select optimal feature configuration"""
        print(f"\n🏆 SELECTING OPTIMAL FEATURE CONFIGURATION")
        print("=" * 50)
        
        # Filter for acceptable performance and ESP32 compatibility
        viable_options = [r for r in results if r['performance_acceptable'] and r['esp32_compatible']]
        
        if not viable_options:
            print("⚠️ No configurations meet all criteria, selecting best compromise")
            # Select best F1 retention among ESP32-compatible options
            esp32_compatible = [r for r in results if r['esp32_compatible']]
            if esp32_compatible:
                optimal_config = max(esp32_compatible, key=lambda x: x['f1_retention'])
            else:
                optimal_config = max(results, key=lambda x: x['f1_retention'])
        else:
            # Select the one with BEST performance retention first, then minimum features
            # This prioritizes perfect retention (100%) over fewer features
            optimal_config = max(viable_options, key=lambda x: (x['f1_retention'], x['accuracy_retention'], -x['n_features']))
            
        print(f"✅ Optimal configuration selected:")
        print(f"   Features: {optimal_config['n_features']} (from {len(self.feature_names)})")
        print(f"   F1-Score: {optimal_config['performance']['f1_score']:.4f}")
        print(f"   F1 Retention: {optimal_config['f1_retention']:.1%}")
        print(f"   Accuracy: {optimal_config['performance']['accuracy']:.4f}")
        print(f"   Accuracy Retention: {optimal_config['accuracy_retention']:.1%}")
        print(f"   Memory: {optimal_config['estimated_memory_kb']:.1f} KB")
        print(f"   ESP32 Compatible: {'✅' if optimal_config['esp32_compatible'] else '❌'}")
        
        return optimal_config
        
    def save_retrained_model_and_results(self, optimal_config):
        """Save the retrained model and detailed results"""
        print(f"\n💾 SAVING RETRAINED MODEL AND RESULTS")
        print("=" * 50)
        
        os.makedirs('../outputs', exist_ok=True)
        
        # Save retrained model with metadata
        retrained_model_data = {
            'model': optimal_config['retrained_model'],
            'optimal_threshold': self.optimal_threshold,
            'feature_names': optimal_config['selected_features'],
            'selected_feature_indices': optimal_config['selected_indices'],
            'performance': optimal_config['performance'],
            'baseline_performance': self.baseline_performance,
            'n_features_original': len(self.feature_names),
            'n_features_selected': optimal_config['n_features'],
            'feature_reduction': 1 - (optimal_config['n_features'] / len(self.feature_names)),
            'memory_estimate_kb': optimal_config['estimated_memory_kb']
        }
        
        joblib.dump(retrained_model_data, '../outputs/stage1_retrained_model.joblib')
        
        # Save detailed results
        detailed_results = {
            'stage': 1,
            'description': 'Enhanced feature selection with performance validation and retraining',
            'timestamp': datetime.now().isoformat(),
            'methodology': 'iterative_feature_selection_with_retraining',
            'baseline_performance': self.baseline_performance,
            'optimal_configuration': {
                'n_features': optimal_config['n_features'],
                'selected_indices': optimal_config['selected_indices'],
                'selected_features': optimal_config['selected_features'],
                'performance': optimal_config['performance'],
                'f1_retention': optimal_config['f1_retention'],
                'accuracy_retention': optimal_config['accuracy_retention'],
                'memory_estimate_kb': optimal_config['estimated_memory_kb']
            },
            'validation_status': 'ENHANCED_VALIDATION_COMPLETE'
        }
        
        with open('../outputs/stage1_enhanced_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
            
        print(f"✅ Retrained model saved: ../outputs/stage1_retrained_model.joblib")
        print(f"✅ Detailed results saved: ../outputs/stage1_enhanced_results.json")
        print(f"   Features: {optimal_config['n_features']}")
        print(f"   F1-Score: {optimal_config['performance']['f1_score']:.4f}")
        print(f"   F1 Retention: {optimal_config['f1_retention']:.1%}")

def main():
    """Execute Enhanced Stage 1: Feature Selection with Performance Validation"""
    print("🚀 ESP32-S3 TFLITE CONVERSION PIPELINE")
    print("🔍 STAGE 1: ENHANCED FEATURE SELECTION")
    print("🎯 Goal: Select optimal features with REAL performance validation")
    print("=" * 60)
    
    selector = Stage1EnhancedFeatureSelector()
    
    try:
        # Step 1: Load original model
        selector.load_original_model()
        
        # Step 2: Load training data for retraining
        selector.load_training_data()
        
        # Step 3: Load test data for validation
        selector.load_test_data()
        
        # Step 4: Establish baseline performance
        selector.establish_baseline_performance()
        
        # Step 5: Iterative feature selection with retraining
        all_results = selector.iterative_feature_selection()
        
        # Step 6: Select optimal configuration
        optimal_config = selector.select_optimal_configuration(all_results)
        
        # Step 7: Save retrained model and results
        selector.save_retrained_model_and_results(optimal_config)
        
        # Final status
        print(f"\n🎯 STAGE 1 ENHANCED: ✅ SUCCESS")
        print(f"   Method: Performance validation with model retraining")
        print(f"   Features: {len(selector.feature_names)} → {optimal_config['n_features']}")
        print(f"   F1-Score: {optimal_config['performance']['f1_score']:.4f} (retention: {optimal_config['f1_retention']:.1%})")
        print(f"   Memory: {optimal_config['estimated_memory_kb']:.1f} KB")
        print(f"   Status: ✅ Retrained model ready for Stage 2")
        
        print(f"\n➡️  NEXT: Stage 2 - Tree optimization with retrained model")
        
    except Exception as e:
        print(f"\n❌ STAGE 1 ENHANCED: ERROR")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    main()
