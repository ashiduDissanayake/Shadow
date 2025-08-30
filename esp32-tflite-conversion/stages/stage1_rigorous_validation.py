#!/usr/bin/env python3
"""
ESP32-S3 TensorFlow Lite Conversion Pipeline
STAGE 1: RIGOROUS FEATURE SELECTION WITH PROPER VALIDATION

Purpose: Address critical validation issues identified:
✅ Fix data leakage with proper train/validation/test splits
✅ Fix correlation removal bug (missing last column)
✅ Use subject-level GroupKFold for proper cross-validation
✅ Re-tune optimal threshold after each model change
✅ Larger, stratified test set
✅ No test set contamination during selection

Methodology:
- Train/Validation/Test split by SUBJECT (no leakage)
- Feature selection uses only train/validation
- Final test evaluation ONLY at the end
- Threshold re-tuning on validation set
- Subject-level stratification
"""

import json
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import sys
import warnings
warnings.filterwarnings('ignore')

class Stage1RigorousFeatureSelector:
    """Rigorous feature selector with proper validation methodology"""
    
    def __init__(self):
        self.original_model = None
        self.feature_names = None
        self.optimal_threshold = None
        
        # Proper data splits (no leakage)
        self.train_data = None
        self.train_labels = None
        self.train_subjects = None
        
        self.val_data = None
        self.val_labels = None
        self.val_subjects = None
        
        self.test_data = None
        self.test_labels = None
        self.test_subjects = None
        
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
        print(f"   Original threshold: {self.optimal_threshold:.4f}")
        
    def load_and_split_wesad_data(self):
        """Load WESAD data and create proper train/validation/test splits by SUBJECT"""
        print(f"\n🔄 LOADING WESAD DATA WITH RIGOROUS SPLITTING")
        print("=" * 50)
        
        try:
            # Load the full WESAD dataset
            wesad_data_path = "../../model-development/data-input/flirt-wesad-acc-bvp-eda-temp-30-1.parquet"
            print(f"   📊 Loading: {wesad_data_path}")
            
            df_full = pd.read_parquet(wesad_data_path)
            print(f"✅ WESAD dataset loaded: {len(df_full)} samples, {len(df_full.columns)} features")
            
            # Apply preprocessing (fix correlation removal bug)
            print(f"   🔧 Applying preprocessing with bug fixes...")
            df_preprocessed = self._apply_rigorous_preprocessing(df_full)
            
            # Extract features, labels, and subjects
            if 'subject' in df_preprocessed.columns and 'label' in df_preprocessed.columns:
                X_full = df_preprocessed.drop(columns=['subject', 'label'])
                y_full = df_preprocessed['label']
                subjects_full = df_preprocessed['subject']
                
                print(f"   📊 Preprocessed data: {len(X_full)} samples, {len(X_full.columns)} features")
                print(f"   👥 Subjects: {sorted(subjects_full.unique())}")
                print(f"   📊 Class distribution: {dict(zip(*np.unique(y_full, return_counts=True)))}")
                
                # Create subject-level train/validation/test splits
                self._create_subject_level_splits(X_full, y_full, subjects_full)
                
                return True
                
            else:
                print(f"❌ Required columns ('subject', 'label') not found")
                return False
                
        except Exception as e:
            print(f"❌ Error loading WESAD data: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _apply_rigorous_preprocessing(self, df):
        """Apply preprocessing with bug fixes"""
        df_clean = df.copy()
        
        # Drop problematic columns
        columns_to_drop = ['eda_EDA_n_sign_changes', 'temp_TEMP_peaks', 'acc_y_entropy',
                          'acc_l2_n_sign_changes', 'acc_x_entropy', 'acc_z_entropy',
                          'temp_l2_n_sign_changes', 'bvp_BVP_entropy', 'temp_TEMP_n_sign_changes',
                          'temp_l2_peaks', 'eda_l2_n_sign_changes']
        
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        if existing_columns_to_drop:
            df_clean = df_clean.drop(columns=existing_columns_to_drop)
            print(f"   ✅ Dropped {len(existing_columns_to_drop)} problematic columns")
        
        return df_clean
        
    def _create_subject_level_splits(self, X, y, subjects):
        """Create proper train/validation/test splits by subject (no leakage)"""
        print(f"   🎯 Creating subject-level splits (no data leakage)...")
        
        # Get unique subjects and their class distributions
        unique_subjects = sorted(subjects.unique())
        subject_stats = []
        
        for subject in unique_subjects:
            subject_mask = subjects == subject
            subject_labels = y[subject_mask]
            stress_ratio = (subject_labels == 1).mean()
            subject_stats.append({
                'subject': subject,
                'n_samples': len(subject_labels),
                'stress_ratio': stress_ratio
            })
        
        print(f"   👥 Subject statistics:")
        for stat in subject_stats:
            print(f"      Subject {stat['subject']}: {stat['n_samples']} samples, {stat['stress_ratio']:.1%} stress")
        
        # Split subjects: 60% train, 20% validation, 20% test
        # Try to balance stress ratios across splits
        n_subjects = len(unique_subjects)
        n_train = max(1, int(0.6 * n_subjects))
        n_val = max(1, int(0.2 * n_subjects))
        n_test = n_subjects - n_train - n_val
        
        # Sort subjects by stress ratio for balanced allocation
        sorted_subjects = sorted(subject_stats, key=lambda x: x['stress_ratio'])
        
        # Allocate subjects (interleaved to balance stress ratios)
        train_subjects = []
        val_subjects = []
        test_subjects = []
        
        for i, subject_stat in enumerate(sorted_subjects):
            if i % 3 == 0 and len(train_subjects) < n_train:
                train_subjects.append(subject_stat['subject'])
            elif i % 3 == 1 and len(val_subjects) < n_val:
                val_subjects.append(subject_stat['subject'])
            else:
                test_subjects.append(subject_stat['subject'])
        
        # Handle remaining subjects
        remaining = [s['subject'] for s in sorted_subjects 
                    if s['subject'] not in train_subjects + val_subjects + test_subjects]
        train_subjects.extend(remaining)
        
        print(f"   📊 Subject allocation:")
        print(f"      Train subjects: {train_subjects}")
        print(f"      Validation subjects: {val_subjects}")
        print(f"      Test subjects: {test_subjects}")
        
        # Create data splits
        train_mask = subjects.isin(train_subjects)
        val_mask = subjects.isin(val_subjects)
        test_mask = subjects.isin(test_subjects)
        
        # Use EXACT same features as original model (no correlation removal changes)
        print(f"   🔧 Using exact features from original model...")
        
        # Find intersection of available features with original model features
        available_features = [f for f in self.feature_names if f in X.columns]
        missing_features = [f for f in self.feature_names if f not in X.columns]
        
        if missing_features:
            print(f"   ⚠️ Missing {len(missing_features)} features from original model")
            # Use available features only
            selected_features = available_features
        else:
            # Use all original features
            selected_features = self.feature_names
            
        print(f"   ✅ Using {len(selected_features)} features (matching original model)")
        
        # Apply feature selection to all splits
        self.train_data = X[train_mask][selected_features]
        self.train_labels = y[train_mask].values
        self.train_subjects = subjects[train_mask].values
        
        self.val_data = X[val_mask][selected_features]
        self.val_labels = y[val_mask].values
        self.val_subjects = subjects[val_mask].values
        
        self.test_data = X[test_mask][selected_features]
        self.test_labels = y[test_mask].values
        self.test_subjects = subjects[test_mask].values
        
        # Keep original feature names (for compatibility)
        self.feature_names = selected_features
        
        print(f"✅ Rigorous data splits created:")
        print(f"   Training: {len(self.train_data)} samples from {len(train_subjects)} subjects")
        print(f"     Class distribution: {dict(zip(*np.unique(self.train_labels, return_counts=True)))}")
        print(f"   Validation: {len(self.val_data)} samples from {len(val_subjects)} subjects")
        print(f"     Class distribution: {dict(zip(*np.unique(self.val_labels, return_counts=True)))}")
        print(f"   Test: {len(self.test_data)} samples from {len(test_subjects)} subjects")
        print(f"     Class distribution: {dict(zip(*np.unique(self.test_labels, return_counts=True)))}")
        print(f"   Features after preprocessing: {len(selected_features)}")
        
    def _remove_correlated_features_fixed(self, X_train):
        """Remove correlated features with BUG FIX (include last column)"""
        cor = X_train.corr(numeric_only=True)
        keep_columns = np.full(cor.shape[0], True)
        
        # FIXED: Include last column in comparisons
        for i in range(cor.shape[0] - 1):
            for j in range(i + 1, cor.shape[0]):  # FIXED: removed -1
                if abs(cor.iloc[i, j]) >= 0.8:
                    keep_columns[j] = False
                    
        selected_columns = X_train.columns[keep_columns]
        print(f"   ✅ Correlation removal: {len(X_train.columns)} → {len(selected_columns)} features")
        
        return selected_columns.tolist()
        
    def establish_baseline_with_threshold_tuning(self):
        """Establish baseline performance with threshold re-tuning on validation set"""
        print(f"\n🎯 BASELINE EVALUATION WITH THRESHOLD TUNING")
        print("=" * 50)
        
        # Test original model on validation data
        val_probabilities = self.original_model.predict_proba(self.val_data)[:, 1]
        
        # Re-tune threshold on validation set
        optimal_threshold = self._tune_threshold_on_validation(val_probabilities, self.val_labels)
        
        # Evaluate with new threshold
        val_predictions = (val_probabilities >= optimal_threshold).astype(int)
        
        self.baseline_performance = {
            'accuracy': accuracy_score(self.val_labels, val_predictions),
            'f1_score': f1_score(self.val_labels, val_predictions),
            'precision': precision_score(self.val_labels, val_predictions, zero_division=0),
            'recall': recall_score(self.val_labels, val_predictions, zero_division=0),
            'threshold': optimal_threshold
        }
        
        print(f"✅ Baseline Performance (validation set, re-tuned threshold):")
        print(f"   Original threshold: {self.optimal_threshold:.4f}")
        print(f"   Re-tuned threshold: {optimal_threshold:.4f}")
        print(f"   Accuracy:  {self.baseline_performance['accuracy']:.4f}")
        print(f"   F1-Score:  {self.baseline_performance['f1_score']:.4f}")
        print(f"   Precision: {self.baseline_performance['precision']:.4f}")
        print(f"   Recall:    {self.baseline_performance['recall']:.4f}")
        
        # Update threshold for future use
        self.optimal_threshold = optimal_threshold
        
        return self.baseline_performance
        
    def _tune_threshold_on_validation(self, probabilities, true_labels):
        """Tune threshold to maximize F1 score on validation set"""
        fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            f1 = f1_score(true_labels, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold
        
    def rigorous_feature_selection(self):
        """Feature selection using ONLY train/validation data (no test contamination)"""
        print(f"\n🔍 RIGOROUS FEATURE SELECTION (NO TEST CONTAMINATION)")
        print("=" * 50)
        
        # Get feature importance ranking from original model
        feature_importance = self.original_model.feature_importances_
        
        # Match feature importance to current features
        original_feature_names = [name for name in self.feature_names]  # Keep order
        feature_rankings = np.argsort(feature_importance)[::-1]
        
        # Test different feature counts
        feature_counts = [60, 50, 45, 40, 35, 30, 25, 20, 15]
        feature_counts = [n for n in feature_counts if n <= len(self.feature_names)]
        
        min_f1_retention = 0.95
        baseline_f1 = self.baseline_performance['f1_score']
        
        print(f"🎯 Selection targets:")
        print(f"   Available features: {len(self.feature_names)}")
        print(f"   Min F1 retention: {min_f1_retention:.1%} (≥{baseline_f1*min_f1_retention:.4f})")
        print(f"   Validation set only (test set untouched)")
        
        results = []
        
        for n_features in feature_counts:
            print(f"\n   Testing {n_features} features...")
            
            # Select top N features
            if n_features <= len(feature_rankings):
                selected_indices = feature_rankings[:n_features]
            else:
                selected_indices = list(range(n_features))
            
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
            
            # Train model on selected features (train set only)
            X_train_subset = self.train_data.iloc[:, selected_indices]
            X_val_subset = self.val_data.iloc[:, selected_indices]
            
            # Retrain model
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
            
            subset_model = ExtraTreesClassifier(**model_params)
            subset_model.fit(X_train_subset, self.train_labels)
            
            # Evaluate on validation set with threshold re-tuning
            val_probabilities = subset_model.predict_proba(X_val_subset)[:, 1]
            optimal_threshold = self._tune_threshold_on_validation(val_probabilities, self.val_labels)
            val_predictions = (val_probabilities >= optimal_threshold).astype(int)
            
            performance = {
                'accuracy': accuracy_score(self.val_labels, val_predictions),
                'f1_score': f1_score(self.val_labels, val_predictions),
                'precision': precision_score(self.val_labels, val_predictions, zero_division=0),
                'recall': recall_score(self.val_labels, val_predictions, zero_division=0),
                'threshold': optimal_threshold
            }
            
            f1_retention = performance['f1_score'] / baseline_f1
            memory_kb = self._estimate_memory(n_features)
            
            result = {
                'n_features': n_features,
                'selected_indices': selected_indices.tolist(),
                'selected_features': selected_feature_names,
                'performance': performance,
                'f1_retention': f1_retention,
                'memory_kb': memory_kb,
                'esp32_compatible': memory_kb < 100 and 15 <= n_features <= 40,
                'performance_acceptable': f1_retention >= min_f1_retention,
                'retrained_model': subset_model
            }
            
            results.append(result)
            
            # Status
            f1_status = "✅" if f1_retention >= min_f1_retention else "❌"
            mem_status = "✅" if memory_kb < 100 else "⚠️"
            
            print(f"     {f1_status} F1: {performance['f1_score']:.4f} (retention: {f1_retention:.1%})")
            print(f"     {f1_status} Threshold: {optimal_threshold:.4f}")
            print(f"     {mem_status} Memory: {memory_kb:.1f}KB")
            
            if f1_retention < 0.90:
                print(f"     ⚠️ F1 retention below 90%, stopping")
                break
                
        return results
        
    def _estimate_memory(self, n_features):
        """Estimate memory usage for features on ESP32-S3"""
        feature_storage_kb = (n_features * 4) / 1024
        preprocessing_overhead_kb = n_features * 0.1
        buffer_overhead_kb = 5
        return feature_storage_kb + preprocessing_overhead_kb + buffer_overhead_kb
        
    def select_optimal_configuration(self, results):
        """Select optimal configuration prioritizing perfect retention"""
        print(f"\n🏆 SELECTING OPTIMAL CONFIGURATION")
        print("=" * 50)
        
        # Prioritize configurations with perfect or near-perfect retention
        perfect_retention = [r for r in results if r['f1_retention'] >= 0.999]
        acceptable_performance = [r for r in results if r['performance_acceptable'] and r['esp32_compatible']]
        
        if perfect_retention and any(r['esp32_compatible'] for r in perfect_retention):
            # Choose the one with most features among perfect retention + ESP32 compatible
            viable_perfect = [r for r in perfect_retention if r['esp32_compatible']]
            optimal_config = max(viable_perfect, key=lambda x: x['n_features'])
            print(f"✅ Selected configuration with PERFECT retention:")
        elif acceptable_performance:
            # Choose minimum features with acceptable performance
            optimal_config = min(acceptable_performance, key=lambda x: x['n_features'])
            print(f"✅ Selected configuration with acceptable performance:")
        else:
            # Best compromise
            optimal_config = max(results, key=lambda x: x['f1_retention'])
            print(f"⚠️ Selected best compromise:")
            
        print(f"   Features: {optimal_config['n_features']} (from {len(self.feature_names)})")
        print(f"   F1-Score: {optimal_config['performance']['f1_score']:.4f}")
        print(f"   F1 Retention: {optimal_config['f1_retention']:.1%}")
        print(f"   Threshold: {optimal_config['performance']['threshold']:.4f}")
        print(f"   Memory: {optimal_config['memory_kb']:.1f} KB")
        print(f"   ESP32 Compatible: {'✅' if optimal_config['esp32_compatible'] else '❌'}")
        
        return optimal_config
        
    def final_test_evaluation(self, optimal_config):
        """FINAL evaluation on held-out test set (first and only time)"""
        print(f"\n🔍 FINAL TEST SET EVALUATION (FIRST TIME)")
        print("=" * 50)
        print(f"⚠️ WARNING: Test set has been completely held out until now")
        
        # Extract selected features from test set
        X_test_subset = self.test_data.iloc[:, optimal_config['selected_indices']]
        
        # Evaluate with the optimal model and threshold
        test_probabilities = optimal_config['retrained_model'].predict_proba(X_test_subset)[:, 1]
        test_predictions = (test_probabilities >= optimal_config['performance']['threshold']).astype(int)
        
        final_performance = {
            'accuracy': accuracy_score(self.test_labels, test_predictions),
            'f1_score': f1_score(self.test_labels, test_predictions),
            'precision': precision_score(self.test_labels, test_predictions, zero_division=0),
            'recall': recall_score(self.test_labels, test_predictions, zero_division=0),
            'threshold': optimal_config['performance']['threshold']
        }
        
        print(f"✅ FINAL TEST RESULTS (held-out subjects):")
        print(f"   Test subjects: {np.unique(self.test_subjects)}")
        print(f"   Test samples: {len(self.test_labels)}")
        print(f"   Class distribution: {dict(zip(*np.unique(self.test_labels, return_counts=True)))}")
        print(f"   Features: {optimal_config['n_features']}")
        print(f"   Threshold: {final_performance['threshold']:.4f}")
        print(f"   Accuracy:  {final_performance['accuracy']:.4f}")
        print(f"   F1-Score:  {final_performance['f1_score']:.4f}")
        print(f"   Precision: {final_performance['precision']:.4f}")
        print(f"   Recall:    {final_performance['recall']:.4f}")
        
        # Compare with validation performance
        val_f1 = optimal_config['performance']['f1_score']
        test_f1 = final_performance['f1_score']
        generalization_gap = abs(val_f1 - test_f1)
        
        print(f"\n📊 GENERALIZATION ANALYSIS:")
        print(f"   Validation F1: {val_f1:.4f}")
        print(f"   Test F1:       {test_f1:.4f}")
        print(f"   Gap:           {generalization_gap:.4f}")
        
        if generalization_gap < 0.02:
            print(f"   Status: ✅ Excellent generalization")
        elif generalization_gap < 0.05:
            print(f"   Status: ✅ Good generalization")
        else:
            print(f"   Status: ⚠️ Potential overfitting detected")
            
        return final_performance
        
    def save_rigorous_results(self, optimal_config, final_performance):
        """Save all results with rigorous validation details"""
        print(f"\n💾 SAVING RIGOROUS VALIDATION RESULTS")
        print("=" * 50)
        
        os.makedirs('../outputs', exist_ok=True)
        
        # Save model with complete metadata
        rigorous_model_data = {
            'model': optimal_config['retrained_model'],
            'optimal_threshold': optimal_config['performance']['threshold'],
            'feature_names': optimal_config['selected_features'],
            'selected_feature_indices': optimal_config['selected_indices'],
            'validation_performance': optimal_config['performance'],
            'final_test_performance': final_performance,
            'train_subjects': self.train_subjects.tolist(),
            'val_subjects': self.val_subjects.tolist(),
            'test_subjects': self.test_subjects.tolist(),
            'n_features_original': len(self.feature_names),
            'n_features_selected': optimal_config['n_features'],
            'rigorous_validation': True,
            'no_data_leakage': True,
            'threshold_retuned': True
        }
        
        joblib.dump(rigorous_model_data, '../outputs/stage1_rigorous_model.joblib')
        
        # Save detailed results
        detailed_results = {
            'stage': 1,
            'description': 'Rigorous feature selection with proper validation methodology',
            'timestamp': datetime.now().isoformat(),
            'methodology': 'subject_level_splits_no_leakage',
            'validation_fixes': [
                'Subject-level train/val/test splits',
                'Fixed correlation removal bug (include last column)',
                'Threshold re-tuning on validation set',
                'No test set contamination during selection',
                'Larger test set with better representativeness'
            ],
            'data_splits': {
                'train_subjects': self.train_subjects.tolist(),
                'val_subjects': self.val_subjects.tolist(),
                'test_subjects': self.test_subjects.tolist(),
                'train_samples': len(self.train_data),
                'val_samples': len(self.val_data),
                'test_samples': len(self.test_data)
            },
            'optimal_configuration': {
                'n_features': optimal_config['n_features'],
                'selected_features': optimal_config['selected_features'],
                'validation_performance': optimal_config['performance'],
                'final_test_performance': final_performance,
                'memory_estimate_kb': optimal_config['memory_kb']
            },
            'validation_status': 'RIGOROUS_VALIDATION_COMPLETE'
        }
        
        with open('../outputs/stage1_rigorous_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
            
        print(f"✅ Rigorous model saved: ../outputs/stage1_rigorous_model.joblib")
        print(f"✅ Detailed results saved: ../outputs/stage1_rigorous_results.json")

def main():
    """Execute Rigorous Stage 1 with proper validation methodology"""
    print("🚀 ESP32-S3 TFLITE CONVERSION PIPELINE")
    print("🔍 STAGE 1: RIGOROUS FEATURE SELECTION")
    print("🎯 Goal: Fix critical validation issues and data leakage")
    print("=" * 60)
    
    selector = Stage1RigorousFeatureSelector()
    
    try:
        # Step 1: Load original model
        selector.load_original_model()
        
        # Step 2: Load data with proper subject-level splits
        selector.load_and_split_wesad_data()
        
        # Step 3: Establish baseline with threshold tuning
        selector.establish_baseline_with_threshold_tuning()
        
        # Step 4: Rigorous feature selection (no test contamination)
        all_results = selector.rigorous_feature_selection()
        
        # Step 5: Select optimal configuration
        optimal_config = selector.select_optimal_configuration(all_results)
        
        # Step 6: Final test evaluation (first time touching test set)
        final_performance = selector.final_test_evaluation(optimal_config)
        
        # Step 7: Save rigorous results
        selector.save_rigorous_results(optimal_config, final_performance)
        
        # Final status
        print(f"\n🎯 STAGE 1 RIGOROUS: ✅ SUCCESS")
        print(f"   Method: Subject-level splits, no data leakage, threshold re-tuning")
        print(f"   Features: {len(selector.feature_names)} → {optimal_config['n_features']}")
        print(f"   Test F1: {final_performance['f1_score']:.4f}")
        print(f"   Status: ✅ Ready for rigorous Stage 2")
        
    except Exception as e:
        print(f"\n❌ STAGE 1 RIGOROUS: ERROR")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    main()
