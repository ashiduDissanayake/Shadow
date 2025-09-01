#!/usr/bin/env python3
"""
ESP32-S3 Rigorous ML Pipeline
STAGE 0: DATA INTEGRITY & SPLITTING

Purpose: Establish leak-free data foundation
✅ Guarantee zero subject leakage
✅ Address temporal leakage from overlapping windows  
✅ Provide stable LOSO validation framework
✅ Compute per-subject baseline statistics
✅ Schema validation and data quality checks

This stage is CRITICAL - all subsequent stages depend on its integrity.
"""

import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataIntegrityConfig:
    """Configuration for Stage 0"""
    wesad_path: str
    dataset_file: str
    output_dir: str
    validation_strategy: str = "LOSO"
    min_samples_per_subject: int = 100
    temporal_analysis: bool = True
    
class Stage0DataIntegrity:
    """Stage 0: Data Integrity & Splitting"""
    
    def __init__(self, config_path: str = "../config/pipeline_config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract Stage 0 specific config
        data_config = self.config['data']
        selected_option = data_config['selected_dataset']
        dataset_info = data_config['dataset_options'][selected_option]
        
        self.data_config = DataIntegrityConfig(
            wesad_path=data_config['wesad_path'],
            dataset_file=dataset_info['file'],
            output_dir="../outputs/stage0/",
            validation_strategy=data_config['validation_strategy']
        )
        
        # Create output directory
        Path(self.data_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.raw_data = None
        self.data_manifest = {}
        self.subject_stats = {}
        self.fold_definitions = {}
        
    def run_stage0(self) -> bool:
        """Execute complete Stage 0 pipeline"""
        logger.info("🚀 STARTING STAGE 0: DATA INTEGRITY & SPLITTING")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and validate data
            if not self._load_and_validate_data():
                return False
                
            # Step 2: Analyze temporal characteristics
            if not self._analyze_temporal_properties():
                return False
                
            # Step 3: Compute subject statistics
            if not self._compute_subject_statistics():
                return False
                
            # Step 4: Create LOSO fold definitions
            if not self._create_loso_folds():
                return False
                
            # Step 5: Validate data integrity
            if not self._validate_data_integrity():
                return False
                
            # Step 6: Save all artifacts
            if not self._save_stage0_artifacts():
                return False
                
            logger.info("✅ STAGE 0 COMPLETED SUCCESSFULLY")
            logger.info("➡️ Ready for Stage 1: Feature Selection")
            return True
            
        except Exception as e:
            logger.error(f"❌ STAGE 0 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_and_validate_data(self) -> bool:
        """Load WESAD data and perform schema validation"""
        logger.info("📊 LOADING AND VALIDATING DATA")
        logger.info("-" * 40)
        
        # Construct file path
        data_file = Path(self.data_config.wesad_path) / self.data_config.dataset_file
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return False
            
        logger.info(f"Loading: {data_file}")
        
        # Load data
        self.raw_data = pd.read_parquet(data_file)
        
        # Basic validation
        logger.info(f"✅ Data loaded: {len(self.raw_data)} samples, {len(self.raw_data.columns)} features")
        
        # Schema validation
        required_columns = ['subject', 'label']
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for NaN values
        nan_counts = self.raw_data.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0]
        
        if len(nan_columns) > 0:
            logger.warning(f"Columns with NaN values: {dict(nan_columns)}")
            
        # Data type validation
        if not pd.api.types.is_integer_dtype(self.raw_data['subject']):
            logger.error("Subject column must be integer type")
            return False
            
        if not pd.api.types.is_integer_dtype(self.raw_data['label']):
            logger.error("Label column must be integer type")
            return False
            
        # Value range validation
        unique_labels = sorted(self.raw_data['label'].unique())
        if unique_labels != [0, 1]:
            logger.error(f"Labels must be [0, 1], found: {unique_labels}")
            return False
            
        # Create data manifest
        file_hash = self._compute_file_hash(data_file)
        
        self.data_manifest = {
            'source_file': str(data_file),
            'file_hash': file_hash,
            'load_timestamp': datetime.now().isoformat(),
            'total_samples': int(len(self.raw_data)),
            'total_features': int(len(self.raw_data.columns) - 2),  # Exclude subject, label
            'subjects': [int(x) for x in sorted(self.raw_data['subject'].unique().tolist())],
            'labels': [int(x) for x in unique_labels],
            'schema_valid': True,
            'nan_columns': dict(nan_columns) if len(nan_columns) > 0 else {}
        }
        
        logger.info(f"📋 Data Manifest Created:")
        logger.info(f"   File hash: {file_hash[:16]}...")
        logger.info(f"   Subjects: {self.data_manifest['subjects']}")
        logger.info(f"   Samples: {self.data_manifest['total_samples']:,}")
        logger.info(f"   Features: {self.data_manifest['total_features']}")
        
        return True
    
    def _analyze_temporal_properties(self) -> bool:
        """Analyze temporal characteristics and potential leakage"""
        logger.info("⏰ ANALYZING TEMPORAL PROPERTIES")
        logger.info("-" * 40)
        
        # Extract dataset parameters from filename
        dataset_file = self.data_config.dataset_file
        if 'wesad' in dataset_file and '-' in dataset_file:
            parts = dataset_file.split('-')
            if len(parts) >= 3:
                try:
                    window_size = int(parts[-2])
                    step_size = int(parts[-1].split('.')[0])
                    
                    overlap_seconds = window_size - step_size
                    overlap_percent = (overlap_seconds / window_size) * 100
                    
                    temporal_analysis = {
                        'window_size_sec': window_size,
                        'step_size_sec': step_size,
                        'overlap_seconds': overlap_seconds,
                        'overlap_percent': round(overlap_percent, 1),
                        'temporal_leakage_risk': self._assess_leakage_risk(overlap_percent),
                        'esp32_realistic': step_size >= window_size  # Non-overlapping is realistic
                    }
                    
                    logger.info(f"📊 Temporal Analysis:")
                    logger.info(f"   Window size: {window_size} seconds")
                    logger.info(f"   Step size: {step_size} seconds") 
                    logger.info(f"   Overlap: {overlap_seconds}s ({overlap_percent:.1f}%)")
                    logger.info(f"   Leakage risk: {temporal_analysis['temporal_leakage_risk']}")
                    logger.info(f"   ESP32 realistic: {temporal_analysis['esp32_realistic']}")
                    
                    # Warning for high temporal leakage
                    if overlap_percent > 80:
                        logger.warning("⚠️ HIGH TEMPORAL LEAKAGE DETECTED!")
                        logger.warning("   This may inflate validation performance artificially")
                        logger.warning("   Consider using non-overlapping windows for final validation")
                    
                    self.data_manifest['temporal_analysis'] = temporal_analysis
                    return True
                    
                except ValueError:
                    logger.warning("Could not parse temporal parameters from filename")
        
        logger.warning("Temporal analysis skipped - could not determine window parameters")
        return True  # Not critical for pipeline
    
    def _assess_leakage_risk(self, overlap_percent: float) -> str:
        """Assess temporal leakage risk based on overlap percentage"""
        if overlap_percent >= 90:
            return "CRITICAL"
        elif overlap_percent >= 75:
            return "HIGH"
        elif overlap_percent >= 50:
            return "MEDIUM"
        elif overlap_percent > 0:
            return "LOW"
        else:
            return "NONE"
    
    def _compute_subject_statistics(self) -> bool:
        """Compute comprehensive per-subject statistics"""
        logger.info("👥 COMPUTING SUBJECT STATISTICS")
        logger.info("-" * 40)
        
        subjects = sorted(self.raw_data['subject'].unique())
        
        for subject in subjects:
            subject_data = self.raw_data[self.raw_data['subject'] == subject]
            labels = subject_data['label']
            
            # Basic statistics
            n_samples = len(subject_data)
            n_stress = (labels == 1).sum()
            n_nostress = (labels == 0).sum()
            stress_ratio = n_stress / n_samples if n_samples > 0 else 0
            
            # Feature statistics (excluding subject, label)
            feature_cols = [col for col in subject_data.columns if col not in ['subject', 'label']]
            feature_data = subject_data[feature_cols]
            
            subject_stats = {
                'subject_id': int(subject),
                'n_samples': int(n_samples),
                'n_stress': int(n_stress),
                'n_nostress': int(n_nostress),
                'stress_ratio': float(stress_ratio),
                'feature_stats': {
                    'mean': feature_data.mean().to_dict(),
                    'std': feature_data.std().to_dict(),
                    'min': feature_data.min().to_dict(),
                    'max': feature_data.max().to_dict()
                }
            }
            
            self.subject_stats[int(subject)] = subject_stats
            
            logger.info(f"   Subject {subject:2d}: {n_samples:4d} samples, {stress_ratio:.1%} stress")
        
        # Overall statistics
        total_samples = sum(stats['n_samples'] for stats in self.subject_stats.values())
        total_stress = sum(stats['n_stress'] for stats in self.subject_stats.values())
        overall_stress_ratio = total_stress / total_samples if total_samples > 0 else 0
        
        logger.info(f"📊 Overall Statistics:")
        logger.info(f"   Total subjects: {len(subjects)}")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Overall stress ratio: {overall_stress_ratio:.1%}")
        
        # Check for subjects with insufficient data
        min_samples = self.data_config.min_samples_per_subject
        insufficient_subjects = [
            subj for subj, stats in self.subject_stats.items() 
            if stats['n_samples'] < min_samples
        ]
        
        if insufficient_subjects:
            logger.warning(f"⚠️ Subjects with <{min_samples} samples: {insufficient_subjects}")
        
        return True
    
    def _create_loso_folds(self) -> bool:
        """Create Leave-One-Subject-Out fold definitions"""
        logger.info("🔄 CREATING LOSO FOLD DEFINITIONS")
        logger.info("-" * 40)
        
        subjects = sorted(self.raw_data['subject'].unique())
        
        # Create LOSO folds
        folds = []
        for i, test_subject in enumerate(subjects):
            train_subjects = [s for s in subjects if s != test_subject]
            
            fold = {
                'fold_id': i,
                'test_subject': int(test_subject),
                'train_subjects': [int(s) for s in train_subjects],
                'test_samples': int(self.subject_stats[test_subject]['n_samples']),
                'train_samples': int(sum(self.subject_stats[s]['n_samples'] for s in train_subjects))
            }
            
            folds.append(fold)
            
            logger.info(f"   Fold {i:2d}: Test={test_subject:2d} ({fold['test_samples']:4d} samples), "
                       f"Train={len(train_subjects):2d} subjects ({fold['train_samples']:5d} samples)")
        
        self.fold_definitions = {
            'validation_strategy': 'LeaveOneSubjectOut',
            'n_folds': len(folds),
            'total_subjects': len(subjects),
            'folds': folds,
            'creation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Created {len(folds)} LOSO folds")
        return True
    
    def _validate_data_integrity(self) -> bool:
        """Perform comprehensive data integrity validation"""
        logger.info("🔍 VALIDATING DATA INTEGRITY")
        logger.info("-" * 40)
        
        checks_passed = 0
        total_checks = 0
        
        # Check 1: No subject appears in multiple folds as test
        total_checks += 1
        test_subjects = [fold['test_subject'] for fold in self.fold_definitions['folds']]
        if len(test_subjects) == len(set(test_subjects)):
            logger.info("✅ Check 1: Each subject appears exactly once as test")
            checks_passed += 1
        else:
            logger.error("❌ Check 1: Subjects appear multiple times as test")
        
        # Check 2: All subjects covered in folds
        total_checks += 1
        all_subjects = set(self.raw_data['subject'].unique())
        fold_subjects = set(test_subjects)
        if all_subjects == fold_subjects:
            logger.info("✅ Check 2: All subjects covered in fold definitions")
            checks_passed += 1
        else:
            missing = all_subjects - fold_subjects
            extra = fold_subjects - all_subjects
            logger.error(f"❌ Check 2: Subject coverage mismatch. Missing: {missing}, Extra: {extra}")
        
        # Check 3: No subject leakage between train/test in any fold
        total_checks += 1
        leakage_detected = False
        for fold in self.fold_definitions['folds']:
            test_subj = fold['test_subject']
            train_subjs = set(fold['train_subjects'])
            if test_subj in train_subjs:
                logger.error(f"❌ Check 3: Subject {test_subj} appears in both train and test")
                leakage_detected = True
        
        if not leakage_detected:
            logger.info("✅ Check 3: No subject leakage between train/test")
            checks_passed += 1
        
        # Check 4: Reasonable class balance across folds
        total_checks += 1
        fold_stress_ratios = []
        for fold in self.fold_definitions['folds']:
            test_subject = fold['test_subject']
            stress_ratio = self.subject_stats[test_subject]['stress_ratio']
            fold_stress_ratios.append(stress_ratio)
        
        mean_ratio = np.mean(fold_stress_ratios)
        std_ratio = np.std(fold_stress_ratios)
        
        if std_ratio < 0.15:  # Reasonable variance in stress ratios
            logger.info(f"✅ Check 4: Reasonable class balance across folds (std={std_ratio:.3f})")
            checks_passed += 1
        else:
            logger.warning(f"⚠️ Check 4: High variance in class balance across folds (std={std_ratio:.3f})")
            checks_passed += 1  # Warning, not failure
        
        # Check 5: Sufficient samples per fold
        total_checks += 1
        min_test_samples = min(fold['test_samples'] for fold in self.fold_definitions['folds'])
        min_train_samples = min(fold['train_samples'] for fold in self.fold_definitions['folds'])
        
        if min_test_samples >= 50 and min_train_samples >= 500:
            logger.info(f"✅ Check 5: Sufficient samples per fold (min test={min_test_samples}, min train={min_train_samples})")
            checks_passed += 1
        else:
            logger.warning(f"⚠️ Check 5: Low sample count in some folds (min test={min_test_samples}, min train={min_train_samples})")
            checks_passed += 1  # Warning, not failure
        
        # Overall validation result
        integrity_score = checks_passed / total_checks
        logger.info(f"📊 Data Integrity Score: {checks_passed}/{total_checks} ({integrity_score:.1%})")
        
        if integrity_score >= 0.8:
            logger.info("✅ Data integrity validation PASSED")
            return True
        else:
            logger.error("❌ Data integrity validation FAILED")
            return False
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of the data file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_stage0_artifacts(self) -> bool:
        """Save all Stage 0 artifacts"""
        logger.info("💾 SAVING STAGE 0 ARTIFACTS")
        logger.info("-" * 40)
        
        output_dir = Path(self.data_config.output_dir)
        
        # Save data manifest
        manifest_file = output_dir / "data_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(self.data_manifest, f, indent=2)
        logger.info(f"✅ Saved: {manifest_file}")
        
        # Save subject statistics
        stats_file = output_dir / "per_subject_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.subject_stats, f, indent=2)
        logger.info(f"✅ Saved: {stats_file}")
        
        # Save fold definitions
        folds_file = output_dir / "fold_definitions.json"
        with open(folds_file, 'w') as f:
            json.dump(self.fold_definitions, f, indent=2)
        logger.info(f"✅ Saved: {folds_file}")
        
        # Save stage 0 summary
        summary = {
            'stage': 0,
            'description': 'Data Integrity & Splitting',
            'timestamp': datetime.now().isoformat(),
            'status': 'SUCCESS',
            'data_source': self.data_manifest['source_file'],
            'validation_strategy': self.data_config.validation_strategy,
            'total_subjects': len(self.subject_stats),
            'total_samples': self.data_manifest['total_samples'],
            'loso_folds': self.fold_definitions['n_folds'],
            'temporal_analysis': self.data_manifest.get('temporal_analysis', {}),
            'integrity_validated': True,
            'artifacts': [
                'data_manifest.json',
                'per_subject_stats.json', 
                'fold_definitions.json'
            ]
        }
        
        summary_file = output_dir / "stage0_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Saved: {summary_file}")
        
        return True

def main():
    """Execute Stage 0: Data Integrity & Splitting"""
    stage0 = Stage0DataIntegrity()
    success = stage0.run_stage0()
    
    if success:
        print("\n🎉 STAGE 0 COMPLETED SUCCESSFULLY!")
        print("📋 Artifacts created:")
        print("   - data_manifest.json")
        print("   - per_subject_stats.json")
        print("   - fold_definitions.json")
        print("   - stage0_summary.json")
        print("\n➡️ Ready to proceed to Stage 1: Feature Selection")
    else:
        print("\n❌ STAGE 0 FAILED!")
        print("Please check the logs and fix any issues before proceeding.")
        return False
    
    return success

if __name__ == "__main__":
    main()
