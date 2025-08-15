"""
WESAD Dataset Loading and Processing Module

This module provides comprehensive functionality for loading and processing
the WESAD (Wearable Stress and Affect Detection) dataset, specifically
optimized for the ShadowAI stress detection pipeline.

Features:
- Multi-subject data loading with robust error handling
- Automatic zip file extraction and management
- BVP signal isolation and quality assessment
- Multi-modal sensor data support (BVP, EDA, TEMP, ACC)
- Data validation and consistency checks
- Stress/baseline/amusement classification support
- Memory-efficient data processing for large datasets

Author: Shadow AI Team
License: MIT
"""

import os
import pickle
import zipfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class WESADLoader:
    """
    Advanced loader for the WESAD (Wearable Stress and Affect Detection) dataset.
    
    The WESAD dataset contains multimodal physiological data from chest and wrist-worn
    devices for stress detection research. This loader provides optimized access to:
    - BVP (Blood Volume Pulse) at 64 Hz
    - EDA (Electrodermal Activity) at 4 Hz  
    - TEMP (Temperature) at 4 Hz
    - ACC (3-axis Accelerometer) at 32 Hz
    - Respiration signals from chest device
    
    Supports both individual and batch loading with comprehensive data validation.
    Automatically handles zip file extraction and management.
    """
    
    def __init__(self, 
                 data_path: str = "data/raw/wesad/",
                 zip_file_path: Optional[str] = None,
                 cache_enabled: bool = True,
                 validate_on_load: bool = True,
                 auto_extract: bool = True):
        """
        Initialize the WESAD dataset loader.
        
        Args:
            data_path: Path to the WESAD dataset directory containing subject folders
            zip_file_path: Path to the WESAD.zip file. If None, looks for WESAD.zip in data_path
            cache_enabled: Whether to cache processed data for faster subsequent loads
            validate_on_load: Whether to validate data integrity during loading
            auto_extract: Whether to automatically extract zip file if needed
        """
        self.data_path = Path(data_path)
        self.cache_enabled = cache_enabled
        self.validate_on_load = validate_on_load
        self.auto_extract = auto_extract
        self.cache_dir = self.data_path / ".cache"
        
        # Determine zip file path
        if zip_file_path is None:
            self.zip_file_path = self.data_path / "WESAD.zip"
        else:
            self.zip_file_path = Path(zip_file_path)
        
        # WESAD protocol labels mapping
        self.labels = {
            'baseline': 1,      # Baseline condition
            'stress': 2,        # Stress condition (TSST)
            'amusement': 3,     # Amusement condition (funny videos)
            'meditation': 4,    # Meditation/relaxation condition
            'transient': 0      # Transient periods between conditions
        }
        
        # Reverse mapping for label decoding
        self.label_names = {v: k for k, v in self.labels.items()}
        
        # Sampling rates for different modalities (Hz)
        self.sampling_rates = {
            'bvp': 64,
            'eda': 4,
            'temp': 4,
            'acc': 32,
            'resp': 700  # Chest sensor
        }
        
        # Create cache directory if enabled
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-extract if needed
        if self.auto_extract:
            self._ensure_extracted()
    
    def _ensure_extracted(self) -> bool:
        """
        Ensure the WESAD dataset is extracted and available.
        
        Returns:
            True if data is available, False otherwise
        """
        # Check if we already have extracted data
        subjects = self._get_available_subjects_from_directory()
        if subjects:
            logger.info(f"Found {len(subjects)} extracted subjects in {self.data_path}")
            return True
        
        # Check if zip file exists
        if not self.zip_file_path.exists():
            logger.error(f"WESAD zip file not found at {self.zip_file_path}")
            logger.info("Please download the WESAD dataset and place WESAD.zip in the data/raw/wesad/ directory")
            return False
        
        # Extract the zip file
        logger.info(f"Extracting WESAD dataset from {self.zip_file_path}")
        return self._extract_zip_file()
    
    def _extract_zip_file(self) -> bool:
        """
        Extract the WESAD zip file to the data directory.
        
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            # Create extraction directory
            self.data_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                
                # Filter for relevant files (pickle files and subject directories)
                relevant_files = [f for f in file_list if 
                                'S' in f and ('.pkl' in f or f.endswith('/'))]
                
                if not relevant_files:
                    logger.error("No relevant WESAD files found in zip archive")
                    return False
                
                logger.info(f"Extracting {len(relevant_files)} files...")
                
                # Extract all files
                zip_ref.extractall(self.data_path)
                
                # Check if extraction created a nested directory structure
                extracted_items = list(self.data_path.iterdir())
                
                # If there's a single directory that contains the actual data, move it up
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    nested_dir = extracted_items[0]
                    if any(item.name.startswith('S') for item in nested_dir.iterdir()):
                        logger.info("Moving extracted files from nested directory")
                        for item in nested_dir.iterdir():
                            shutil.move(str(item), str(self.data_path / item.name))
                        nested_dir.rmdir()
                
                # Verify extraction
                subjects = self._get_available_subjects_from_directory()
                if subjects:
                    logger.info(f"Successfully extracted data for {len(subjects)} subjects")
                    return True
                else:
                    logger.error("Extraction completed but no valid subjects found")
                    return False
                    
        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file: {self.zip_file_path}")
            return False
        except Exception as e:
            logger.error(f"Error extracting zip file: {str(e)}")
            return False
    
    def _get_available_subjects_from_directory(self) -> List[int]:
        """Get list of available subject IDs from the dataset directory."""
        if not self.data_path.exists():
            return []
        
        subjects = []
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name.startswith('S'):
                try:
                    subject_id = int(item.name[1:])
                    # Verify that the subject has a pickle file
                    pickle_file = item / f"S{subject_id}.pkl"
                    if pickle_file.exists():
                        subjects.append(subject_id)
                except ValueError:
                    continue
        
        return sorted(subjects)
    
    def check_dataset_status(self) -> Dict:
        """
        Check the current status of the WESAD dataset.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            'zip_exists': self.zip_file_path.exists(),
            'zip_path': str(self.zip_file_path),
            'extracted_data_exists': False,
            'extracted_subjects': [],
            'data_path': str(self.data_path),
            'ready_for_use': False
        }
        
        if status['zip_exists']:
            status['zip_size_mb'] = self.zip_file_path.stat().st_size / (1024 * 1024)
        
        # Check for extracted data
        subjects = self._get_available_subjects_from_directory()
        if subjects:
            status['extracted_data_exists'] = True
            status['extracted_subjects'] = subjects
            status['ready_for_use'] = True
        
        return status
    
    def load_bvp_data(self, 
                      subjects: Optional[List[int]] = None,
                      conditions: Optional[List[str]] = None,
                      quality_threshold: float = 0.7) -> Dict:
        """
        Load BVP data from WESAD dataset with quality filtering.
        
        Args:
            subjects: List of subject IDs to load. If None, loads all available subjects
            conditions: List of conditions to include ['baseline', 'stress', 'amusement', 'meditation']
            quality_threshold: Minimum signal quality score (0.0 to 1.0) for inclusion
            
        Returns:
            Dictionary containing BVP data and metadata for each subject:
            {
                subject_id: {
                    'bvp': np.ndarray,           # BVP signal values
                    'labels': np.ndarray,        # Condition labels
                    'timestamps': np.ndarray,    # Time indices
                    'sampling_rate': int,        # Sampling frequency (64 Hz)
                    'quality_score': float,      # Signal quality assessment
                    'conditions': List[str],     # Unique conditions present
                    'duration_minutes': float    # Total duration in minutes
                }
            }
        """
        # Ensure data is extracted
        if not self._ensure_extracted():
            logger.error("Cannot load data: WESAD dataset not available")
            return {}
        
        logger.info("Loading BVP data from WESAD dataset")
        
        if subjects is None:
            subjects = self._get_available_subjects()
            
        if conditions is None:
            conditions = ['baseline', 'stress', 'amusement', 'meditation']
            
        # Convert condition names to label values
        target_labels = [self.labels[cond] for cond in conditions if cond in self.labels]
        
        data = {}
        
        for subject_id in subjects:
            cache_key = f"bvp_s{subject_id}_{'_'.join(conditions)}.pkl"
            cache_path = self.cache_dir / cache_key
            
            # Try loading from cache first
            if self.cache_enabled and cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        subject_data = pickle.load(f)
                    logger.info(f"Loaded subject {subject_id} BVP data from cache")
                    data[subject_id] = subject_data
                    continue
                except Exception as e:
                    logger.warning(f"Cache read failed for subject {subject_id}: {e}")
            
            # Load data from source
            try:
                subject_data = self._load_subject_bvp(subject_id, target_labels, quality_threshold)
                if subject_data and subject_data['quality_score'] >= quality_threshold:
                    data[subject_id] = subject_data
                    
                    # Cache the processed data
                    if self.cache_enabled:
                        try:
                            with open(cache_path, 'wb') as f:
                                pickle.dump(subject_data, f)
                        except Exception as e:
                            logger.warning(f"Cache write failed for subject {subject_id}: {e}")
                    
                    logger.info(f"Successfully loaded subject {subject_id}: "
                              f"{len(subject_data['bvp'])} samples, "
                              f"quality={subject_data['quality_score']:.3f}")
                else:
                    logger.warning(f"Subject {subject_id} excluded due to low quality or loading error")
                    
            except Exception as e:
                logger.error(f"Error loading subject {subject_id}: {str(e)}")
                
        logger.info(f"Successfully loaded BVP data for {len(data)} subjects")
        return data
    
    def load_multimodal_data(self, 
                           subjects: Optional[List[int]] = None,
                           modalities: Optional[List[str]] = None) -> Dict:
        """
        Load multimodal sensor data from WESAD dataset.
        
        Args:
            subjects: List of subject IDs to load
            modalities: List of modalities to load ['bvp', 'eda', 'temp', 'acc']
            
        Returns:
            Dictionary containing all requested sensor data for each subject
        """
        # Ensure data is extracted
        if not self._ensure_extracted():
            logger.error("Cannot load data: WESAD dataset not available")
            return {}
        
        logger.info("Loading multimodal data from WESAD dataset")
        
        if subjects is None:
            subjects = self._get_available_subjects()
            
        if modalities is None:
            modalities = ['bvp', 'eda', 'temp', 'acc']
            
        data = {}
        
        for subject_id in subjects:
            try:
                subject_data = self._load_subject_multimodal(subject_id, modalities)
                if subject_data:
                    data[subject_id] = subject_data
                    logger.info(f"Successfully loaded multimodal data for subject {subject_id}")
                    
            except Exception as e:
                logger.error(f"Error loading multimodal data for subject {subject_id}: {str(e)}")
                
        return data
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the WESAD dataset.
        
        Returns:
            Dictionary containing dataset statistics and metadata
        """
        # Initialize base stats structure that always includes required keys
        stats = {
            'total_subjects': 0,
            'subject_ids': [],
            'available_modalities': list(self.sampling_rates.keys()),
            'sampling_rates': self.sampling_rates,
            'condition_labels': self.labels,
            'condition_names': list(self.labels.keys()),
            'protocol_info': {
                'baseline_duration': '20 minutes',
                'stress_duration': '10-15 minutes (TSST)',
                'amusement_duration': '5-10 minutes',
                'meditation_duration': '8 minutes'
            }
        }
        
        # Try to ensure data is extracted
        if not self._ensure_extracted():
            stats['error'] = 'WESAD dataset not available'
            stats['data_path'] = str(self.data_path)
            return stats
        
        subjects = self._get_available_subjects()
        
        # Update stats with actual data
        stats.update({
            'total_subjects': len(subjects),
            'subject_ids': subjects
        })
        
        # Analyze data availability per subject
        subject_stats = []
        for subject_id in subjects:
            subject_info = self._analyze_subject_data(subject_id)
            subject_stats.append(subject_info)
            
        stats['subject_analysis'] = subject_stats
        stats['data_path'] = str(self.data_path)
        
        return stats
    
    def validate_dataset(self) -> Dict:
        """
        Comprehensive validation of the WESAD dataset.
        
        Returns:
            Dictionary containing validation results and issues found
        """
        logger.info("Validating WESAD dataset integrity")
        
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'subject_status': {},
            'summary': {}
        }
        
        # Ensure data is extracted
        if not self._ensure_extracted():
            validation_results['valid'] = False
            validation_results['issues'].append("WESAD dataset not available or extractable")
            return validation_results
        
        subjects = self._get_available_subjects()
        
        if not subjects:
            validation_results['valid'] = False
            validation_results['issues'].append("No valid subject directories found")
            return validation_results
        
        valid_subjects = 0
        total_bvp_samples = 0
        
        for subject_id in subjects:
            subject_status = self._validate_subject(subject_id)
            validation_results['subject_status'][subject_id] = subject_status
            
            if subject_status['valid']:
                valid_subjects += 1
                total_bvp_samples += subject_status.get('bvp_samples', 0)
            else:
                validation_results['issues'].extend(subject_status['issues'])
        
        validation_results['summary'] = {
            'total_subjects': len(subjects),
            'valid_subjects': valid_subjects,
            'total_bvp_samples': total_bvp_samples,
            'avg_samples_per_subject': total_bvp_samples / max(valid_subjects, 1)
        }
        
        if valid_subjects == 0:
            validation_results['valid'] = False
            validation_results['issues'].append("No valid subjects found")
        
        logger.info(f"Dataset validation complete: {valid_subjects}/{len(subjects)} subjects valid")
        return validation_results
    
    def _load_subject_bvp(self, subject_id: int, target_labels: List[int], quality_threshold: float) -> Optional[Dict]:
        """Load and process BVP data for a single subject."""
        subject_dir = self.data_path / f"S{subject_id}"
        
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return None
        
        # Look for pickle file (original WESAD format)
        pickle_file = subject_dir / f"S{subject_id}.pkl"
        
        if not pickle_file.exists():
            logger.warning(f"Data file not found: {pickle_file}")
            return None
        
        try:
            # Load the pickle file
            with open(pickle_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract wrist data (contains BVP)
            wrist_data = subject_data['signal']['wrist']
            bvp_signal = wrist_data['BVP'].flatten()
            labels = subject_data['label'].flatten()
            
            # Fix sampling rate mismatch between BVP and labels
            if len(bvp_signal) != len(labels):
                logger.info(f"Sampling rate mismatch for subject {subject_id}: BVP={len(bvp_signal)}, Labels={len(labels)}")
                
                # Resample labels to match BVP length (typically labels are at higher sampling rate)
                if len(labels) > len(bvp_signal):
                    # Calculate resampling factor and store original length
                    original_label_length = len(labels)
                    resample_factor = len(labels) / len(bvp_signal)
                    # Create new indices for resampling
                    resampled_indices = np.floor(np.arange(len(bvp_signal)) * resample_factor).astype(int)
                    # Make sure indices are within bounds
                    resampled_indices = np.minimum(resampled_indices, len(labels) - 1)
                    # Resample labels to match BVP length
                    labels = labels[resampled_indices]
                    logger.info(f"Resampled labels from {original_label_length} to {len(labels)} samples")
                else:
                    # In rare case where BVP has more samples, truncate BVP
                    logger.warning(f"Unusual case: BVP has more samples than labels for subject {subject_id}")
                    bvp_signal = bvp_signal[:len(labels)]
            
            # Now filter for target conditions (labels and BVP are same length now)
            mask = np.isin(labels, target_labels)
            filtered_bvp = bvp_signal[mask]
            filtered_labels = labels[mask]
            
            if len(filtered_bvp) == 0:
                logger.warning(f"No data found for target conditions in subject {subject_id}")
                return None
            
            # Calculate quality score
            quality_score = self._assess_signal_quality(filtered_bvp)
            
            # Create timestamps
            timestamps = np.arange(len(filtered_bvp)) / self.sampling_rates['bvp']
            
            # Get unique conditions present
            unique_labels = np.unique(filtered_labels)
            conditions = [self.label_names[label] for label in unique_labels if label in self.label_names]
            
            return {
                'bvp': filtered_bvp,
                'labels': filtered_labels,
                'timestamps': timestamps,
                'sampling_rate': self.sampling_rates['bvp'],
                'quality_score': quality_score,
                'conditions': conditions,
                'duration_minutes': len(filtered_bvp) / self.sampling_rates['bvp'] / 60
            }
            
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            return None
    
    def _load_subject_multimodal(self, subject_id: int, modalities: List[str]) -> Optional[Dict]:
        """Load multimodal data for a single subject."""
        subject_dir = self.data_path / f"S{subject_id}"
        pickle_file = subject_dir / f"S{subject_id}.pkl"
        
        if not pickle_file.exists():
            return None
        
        try:
            with open(pickle_file, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            result = {}
            wrist_data = subject_data['signal']['wrist']
            chest_data = subject_data['signal']['chest']
            labels = subject_data['label'].flatten()
            
            # Load requested modalities
            for modality in modalities:
                if modality == 'bvp' and 'BVP' in wrist_data:
                    result['bvp'] = wrist_data['BVP'].flatten()
                elif modality == 'eda' and 'EDA' in wrist_data:
                    result['eda'] = wrist_data['EDA'].flatten()
                elif modality == 'temp' and 'TEMP' in wrist_data:
                    result['temp'] = wrist_data['TEMP'].flatten()
                elif modality == 'acc' and 'ACC' in wrist_data:
                    result['acc'] = wrist_data['ACC']  # 3D accelerometer data
                elif modality == 'resp' and 'Resp' in chest_data:
                    result['resp'] = chest_data['Resp'].flatten()
            
            result['labels'] = labels
            result['sampling_rates'] = self.sampling_rates
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading multimodal data for subject {subject_id}: {str(e)}")
            return None
    
    def _assess_signal_quality(self, signal: np.ndarray) -> float:
        """
        Assess BVP signal quality using multiple metrics.
        
        Args:
            signal: BVP signal array
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if len(signal) == 0:
            return 0.0
        
        # Check for valid range (typical BVP values)
        range_score = 1.0 if np.std(signal) > 0.01 else 0.0
        
        # Check for saturation
        saturation_score = 1.0 if len(np.unique(signal)) > len(signal) * 0.1 else 0.0
        
        # Check for reasonable variance
        variance_score = min(1.0, np.std(signal) / (np.mean(np.abs(signal)) + 1e-6))
        
        # Check for outliers
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        outlier_threshold = 3 * iqr
        outliers = np.sum(np.abs(signal - np.median(signal)) > outlier_threshold)
        outlier_score = max(0.0, 1.0 - outliers / len(signal))
        
        # Combine scores
        quality_score = np.mean([range_score, saturation_score, variance_score, outlier_score])
        
        return float(quality_score)
    
    def _get_available_subjects(self) -> List[int]:
        """Get list of available subject IDs from the dataset directory."""
        return self._get_available_subjects_from_directory()
    
    def _analyze_subject_data(self, subject_id: int) -> Dict:
        """Analyze data availability and quality for a single subject."""
        subject_dir = self.data_path / f"S{subject_id}"
        pickle_file = subject_dir / f"S{subject_id}.pkl"
        
        analysis = {
            'subject_id': subject_id,
            'data_available': False,
            'file_size_mb': 0,
            'modalities': [],
            'conditions': [],
            'total_duration_minutes': 0
        }
        
        if pickle_file.exists():
            analysis['data_available'] = True
            analysis['file_size_mb'] = pickle_file.stat().st_size / (1024 * 1024)
            
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Check available modalities
                if 'signal' in data:
                    wrist_data = data['signal'].get('wrist', {})
                    chest_data = data['signal'].get('chest', {})
                    
                    if 'BVP' in wrist_data:
                        analysis['modalities'].append('bvp')
                    if 'EDA' in wrist_data:
                        analysis['modalities'].append('eda')
                    if 'TEMP' in wrist_data:
                        analysis['modalities'].append('temp')
                    if 'ACC' in wrist_data:
                        analysis['modalities'].append('acc')
                    if 'Resp' in chest_data:
                        analysis['modalities'].append('resp')
                
                # Check available conditions
                if 'label' in data:
                    labels = data['label'].flatten()
                    unique_labels = np.unique(labels)
                    analysis['conditions'] = [self.label_names.get(label, f'unknown_{label}') 
                                            for label in unique_labels]
                    
                    # Estimate duration based on BVP sampling rate
                    if len(labels) > 0:
                        analysis['total_duration_minutes'] = len(labels) / self.sampling_rates['bvp'] / 60
                        
            except Exception as e:
                logger.warning(f"Error analyzing subject {subject_id}: {e}")
        
        return analysis
    
    def _validate_subject(self, subject_id: int) -> Dict:
        """Validate data integrity for a single subject."""
        validation = {
            'valid': False,
            'issues': [],
            'warnings': [],
            'bvp_samples': 0
        }
        
        subject_dir = self.data_path / f"S{subject_id}"
        pickle_file = subject_dir / f"S{subject_id}.pkl"
        
        if not pickle_file.exists():
            validation['issues'].append(f"Data file missing for subject {subject_id}")
            return validation
        
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Check data structure
            if 'signal' not in data:
                validation['issues'].append(f"Missing 'signal' key for subject {subject_id}")
                return validation
            
            if 'label' not in data:
                validation['issues'].append(f"Missing 'label' key for subject {subject_id}")
                return validation
            
            # Check BVP data
            wrist_data = data['signal'].get('wrist', {})
            if 'BVP' not in wrist_data:
                validation['issues'].append(f"Missing BVP data for subject {subject_id}")
                return validation
            
            bvp_data = wrist_data['BVP'].flatten()
            labels = data['label'].flatten()
            
            validation['bvp_samples'] = len(bvp_data)
            
            # Check data length consistency
            if len(bvp_data) != len(labels):
                validation['issues'].append(
                    f"Data length mismatch for subject {subject_id}: "
                    f"BVP={len(bvp_data)}, Labels={len(labels)}"
                )
                return validation
            
            # Check for minimum data length
            min_samples = 60 * self.sampling_rates['bvp']  # At least 1 minute
            if len(bvp_data) < min_samples:
                validation['warnings'].append(
                    f"Very short recording for subject {subject_id}: "
                    f"{len(bvp_data)/self.sampling_rates['bvp']:.1f} seconds"
                )
            
            # Check label validity
            valid_labels = set(self.labels.values())
            actual_labels = set(labels)
            invalid_labels = actual_labels - valid_labels
            
            if invalid_labels:
                validation['warnings'].append(
                    f"Unknown labels found for subject {subject_id}: {invalid_labels}"
                )
            
            validation['valid'] = True
            
        except Exception as e:
            validation['issues'].append(f"Error validating subject {subject_id}: {str(e)}")
        
        return validation

    def create_train_test_split(self, 
                              data: Dict, 
                              test_subjects: Optional[List[int]] = None,
                              test_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Create train/test split following Leave-One-Subject-Out (LOSO) methodology.
        
        Args:
            data: Loaded dataset
            test_subjects: Specific subjects for testing. If None, random selection
            test_ratio: Ratio of subjects for testing
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        subjects = list(data.keys())
        
        if test_subjects is None:
            # Random selection
            n_test = max(1, int(len(subjects) * test_ratio))
            test_subjects = np.random.choice(subjects, n_test, replace=False)
        
        train_data = {s: data[s] for s in subjects if s not in test_subjects}
        test_data = {s: data[s] for s in subjects if s in test_subjects}
        
        logger.info(f"Train/test split: {len(train_data)} training, {len(test_data)} testing subjects")
        
        return train_data, test_data