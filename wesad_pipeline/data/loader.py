"""
WESAD Data Loading Module

Handles WESAD dataset loading with comprehensive error handling and validation.
Reuses and extends functionality from the existing shadowAI.data.wesad_loader.

Features:
- Load WESAD pickle files
- Handle subject validation and missing files
- Extract BVP signals and labels
- Robust error handling with detailed logging
- Progress tracking for batch operations

Author: Shadow AI Team
License: MIT
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm

# Import existing functionality from shadowAI
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from shadowAI.data import WESADLoader
from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WESADDataLoader:
    """
    WESAD dataset loader for the analysis pipeline.
    
    Extends the existing shadowAI WESADLoader with pipeline-specific functionality
    and configuration integration.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the WESAD data loader.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the underlying WESAD loader
        self.loader = WESADLoader(
            data_path=config.dataset.wesad_path,
            cache_enabled=config.enable_caching,
            validate_on_load=True,
            auto_extract=True
        )
        
        # Statistics tracking
        self.stats = {
            'subjects_loaded': 0,
            'subjects_failed': 0,
            'total_signals': 0,
            'total_samples': 0,
            'conditions_found': set()
        }
        
        self.logger.info(f"WESAD data loader initialized with {len(config.dataset.subjects)} subjects")
    
    def validate_subjects(self, subjects: Optional[List[int]] = None) -> List[int]:
        """
        Validate and filter available subjects.
        
        Args:
            subjects: List of subject IDs to validate. If None, uses config subjects.
            
        Returns:
            List of validated subject IDs
        """
        if subjects is None:
            subjects = self.config.dataset.subjects
        
        # Get available subjects from the loader
        dataset_stats = self.loader.get_dataset_statistics()
        available_subjects = dataset_stats.get('subject_ids', [])
        
        # Filter to only available subjects
        valid_subjects = [s for s in subjects if s in available_subjects]
        missing_subjects = [s for s in subjects if s not in available_subjects]
        
        if missing_subjects:
            self.logger.warning(f"Missing subjects: {missing_subjects}")
        
        if not valid_subjects:
            self.logger.error("No valid subjects found")
            raise ValueError("No valid subjects available for loading")
        
        self.logger.info(f"Validated {len(valid_subjects)} subjects: {valid_subjects}")
        return valid_subjects
    
    def load_subject_data(self, subject_id: int) -> Optional[Dict]:
        """
        Load data for a single subject.
        
        Args:
            subject_id: Subject ID to load
            
        Returns:
            Dictionary containing subject data or None if loading failed
        """
        try:
            self.logger.debug(f"Loading subject {subject_id}")
            
            # Load BVP data using the existing loader
            bvp_data = self.loader.load_bvp_data(
                subjects=[subject_id],
                conditions=self.config.dataset.target_conditions,
                quality_threshold=self.config.analysis.quality_threshold
            )
            
            if subject_id not in bvp_data:
                self.logger.warning(f"No data loaded for subject {subject_id}")
                return None
            
            subject_data = bvp_data[subject_id]
            
            # Validate loaded data
            if not self._validate_subject_data(subject_data):
                self.logger.warning(f"Data validation failed for subject {subject_id}")
                return None
            
            # Update statistics
            self.stats['subjects_loaded'] += 1
            self.stats['total_signals'] += 1
            self.stats['total_samples'] += len(subject_data.get('bvp', []))
            
            # Track conditions found
            if 'labels' in subject_data:
                unique_labels = np.unique(subject_data['labels'])
                for label in unique_labels:
                    condition_name = self.config.get_label_name(label)
                    self.stats['conditions_found'].add(condition_name)
            
            self.logger.debug(f"Successfully loaded subject {subject_id}")
            return subject_data
            
        except Exception as e:
            self.logger.error(f"Failed to load subject {subject_id}: {str(e)}")
            self.stats['subjects_failed'] += 1
            return None
    
    def load_multiple_subjects(self, subjects: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        Load data for multiple subjects with progress tracking.
        
        Args:
            subjects: List of subject IDs to load. If None, uses config subjects.
            
        Returns:
            Dictionary mapping subject IDs to their data
        """
        subjects = self.validate_subjects(subjects)
        loaded_data = {}
        
        self.logger.info(f"Loading data for {len(subjects)} subjects")
        
        # Load subjects with progress bar
        with tqdm(subjects, desc="Loading subjects") as pbar:
            for subject_id in pbar:
                pbar.set_description(f"Loading subject {subject_id}")
                
                subject_data = self.load_subject_data(subject_id)
                if subject_data is not None:
                    loaded_data[subject_id] = subject_data
                
                # Update progress bar with current stats
                pbar.set_postfix({
                    'loaded': self.stats['subjects_loaded'],
                    'failed': self.stats['subjects_failed']
                })
        
        self.logger.info(f"Loaded {len(loaded_data)} subjects successfully")
        return loaded_data
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        # Get basic stats from the underlying loader
        base_stats = self.loader.get_dataset_statistics()
        
        # Add pipeline-specific statistics
        pipeline_stats = {
            'pipeline_stats': self.stats.copy(),
            'config_subjects': len(self.config.dataset.subjects),
            'target_conditions': self.config.dataset.target_conditions,
            'sampling_rate': self.config.dataset.bvp_sampling_rate,
            'window_size_seconds': self.config.analysis.window_size_seconds,
            'overlap_seconds': self.config.analysis.overlap_seconds
        }
        
        # Convert set to list for JSON serialization
        pipeline_stats['pipeline_stats']['conditions_found'] = list(self.stats['conditions_found'])
        
        # Combine statistics
        combined_stats = {**base_stats, **pipeline_stats}
        
        return combined_stats
    
    def _validate_subject_data(self, subject_data: Dict) -> bool:
        """
        Validate loaded subject data.
        
        Args:
            subject_data: Subject data dictionary
            
        Returns:
            True if data is valid, False otherwise
        """
        required_keys = ['bvp', 'labels', 'quality_score']
        
        # Check required keys
        for key in required_keys:
            if key not in subject_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        # Check data types and shapes
        bvp = subject_data['bvp']
        labels = subject_data['labels']
        
        if not isinstance(bvp, np.ndarray):
            self.logger.error("BVP data must be numpy array")
            return False
        
        if not isinstance(labels, np.ndarray):
            self.logger.error("Labels data must be numpy array")
            return False
        
        if len(bvp) == 0:
            self.logger.error("BVP data is empty")
            return False
        
        if len(labels) == 0:
            self.logger.error("Labels data is empty")
            return False
        
        # Check quality score
        quality_score = subject_data.get('quality_score', 0)
        if quality_score < self.config.analysis.quality_threshold:
            self.logger.warning(f"Quality score {quality_score:.3f} below threshold {self.config.analysis.quality_threshold}")
            return False
        
        # Check sampling rate consistency (approximate)
        expected_samples = len(labels) * (self.config.dataset.bvp_sampling_rate / self.config.dataset.resp_sampling_rate)
        actual_samples = len(bvp)
        
        # Allow some tolerance for sampling rate differences
        if abs(actual_samples - expected_samples) > expected_samples * 0.1:
            self.logger.warning(f"Sample count mismatch: expected ~{expected_samples:.0f}, got {actual_samples}")
        
        return True
    
    def reset_statistics(self) -> None:
        """Reset loading statistics."""
        self.stats = {
            'subjects_loaded': 0,
            'subjects_failed': 0,
            'total_signals': 0,
            'total_samples': 0,
            'conditions_found': set()
        }
        self.logger.debug("Statistics reset")
    
    def get_subject_file_path(self, subject_id: int) -> Path:
        """
        Get the file path for a specific subject.
        
        Args:
            subject_id: Subject ID
            
        Returns:
            Path to the subject's pickle file
        """
        return Path(self.config.dataset.wesad_path) / f"S{subject_id}" / f"S{subject_id}.pkl"
    
    def check_subject_availability(self, subject_id: int) -> bool:
        """
        Check if a subject's data file is available.
        
        Args:
            subject_id: Subject ID to check
            
        Returns:
            True if subject data is available, False otherwise
        """
        file_path = self.get_subject_file_path(subject_id)
        return file_path.exists()
    
    def get_available_subjects(self) -> List[int]:
        """
        Get list of all available subjects in the dataset.
        
        Returns:
            List of available subject IDs
        """
        dataset_stats = self.loader.get_dataset_statistics()
        return dataset_stats.get('subject_ids', [])