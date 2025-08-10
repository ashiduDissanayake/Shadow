"""
WESAD Dataset Loader

This module handles loading and basic processing of the WESAD dataset for stress detection.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class WESADLoader:
    """
    Loader for the WESAD (Wearable Stress and Affect Detection) dataset.
    
    The WESAD dataset contains multimodal data from wrist-worn devices including:
    - BVP (Blood Volume Pulse)
    - EDA (Electrodermal Activity) 
    - TEMP (Temperature)
    - ACC (Accelerometer)
    """
    
    def __init__(self, data_path: str = "data/raw/wesad/"):
        """
        Initialize the WESAD loader.
        
        Args:
            data_path: Path to the WESAD dataset directory
        """
        self.data_path = data_path
        self.subjects = []
        self.labels = {
            'baseline': 0,
            'stress': 1, 
            'amusement': 2,
            'meditation': 3
        }
        
    def load_bvp_data(self, subjects: Optional[list] = None) -> Dict:
        """
        Load BVP data from WESAD dataset.
        
        Args:
            subjects: List of subject IDs to load. If None, loads all subjects.
            
        Returns:
            Dictionary containing BVP data and labels for each subject
        """
        if subjects is None:
            subjects = self._get_available_subjects()
            
        data = {}
        
        for subject in subjects:
            logger.info(f"Loading BVP data for subject {subject}")
            
            try:
                # Load BVP data
                bvp_path = os.path.join(self.data_path, f"S{subject}", "S{subject}_BVP.csv")
                if os.path.exists(bvp_path):
                    bvp_data = pd.read_csv(bvp_path)
                    
                    # Load labels
                    labels_path = os.path.join(self.data_path, f"S{subject}", "S{subject}_labels.csv")
                    if os.path.exists(labels_path):
                        labels = pd.read_csv(labels_path)
                        
                        data[subject] = {
                            'bvp': bvp_data.values.flatten(),
                            'labels': labels.values.flatten(),
                            'sampling_rate': 64  # WESAD BVP sampling rate
                        }
                        
                        logger.info(f"Successfully loaded subject {subject}: {len(bvp_data)} BVP samples")
                    else:
                        logger.warning(f"Labels file not found for subject {subject}")
                else:
                    logger.warning(f"BVP file not found for subject {subject}")
                    
            except Exception as e:
                logger.error(f"Error loading subject {subject}: {str(e)}")
                
        return data
    
    def load_multimodal_data(self, subjects: Optional[list] = None) -> Dict:
        """
        Load all modalities (BVP, EDA, TEMP, ACC) from WESAD dataset.
        
        Args:
            subjects: List of subject IDs to load. If None, loads all subjects.
            
        Returns:
            Dictionary containing all sensor data and labels for each subject
        """
        if subjects is None:
            subjects = self._get_available_subjects()
            
        data = {}
        modalities = ['BVP', 'EDA', 'TEMP', 'ACC']
        
        for subject in subjects:
            logger.info(f"Loading multimodal data for subject {subject}")
            
            subject_data = {}
            
            for modality in modalities:
                try:
                    file_path = os.path.join(self.data_path, f"S{subject}", f"S{subject}_{modality}.csv")
                    if os.path.exists(file_path):
                        modality_data = pd.read_csv(file_path)
                        subject_data[modality.lower()] = modality_data.values.flatten()
                    else:
                        logger.warning(f"{modality} file not found for subject {subject}")
                        
                except Exception as e:
                    logger.error(f"Error loading {modality} for subject {subject}: {str(e)}")
            
            # Load labels
            labels_path = os.path.join(self.data_path, f"S{subject}", "S{subject}_labels.csv")
            if os.path.exists(labels_path):
                labels = pd.read_csv(labels_path)
                subject_data['labels'] = labels.values.flatten()
                subject_data['sampling_rate'] = 64
                
                data[subject] = subject_data
                logger.info(f"Successfully loaded multimodal data for subject {subject}")
            else:
                logger.warning(f"Labels file not found for subject {subject}")
                
        return data
    
    def get_label_mapping(self) -> Dict:
        """
        Get the label mapping for WESAD dataset.
        
        Returns:
            Dictionary mapping label names to numeric values
        """
        return self.labels.copy()
    
    def get_subject_info(self) -> Dict:
        """
        Get information about available subjects.
        
        Returns:
            Dictionary containing subject information
        """
        subjects = self._get_available_subjects()
        
        info = {
            'total_subjects': len(subjects),
            'subjects': subjects,
            'available_modalities': ['bvp', 'eda', 'temp', 'acc'],
            'sampling_rate': 64,
            'label_mapping': self.labels
        }
        
        return info
    
    def _get_available_subjects(self) -> list:
        """
        Get list of available subject directories.
        
        Returns:
            List of subject IDs
        """
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return []
            
        subjects = []
        for item in os.listdir(self.data_path):
            if item.startswith('S') and os.path.isdir(os.path.join(self.data_path, item)):
                try:
                    subject_id = int(item[1:])  # Extract number from 'S1', 'S2', etc.
                    subjects.append(subject_id)
                except ValueError:
                    continue
                    
        return sorted(subjects)
    
    def validate_data(self, data: Dict) -> bool:
        """
        Validate loaded data for consistency.
        
        Args:
            data: Dictionary containing loaded data
            
        Returns:
            True if data is valid, False otherwise
        """
        if not data:
            logger.error("No data provided for validation")
            return False
            
        for subject, subject_data in data.items():
            # Check required keys
            required_keys = ['bvp', 'labels']
            for key in required_keys:
                if key not in subject_data:
                    logger.error(f"Missing required key '{key}' for subject {subject}")
                    return False
            
            # Check data lengths
            bvp_length = len(subject_data['bvp'])
            labels_length = len(subject_data['labels'])
            
            if bvp_length != labels_length:
                logger.error(f"Data length mismatch for subject {subject}: BVP={bvp_length}, Labels={labels_length}")
                return False
                
            # Check for valid labels
            valid_labels = set(self.labels.values())
            actual_labels = set(subject_data['labels'])
            
            if not actual_labels.issubset(valid_labels):
                logger.error(f"Invalid labels found for subject {subject}: {actual_labels - valid_labels}")
                return False
                
        logger.info("Data validation passed")
        return True
