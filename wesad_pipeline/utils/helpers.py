"""
Helper Utilities for WESAD Analysis Pipeline

Common utility functions, color mappings, statistical computations,
and file I/O helpers for the WESAD analysis pipeline.

Features:
- Safe label lookup and color mappings
- Statistical computation helpers
- File I/O utilities
- Progress tracking utilities
- Data validation helpers
- Common pipeline utilities

Author: Shadow AI Team
License: MIT
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import warnings
from collections import Counter

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WESADHelpers:
    """
    Helper utilities class for WESAD analysis pipeline.
    
    Provides common utility functions for data processing, validation,
    and pipeline operations.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the helpers.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def safe_label_lookup(self, label_id: Union[int, float], default: str = "unknown") -> str:
        """
        Safely look up condition name from label ID.
        
        Args:
            label_id: Label ID to look up
            default: Default value if lookup fails
            
        Returns:
            Condition name string
        """
        try:
            # Handle NaN or invalid values
            if pd.isna(label_id) or not isinstance(label_id, (int, float)):
                return default
            
            label_id = int(label_id)
            return self.config.get_label_name(label_id)
            
        except Exception:
            return default
    
    def get_condition_color(self, condition: str, alpha: float = 1.0) -> str:
        """
        Get color for a condition with optional alpha.
        
        Args:
            condition: Condition name
            alpha: Transparency level (0-1)
            
        Returns:
            Color string (hex or rgba)
        """
        base_color = self.config.get_condition_color(condition)
        
        if alpha < 1.0:
            # Convert hex to rgba
            if base_color.startswith('#'):
                hex_color = base_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
        
        return base_color
    
    def validate_array_data(self, data: np.ndarray, name: str = "data",
                          min_length: int = 1, check_finite: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate numpy array data.
        
        Args:
            data: Array to validate
            name: Name for error messages
            min_length: Minimum required length
            check_finite: Whether to check for finite values
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if it's an array
        if not isinstance(data, np.ndarray):
            issues.append(f"{name} is not a numpy array")
            return False, issues
        
        # Check length
        if len(data) < min_length:
            issues.append(f"{name} length ({len(data)}) is below minimum ({min_length})")
        
        # Check for empty array
        if data.size == 0:
            issues.append(f"{name} is empty")
        
        # Check for finite values
        if check_finite and data.size > 0:
            if not np.all(np.isfinite(data)):
                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                issues.append(f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf")
        
        return len(issues) == 0, issues
    
    def calculate_basic_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistical measures for data.
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary containing statistical measures
        """
        if len(data) == 0:
            return {}
        
        try:
            # Remove non-finite values for statistics
            clean_data = data[np.isfinite(data)]
            
            if len(clean_data) == 0:
                return {'valid_samples': 0}
            
            stats = {
                'count': len(data),
                'valid_samples': len(clean_data),
                'mean': float(np.mean(clean_data)),
                'median': float(np.median(clean_data)),
                'std': float(np.std(clean_data)),
                'var': float(np.var(clean_data)),
                'min': float(np.min(clean_data)),
                'max': float(np.max(clean_data)),
                'range': float(np.ptp(clean_data)),
                'q25': float(np.percentile(clean_data, 25)),
                'q75': float(np.percentile(clean_data, 75)),
                'iqr': float(np.percentile(clean_data, 75) - np.percentile(clean_data, 25))
            }
            
            # Additional statistics
            if len(clean_data) > 1:
                stats['skewness'] = self._calculate_skewness(clean_data)
                stats['kurtosis'] = self._calculate_kurtosis(clean_data)
                stats['cv'] = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else np.inf
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {str(e)}")
            return {}
    
    def save_array_data(self, data: Union[np.ndarray, Dict], filepath: Union[str, Path],
                       format: str = 'npy') -> bool:
        """
        Save array data to file.
        
        Args:
            data: Data to save
            filepath: Output file path
            format: Output format ('npy', 'npz', 'csv', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'npy':
                if isinstance(data, dict):
                    self.logger.warning("Cannot save dict as .npy, using .npz instead")
                    np.savez(filepath.with_suffix('.npz'), **data)
                else:
                    np.save(filepath, data)
            
            elif format == 'npz':
                if isinstance(data, dict):
                    np.savez(filepath, **data)
                else:
                    np.savez(filepath, data=data)
            
            elif format == 'csv':
                if isinstance(data, dict):
                    # Save as multiple CSV files or combined DataFrame
                    df_data = {}
                    for key, value in data.items():
                        if isinstance(value, np.ndarray) and value.ndim == 1:
                            df_data[key] = value
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.to_csv(filepath.with_suffix('.csv'), index=False)
                    else:
                        self.logger.warning("No suitable data for CSV format")
                        return False
                else:
                    if data.ndim == 1:
                        df = pd.DataFrame({'data': data})
                        df.to_csv(filepath, index=False)
                    elif data.ndim == 2:
                        df = pd.DataFrame(data)
                        df.to_csv(filepath, index=False)
                    else:
                        self.logger.warning("Cannot save multidimensional array as CSV")
                        return False
            
            elif format == 'json':
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._convert_for_json(data)
                with open(filepath.with_suffix('.json'), 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False
            
            self.logger.debug(f"Saved data to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            return False
    
    def load_array_data(self, filepath: Union[str, Path], format: str = 'auto') -> Optional[Union[np.ndarray, Dict]]:
        """
        Load array data from file.
        
        Args:
            filepath: Input file path
            format: Input format ('auto', 'npy', 'npz', 'csv', 'json')
            
        Returns:
            Loaded data or None if failed
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return None
            
            # Auto-detect format from extension
            if format == 'auto':
                suffix = filepath.suffix.lower()
                format_map = {'.npy': 'npy', '.npz': 'npz', '.csv': 'csv', '.json': 'json'}
                format = format_map.get(suffix, 'npy')
            
            if format == 'npy':
                return np.load(filepath)
            
            elif format == 'npz':
                npz_data = np.load(filepath)
                return dict(npz_data)
            
            elif format == 'csv':
                df = pd.read_csv(filepath)
                return df.to_numpy()
            
            elif format == 'json':
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                return self._convert_from_json(json_data)
            
            else:
                self.logger.error(f"Unsupported format: {format}")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            return None
    
    def create_timestamp_string(self, include_milliseconds: bool = False) -> str:
        """
        Create formatted timestamp string.
        
        Args:
            include_milliseconds: Whether to include milliseconds
            
        Returns:
            Formatted timestamp string
        """
        now = datetime.now()
        if include_milliseconds:
            return now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits from microseconds
        else:
            return now.strftime(self.config.output.timestamp_format)
    
    def ensure_output_directory(self, subdirectory: Optional[str] = None) -> Path:
        """
        Ensure output directory exists.
        
        Args:
            subdirectory: Optional subdirectory name
            
        Returns:
            Path to the output directory
        """
        base_path = Path(self.config.output.output_path)
        
        if subdirectory:
            output_path = base_path / subdirectory
        else:
            output_path = base_path
        
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def filter_by_quality(self, data: List[Dict], quality_key: str = 'quality',
                         threshold: Optional[float] = None) -> List[Dict]:
        """
        Filter data by quality threshold.
        
        Args:
            data: List of data dictionaries
            quality_key: Key containing quality scores
            threshold: Quality threshold (uses config if None)
            
        Returns:
            Filtered data list
        """
        if threshold is None:
            threshold = self.config.analysis.quality_threshold
        
        filtered_data = []
        for item in data:
            if quality_key in item and item[quality_key] >= threshold:
                filtered_data.append(item)
        
        return filtered_data
    
    def group_by_condition(self, data: List[Dict], label_key: str = 'label') -> Dict[str, List[Dict]]:
        """
        Group data by condition labels.
        
        Args:
            data: List of data dictionaries
            label_key: Key containing label IDs
            
        Returns:
            Dictionary mapping condition names to data lists
        """
        grouped_data = {}
        
        for item in data:
            if label_key in item:
                label_id = item[label_key]
                condition_name = self.safe_label_lookup(label_id)
                
                if condition_name not in grouped_data:
                    grouped_data[condition_name] = []
                
                grouped_data[condition_name].append(item)
        
        return grouped_data
    
    def calculate_condition_statistics(self, grouped_data: Dict[str, List[Dict]],
                                     value_key: str = 'quality') -> Dict[str, Dict]:
        """
        Calculate statistics for each condition group.
        
        Args:
            grouped_data: Data grouped by condition
            value_key: Key containing values to analyze
            
        Returns:
            Dictionary mapping condition names to statistics
        """
        condition_stats = {}
        
        for condition_name, condition_data in grouped_data.items():
            values = []
            for item in condition_data:
                if value_key in item:
                    values.append(item[value_key])
            
            if values:
                condition_stats[condition_name] = self.calculate_basic_statistics(np.array(values))
                condition_stats[condition_name]['count'] = len(condition_data)
            else:
                condition_stats[condition_name] = {'count': 0}
        
        return condition_stats
    
    def validate_pipeline_data(self, data: Dict, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate pipeline data structure.
        
        Args:
            data: Data dictionary to validate
            required_keys: List of required keys
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required keys
        for key in required_keys:
            if key not in data:
                issues.append(f"Missing required key: {key}")
        
        # Check data types and content
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                is_valid, array_issues = self.validate_array_data(value, key)
                if not is_valid:
                    issues.extend(array_issues)
            elif isinstance(value, list) and len(value) == 0:
                issues.append(f"Empty list for key: {key}")
        
        return len(issues) == 0, issues
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _convert_for_json(self, data: Any) -> Any:
        """Convert data for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._convert_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    def _convert_from_json(self, data: Any) -> Any:
        """Convert data from JSON format."""
        if isinstance(data, dict):
            converted = {}
            for key, value in data.items():
                if isinstance(value, list):
                    # Try to convert lists to numpy arrays
                    try:
                        converted[key] = np.array(value)
                    except:
                        converted[key] = value
                else:
                    converted[key] = self._convert_from_json(value)
            return converted
        elif isinstance(data, list):
            try:
                # Try to convert to numpy array
                return np.array(data)
            except:
                return [self._convert_from_json(item) for item in data]
        else:
            return data