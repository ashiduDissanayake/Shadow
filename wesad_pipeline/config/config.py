"""
Configuration settings for WESAD Analysis Pipeline

Centralized configuration for all parameters including WESAD dataset parameters,
analysis parameters, and output configurations.

Features:
- WESAD dataset parameters (sampling rates, label mappings, available subjects)
- Analysis parameters (window sizes, quality thresholds, heart rate ranges)
- Output paths and visualization settings
- Configuration validation and error handling

Author: Shadow AI Team
License: MIT
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class WESADDatasetConfig:
    """WESAD dataset specific configuration."""
    
    # Dataset path and file management
    wesad_path: str = "data/raw/wesad/"
    subjects: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17])
    
    # Sampling rates (Hz)
    bvp_sampling_rate: int = 64
    resp_sampling_rate: int = 700  # Chest sensor
    eda_sampling_rate: int = 4
    temp_sampling_rate: int = 4
    acc_sampling_rate: int = 32
    
    # Label mappings
    label_mapping: Dict[str, int] = field(default_factory=lambda: {
        'transient': 0,     # Transient periods between conditions
        'baseline': 1,      # Baseline condition
        'stress': 2,        # Stress condition (TSST)
        'amusement': 3,     # Amusement condition (funny videos)
        'meditation': 4     # Meditation/relaxation condition
    })
    
    # Target conditions for analysis
    target_conditions: List[str] = field(default_factory=lambda: ['baseline', 'stress', 'amusement'])

@dataclass 
class AnalysisConfig:
    """Analysis parameters configuration."""
    
    # Windowing parameters
    window_size_seconds: int = 60
    overlap_seconds: int = 5
    min_window_quality: float = 0.6
    
    # Signal quality parameters
    quality_threshold: float = 0.6
    enable_quality_assessment: bool = True
    
    # Heart rate parameters
    min_heart_rate: int = 40
    max_heart_rate: int = 200
    hr_smoothing_window: int = 5
    
    # Signal filtering
    filter_low_hz: float = 0.7
    filter_high_hz: float = 3.7
    filter_order: int = 3
    
    # Feature extraction
    enable_time_domain: bool = True
    enable_frequency_domain: bool = True
    enable_nonlinear_features: bool = False

@dataclass
class VisualizationConfig:
    """Visualization and plotting configuration."""
    
    # Plot settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    
    # Colors for conditions
    condition_colors: Dict[str, str] = field(default_factory=lambda: {
        'baseline': '#2E8B57',     # Sea green
        'stress': '#DC143C',       # Crimson
        'amusement': '#4169E1',    # Royal blue
        'meditation': '#9370DB',   # Medium purple
        'transient': '#808080'     # Gray
    })
    
    # Output settings
    save_plots: bool = True
    show_plots: bool = False
    plot_format: str = 'png'

@dataclass
class OutputConfig:
    """Output paths and file management configuration."""
    
    # Base output directory
    output_path: str = "wesad_analysis"
    
    # Subdirectories
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    data_dir: str = "processed_data"
    logs_dir: str = "logs"
    
    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    include_timestamp: bool = True
    
    # Export formats
    export_numpy: bool = True
    export_csv: bool = True
    export_json: bool = True

@dataclass
class WESADConfig:
    """Main configuration container for WESAD Analysis Pipeline."""
    
    # Component configurations
    dataset: WESADDatasetConfig = field(default_factory=WESADDatasetConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    enable_file_logging: bool = True
    
    # Processing configuration
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate paths
        if not os.path.exists(os.path.dirname(self.dataset.wesad_path)):
            logger.warning(f"WESAD path parent directory does not exist: {self.dataset.wesad_path}")
        
        # Validate subjects
        if not self.dataset.subjects:
            errors.append("At least one subject must be specified")
        
        if any(s < 1 or s > 17 for s in self.dataset.subjects):
            errors.append("Subject IDs must be between 1 and 17")
        
        # Validate sampling rates
        if self.dataset.bvp_sampling_rate <= 0:
            errors.append("BVP sampling rate must be positive")
        
        # Validate window parameters
        if self.analysis.window_size_seconds <= 0:
            errors.append("Window size must be positive")
        
        if self.analysis.overlap_seconds >= self.analysis.window_size_seconds:
            errors.append("Overlap must be less than window size")
        
        # Validate quality thresholds
        if not 0 <= self.analysis.quality_threshold <= 1:
            errors.append("Quality threshold must be between 0 and 1")
        
        if not 0 <= self.analysis.min_window_quality <= 1:
            errors.append("Minimum window quality must be between 0 and 1")
        
        # Validate heart rate parameters
        if self.analysis.min_heart_rate >= self.analysis.max_heart_rate:
            errors.append("Minimum heart rate must be less than maximum heart rate")
        
        if self.analysis.min_heart_rate <= 0:
            errors.append("Minimum heart rate must be positive")
        
        # Validate filter parameters
        if self.analysis.filter_low_hz >= self.analysis.filter_high_hz:
            errors.append("Low cutoff frequency must be less than high cutoff frequency")
        
        if self.analysis.filter_low_hz <= 0:
            errors.append("Filter frequencies must be positive")
        
        if self.analysis.filter_high_hz >= self.dataset.bvp_sampling_rate / 2:
            errors.append("High cutoff frequency must be less than Nyquist frequency")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            raise ValueError(f"Configuration validation failed with {len(errors)} errors")
        
        logger.info("Configuration validation successful")
        return True
    
    def create_output_directories(self) -> None:
        """Create all required output directories."""
        base_path = Path(self.output.output_path)
        
        directories = [
            base_path,
            base_path / self.output.plots_dir,
            base_path / self.output.reports_dir,
            base_path / self.output.data_dir,
            base_path / self.output.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def get_label_name(self, label_id: int) -> str:
        """Get label name from label ID."""
        reverse_mapping = {v: k for k, v in self.dataset.label_mapping.items()}
        return reverse_mapping.get(label_id, f"unknown_{label_id}")
    
    def get_label_id(self, label_name: str) -> int:
        """Get label ID from label name."""
        return self.dataset.label_mapping.get(label_name, 0)
    
    def get_condition_color(self, condition: str) -> str:
        """Get color for a specific condition."""
        return self.visualization.condition_colors.get(condition, '#808080')
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'dataset': {
                'wesad_path': self.dataset.wesad_path,
                'subjects': self.dataset.subjects,
                'bvp_sampling_rate': self.dataset.bvp_sampling_rate,
                'label_mapping': self.dataset.label_mapping,
                'target_conditions': self.dataset.target_conditions
            },
            'analysis': {
                'window_size_seconds': self.analysis.window_size_seconds,
                'overlap_seconds': self.analysis.overlap_seconds,
                'quality_threshold': self.analysis.quality_threshold,
                'min_heart_rate': self.analysis.min_heart_rate,
                'max_heart_rate': self.analysis.max_heart_rate,
                'filter_low_hz': self.analysis.filter_low_hz,
                'filter_high_hz': self.analysis.filter_high_hz
            },
            'visualization': {
                'figure_size': self.visualization.figure_size,
                'dpi': self.visualization.dpi,
                'condition_colors': self.visualization.condition_colors
            },
            'output': {
                'output_path': self.output.output_path,
                'save_plots': self.visualization.save_plots,
                'export_formats': {
                    'numpy': self.output.export_numpy,
                    'csv': self.output.export_csv,
                    'json': self.output.export_json
                }
            }
        }

def create_default_config(**kwargs) -> WESADConfig:
    """Create a default configuration with optional overrides."""
    config = WESADConfig()
    
    # Apply any override parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config

def load_config_from_dict(config_dict: Dict) -> WESADConfig:
    """Load configuration from dictionary."""
    config = WESADConfig()
    
    # Update dataset configuration
    if 'dataset' in config_dict:
        dataset_config = config_dict['dataset']
        for key, value in dataset_config.items():
            if hasattr(config.dataset, key):
                setattr(config.dataset, key, value)
    
    # Update analysis configuration
    if 'analysis' in config_dict:
        analysis_config = config_dict['analysis']
        for key, value in analysis_config.items():
            if hasattr(config.analysis, key):
                setattr(config.analysis, key, value)
    
    # Update visualization configuration
    if 'visualization' in config_dict:
        viz_config = config_dict['visualization']
        for key, value in viz_config.items():
            if hasattr(config.visualization, key):
                setattr(config.visualization, key, value)
    
    # Update output configuration
    if 'output' in config_dict:
        output_config = config_dict['output']
        for key, value in output_config.items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)
    
    config.validate()
    return config