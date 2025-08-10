"""
Helper utilities for Shadow AI models.
"""

import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def ensure_directories(base_path: str, directories: list):
    """
    Ensure required directories exist.
    
    Args:
        base_path: Base directory path
        directories: List of subdirectories to create
    """
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent

def get_model_paths() -> Dict[str, str]:
    """
    Get standard model-related paths.
    
    Returns:
        Dictionary of model paths
    """
    project_root = get_project_root()
    
    return {
        'models_dir': str(project_root / 'models'),
        'data_dir': str(project_root / 'models' / 'data'),
        'results_dir': str(project_root / 'models' / 'results'),
        'configs_dir': str(project_root / 'models' / 'configs'),
        'logs_dir': str(project_root / 'models' / 'results' / 'logs'),
        'plots_dir': str(project_root / 'models' / 'results' / 'plots'),
        'saved_models_dir': str(project_root / 'models' / 'results' / 'models'),
        'metrics_dir': str(project_root / 'models' / 'results' / 'metrics')
    }
