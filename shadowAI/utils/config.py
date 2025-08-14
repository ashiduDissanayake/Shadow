"""
Configuration Management Module

This module provides comprehensive configuration management for the ShadowAI
stress detection pipeline, including default configurations, validation,
environment-specific settings, and configuration persistence.

Features:
- Hierarchical configuration management
- Environment-specific configuration loading
- Configuration validation and schema enforcement
- Configuration persistence and loading
- Dynamic configuration updates
- Configuration versioning and migration

Author: Shadow AI Team
License: MIT
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data processing configuration."""
    # Dataset configuration
    wesad_path: str = "data/raw/wesad/"
    cache_enabled: bool = True
    cache_path: str = "data/cache/"
    
    # BVP processing
    sampling_rate: int = 64
    window_size_seconds: int = 60
    overlap_seconds: int = 5
    filter_low: float = 0.7
    filter_high: float = 3.7
    filter_order: int = 3
    
    # Quality assessment
    quality_threshold: float = 0.6
    min_heart_rate: int = 40
    max_heart_rate: int = 200
    
    # Normalization
    normalization_method: str = "zscore"  # zscore, minmax, robust
    fit_on_train: bool = True

@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    # Architecture
    bvp_input_shape: List[int] = field(default_factory=lambda: [3840, 1])
    hrv_input_shape: List[int] = field(default_factory=lambda: [20])
    num_classes: int = 4
    
    # CNN layers
    cnn_filters: List[int] = field(default_factory=lambda: [16, 32, 64, 32])
    cnn_kernels: List[int] = field(default_factory=lambda: [16, 8, 4, 4])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    cnn_pools: List[int] = field(default_factory=lambda: [4, 4, 4, 2])
    
    # Dense layers
    dense_units: List[int] = field(default_factory=lambda: [32, 16])
    combined_units: List[int] = field(default_factory=lambda: [64, 32])
    
    # Regularization
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2

@dataclass
class QATConfig:
    """Quantization-Aware Training configuration."""
    # Quantization settings
    quantization_type: str = "int8"  # int8, float16, dynamic
    optimization_level: str = "default"  # default, size, latency
    representative_dataset_size: int = 1000
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    quantize_delay: int = 5
    
    # Validation
    min_accuracy_retention: float = 0.95
    target_device: str = "esp32_s3"
    memory_limit_mb: float = 8.0
    inference_time_limit_ms: float = 100.0

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    # Target platform
    target_platform: str = "esp32-s3"
    board_type: str = "ESP32-S3-DevKitC-1"
    
    # TFLite conversion
    tflite_optimization: bool = True
    tflite_quantization: str = "int8"
    enable_optimizations: bool = True
    
    # ESP32 specific
    flash_size: str = "8MB"
    spiram_size: str = "8MB"
    cpu_frequency: str = "240MHz"
    
    # Sensor configuration
    sensor_type: str = "MAX30102"
    i2c_sda_pin: int = 21
    i2c_scl_pin: int = 22
    i2c_frequency: int = 400000
    
    # Communication
    enable_bluetooth: bool = True
    enable_wifi: bool = False
    wifi_ssid: str = ""
    wifi_password: str = ""
    
    # Power management
    enable_deep_sleep: bool = True
    sleep_duration_seconds: int = 300
    battery_monitoring: bool = True

@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    # Cross-validation
    cv_strategy: str = "loso"  # loso, kfold, stratified
    n_folds: int = 5
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "auc_roc"
    ])
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Performance thresholds
    min_accuracy: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_f1: float = 0.75
    
    # Deployment requirements
    max_inference_time_ms: float = 100.0
    max_model_size_mb: float = 8.0
    min_deployment_score: float = 0.8

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/shadowai.log"
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class ShadowAIConfig:
    """Main ShadowAI configuration container."""
    # Configuration metadata
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: str = "development"  # development, staging, production
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    qat: QATConfig = field(default_factory=QATConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paths
    project_root: str = field(default_factory=lambda: os.getcwd())
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    results_dir: str = "results"


class Config:
    """
    Comprehensive configuration manager for ShadowAI pipeline.
    
    Provides hierarchical configuration management with validation,
    environment-specific loading, and persistence capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. Uses default if None.
        """
        self.config_path = config_path
        self._config = ShadowAIConfig()
        self._config_schema = None
        self._environment_configs = {}
        
        # Load configuration if path provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        logger.info(f"Configuration manager initialized for environment: {self._config.environment}")
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            self.config_path = str(config_path)
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration. Uses current path if None.
        """
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None:
            raise ValueError("No configuration path specified")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert configuration to dictionary
        config_dict = asdict(self._config)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> ShadowAIConfig:
        """Get complete configuration object."""
        return self._config
    
    def get_data_config(self) -> DataConfig:
        """Get data processing configuration."""
        return self._config.data
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self._config.model
    
    def get_qat_config(self) -> QATConfig:
        """Get QAT configuration."""
        return self._config.qat
    
    def get_deployment_config(self) -> DeploymentConfig:
        """Get deployment configuration."""
        return self._config.deployment
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        return self._config.evaluation
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self._config.logging
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                if isinstance(getattr(self._config, key), (DataConfig, ModelConfig, QATConfig, 
                                                         DeploymentConfig, EvaluationConfig, LoggingConfig)):
                    # Update nested configuration
                    nested_config = getattr(self._config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                        else:
                            logger.warning(f"Unknown nested configuration parameter: {key}.{nested_key}")
                else:
                    setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def set_environment(self, environment: str) -> None:
        """
        Set configuration environment.
        
        Args:
            environment: Environment name (development, staging, production)
        """
        valid_environments = ["development", "staging", "production"]
        
        if environment not in valid_environments:
            raise ValueError(f"Invalid environment. Must be one of: {valid_environments}")
        
        self._config.environment = environment
        
        # Load environment-specific configuration if available
        self._load_environment_config(environment)
        
        logger.info(f"Environment set to: {environment}")
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate configuration parameters.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Validate data configuration
        data_validation = self._validate_data_config()
        validation_results["errors"].extend(data_validation.get("errors", []))
        validation_results["warnings"].extend(data_validation.get("warnings", []))
        
        # Validate model configuration
        model_validation = self._validate_model_config()
        validation_results["errors"].extend(model_validation.get("errors", []))
        validation_results["warnings"].extend(model_validation.get("warnings", []))
        
        # Validate deployment configuration
        deployment_validation = self._validate_deployment_config()
        validation_results["errors"].extend(deployment_validation.get("errors", []))
        validation_results["warnings"].extend(deployment_validation.get("warnings", []))
        
        # Log validation results
        if validation_results["errors"]:
            logger.error(f"Configuration validation failed: {len(validation_results['errors'])} errors")
        elif validation_results["warnings"]:
            logger.warning(f"Configuration validation completed with {len(validation_results['warnings'])} warnings")
        else:
            logger.info("Configuration validation passed")
        
        return validation_results
    
    def create_environment_config(self, environment: str, overrides: Dict[str, Any]) -> None:
        """
        Create environment-specific configuration.
        
        Args:
            environment: Environment name
            overrides: Configuration overrides for the environment
        """
        self._environment_configs[environment] = overrides
        logger.info(f"Environment configuration created for: {environment}")
    
    def get_paths(self) -> Dict[str, str]:
        """Get all configured paths."""
        base_path = Path(self._config.project_root)
        
        return {
            "project_root": str(base_path),
            "data_dir": str(base_path / self._config.data_dir),
            "models_dir": str(base_path / self._config.models_dir),
            "logs_dir": str(base_path / self._config.logs_dir),
            "results_dir": str(base_path / self._config.results_dir),
            "wesad_path": str(base_path / self._config.data.wesad_path),
            "cache_path": str(base_path / self._config.data.cache_path)
        }
    
    def ensure_directories(self) -> None:
        """Create necessary directories based on configuration."""
        paths = self.get_paths()
        
        for path_name, path_value in paths.items():
            if path_name != "project_root":  # Don't create project root
                Path(path_value).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {path_value}")
    
    def export_config_for_deployment(self, target_format: str = "json") -> str:
        """
        Export configuration in deployment-ready format.
        
        Args:
            target_format: Export format (json, yaml, cpp_header)
            
        Returns:
            Formatted configuration string
        """
        deployment_config = {
            "sampling_rate": self._config.data.sampling_rate,
            "window_size_seconds": self._config.data.window_size_seconds,
            "num_classes": self._config.model.num_classes,
            "sensor_pins": {
                "sda": self._config.deployment.i2c_sda_pin,
                "scl": self._config.deployment.i2c_scl_pin
            },
            "communication": {
                "bluetooth": self._config.deployment.enable_bluetooth,
                "wifi": self._config.deployment.enable_wifi
            },
            "power_management": {
                "deep_sleep": self._config.deployment.enable_deep_sleep,
                "sleep_duration": self._config.deployment.sleep_duration_seconds
            }
        }
        
        if target_format == "json":
            return json.dumps(deployment_config, indent=2)
        elif target_format == "yaml":
            return yaml.dump(deployment_config, default_flow_style=False, indent=2)
        elif target_format == "cpp_header":
            return self._generate_cpp_config_header(deployment_config)
        else:
            raise ValueError(f"Unsupported export format: {target_format}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        def update_dataclass_from_dict(obj, data):
            for key, value in data.items():
                if hasattr(obj, key):
                    current_value = getattr(obj, key)
                    if isinstance(current_value, (DataConfig, ModelConfig, QATConfig, 
                                                DeploymentConfig, EvaluationConfig, LoggingConfig)):
                        # Recursively update nested dataclass
                        if isinstance(value, dict):
                            update_dataclass_from_dict(current_value, value)
                    else:
                        setattr(obj, key, value)
        
        update_dataclass_from_dict(self._config, config_dict)
    
    def _load_environment_config(self, environment: str) -> None:
        """Load environment-specific configuration overrides."""
        if environment in self._environment_configs:
            overrides = self._environment_configs[environment]
            self._update_config_from_dict(overrides)
            logger.info(f"Applied environment-specific configuration for: {environment}")
    
    def _validate_data_config(self) -> Dict[str, List[str]]:
        """Validate data configuration parameters."""
        results = {"errors": [], "warnings": []}
        
        # Check sampling rate
        if self._config.data.sampling_rate <= 0:
            results["errors"].append("Sampling rate must be positive")
        elif self._config.data.sampling_rate < 32:
            results["warnings"].append("Low sampling rate may affect signal quality")
        
        # Check window size
        if self._config.data.window_size_seconds <= 0:
            results["errors"].append("Window size must be positive")
        elif self._config.data.window_size_seconds < 30:
            results["warnings"].append("Short window size may reduce accuracy")
        
        # Check filter parameters
        if self._config.data.filter_low >= self._config.data.filter_high:
            results["errors"].append("Filter low frequency must be less than high frequency")
        
        # Check thresholds
        if not 0 <= self._config.data.quality_threshold <= 1:
            results["errors"].append("Quality threshold must be between 0 and 1")
        
        return results
    
    def _validate_model_config(self) -> Dict[str, List[str]]:
        """Validate model configuration parameters."""
        results = {"errors": [], "warnings": []}
        
        # Check input shapes
        if len(self._config.model.bvp_input_shape) != 2:
            results["errors"].append("BVP input shape must be 2D")
        
        # Check layer configurations
        layer_configs = [
            self._config.model.cnn_filters,
            self._config.model.cnn_kernels,
            self._config.model.cnn_strides,
            self._config.model.cnn_pools
        ]
        
        if not all(len(config) == len(layer_configs[0]) for config in layer_configs):
            results["errors"].append("All CNN layer configurations must have same length")
        
        # Check training parameters
        if self._config.model.learning_rate <= 0:
            results["errors"].append("Learning rate must be positive")
        
        if self._config.model.batch_size <= 0:
            results["errors"].append("Batch size must be positive")
        
        return results
    
    def _validate_deployment_config(self) -> Dict[str, List[str]]:
        """Validate deployment configuration parameters."""
        results = {"errors": [], "warnings": []}
        
        # Check GPIO pins
        valid_gpio_range = range(0, 48)  # ESP32-S3 GPIO range
        
        if self._config.deployment.i2c_sda_pin not in valid_gpio_range:
            results["errors"].append(f"Invalid SDA pin: {self._config.deployment.i2c_sda_pin}")
        
        if self._config.deployment.i2c_scl_pin not in valid_gpio_range:
            results["errors"].append(f"Invalid SCL pin: {self._config.deployment.i2c_scl_pin}")
        
        if self._config.deployment.i2c_sda_pin == self._config.deployment.i2c_scl_pin:
            results["errors"].append("SDA and SCL pins cannot be the same")
        
        # Check I2C frequency
        valid_i2c_freqs = [100000, 400000, 1000000]  # Standard I2C frequencies
        if self._config.deployment.i2c_frequency not in valid_i2c_freqs:
            results["warnings"].append(f"Non-standard I2C frequency: {self._config.deployment.i2c_frequency}")
        
        return results
    
    def _generate_cpp_config_header(self, config_dict: Dict[str, Any]) -> str:
        """Generate C++ header file from configuration."""
        header_content = """/*
 * ShadowAI Configuration Header
 * Auto-generated from Python configuration
 */

#ifndef SHADOWAI_CONFIG_H
#define SHADOWAI_CONFIG_H

"""
        
        # Convert configuration to C++ defines
        def dict_to_defines(d, prefix=""):
            defines = ""
            for key, value in d.items():
                define_name = f"{prefix}{key.upper()}"
                
                if isinstance(value, dict):
                    defines += dict_to_defines(value, f"{define_name}_")
                elif isinstance(value, bool):
                    defines += f"#define {define_name} {int(value)}\\n"
                elif isinstance(value, (int, float)):
                    defines += f"#define {define_name} {value}\\n"
                elif isinstance(value, str):
                    defines += f'#define {define_name} "{value}"\\n'
            
            return defines
        
        header_content += dict_to_defines(config_dict, "SHADOWAI_")
        header_content += "\\n#endif // SHADOWAI_CONFIG_H\\n"
        
        return header_content


def load_default_config() -> Config:
    """Load default ShadowAI configuration."""
    return Config()


def load_config_from_file(config_path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Config instance
    """
    return Config(config_path)


def create_development_config() -> Config:
    """Create configuration optimized for development."""
    config = Config()
    
    # Development-specific overrides
    config.update_config(
        environment="development",
        logging={
            "level": "DEBUG",
            "console_enabled": True,
            "file_enabled": True
        },
        data={
            "cache_enabled": True,
            "quality_threshold": 0.5  # More lenient for development
        },
        model={
            "epochs": 10,  # Faster training for development
            "batch_size": 16
        }
    )
    
    return config


def create_production_config() -> Config:
    """Create configuration optimized for production."""
    config = Config()
    
    # Production-specific overrides
    config.update_config(
        environment="production",
        logging={
            "level": "INFO",
            "console_enabled": False,
            "file_enabled": True
        },
        data={
            "cache_enabled": True,
            "quality_threshold": 0.7  # Stricter for production
        },
        qat={
            "min_accuracy_retention": 0.95,
            "target_device": "esp32_s3"
        }
    )
    
    return config


def validate_environment_config(config: Config) -> bool:
    """
    Validate configuration for current environment.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    validation_results = config.validate_config()
    
    if validation_results["errors"]:
        logger.error("Configuration validation failed:")
        for error in validation_results["errors"]:
            logger.error(f"  - {error}")
        return False
    
    if validation_results["warnings"]:
        logger.warning("Configuration validation warnings:")
        for warning in validation_results["warnings"]:
            logger.warning(f"  - {warning}")
    
    return True