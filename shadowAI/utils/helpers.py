"""
Helper Utilities Module

This module provides comprehensive utility functions for the ShadowAI
stress detection pipeline, including logging setup, file operations,
data validation, performance monitoring, and common helper functions.

Features:
- Advanced logging configuration with rotation and filtering
- File and directory management utilities
- Data validation and sanitization functions
- Performance monitoring and profiling tools
- Memory and resource management utilities
- Error handling and debugging helpers
- Cross-platform compatibility functions

Author: Shadow AI Team
License: MIT
"""

import os
import sys
import logging
import logging.handlers
import time
import json
import yaml
import pickle
import hashlib
import platform
import psutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import warnings
from datetime import datetime, timedelta
import threading
import queue

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  console_enabled: bool = True,
                  file_enabled: bool = True,
                  max_file_size_mb: int = 10,
                  backup_count: int = 5,
                  log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. Auto-generated if None
        console_enabled: Enable console logging
        file_enabled: Enable file logging
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('shadowai')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_enabled:
        if log_file is None:
            # Auto-generate log file path
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"shadowai_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: level={log_level}, file={log_file}")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is unsupported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str, format: str = "auto") -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format (auto, yaml, json)
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format == "auto":
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            format = "yaml"
        elif config_path.suffix.lower() == '.json':
            format = "json"
        else:
            format = "yaml"  # Default to YAML
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if format == "yaml":
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format == "json":
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
    except Exception as e:
        raise ValueError(f"Failed to save configuration: {e}")


def ensure_directories(*directories: str) -> None:
    """
    Ensure directories exist, creating them if necessary.
    
    Args:
        *directories: Variable number of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Look for common project root indicators
    current_path = Path.cwd()
    
    indicators = [
        'setup.py',
        'pyproject.toml',
        'requirements.txt',
        'README.md',
        '.git'
    ]
    
    # Walk up the directory tree looking for indicators
    for parent in [current_path] + list(current_path.parents):
        if any((parent / indicator).exists() for indicator in indicators):
            return parent
    
    # Default to current working directory
    return current_path


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary containing system information
    """
    try:
        cpu_info = {
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'usage_percent': psutil.cpu_percent(interval=1),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        memory_info = psutil.virtual_memory()
        memory = {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'used_gb': memory_info.used / (1024**3),
            'usage_percent': memory_info.percent
        }
        
        disk_info = psutil.disk_usage('/')
        disk = {
            'total_gb': disk_info.total / (1024**3),
            'free_gb': disk_info.free / (1024**3),
            'used_gb': disk_info.used / (1024**3),
            'usage_percent': (disk_info.used / disk_info.total) * 100
        }
        
    except Exception as e:
        # Fallback for systems without psutil
        cpu_info = {'count': os.cpu_count(), 'error': str(e)}
        memory = {'error': str(e)}
        disk = {'error': str(e)}
    
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'architecture': platform.architecture(),
        'python_version': platform.python_version(),
        'cpu': cpu_info,
        'memory': memory,
        'disk': disk,
        'hostname': platform.node(),
        'timestamp': datetime.now().isoformat()
    }


def validate_data_types(data: Dict[str, Any], schema: Dict[str, type]) -> Tuple[bool, List[str]]:
    """
    Validate data types against a schema.
    
    Args:
        data: Data dictionary to validate
        schema: Schema dictionary with expected types
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    for key, expected_type in schema.items():
        if key not in data:
            errors.append(f"Missing required key: {key}")
            continue
        
        value = data[key]
        if not isinstance(value, expected_type):
            errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")
    
    return len(errors) == 0, errors


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for cross-platform compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def safe_pickle_load(file_path: str) -> Any:
    """
    Safely load pickle file with error handling.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object or None if failed
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load pickle file {file_path}: {e}")
        return None


def safe_pickle_save(obj: Any, file_path: str) -> bool:
    """
    Safely save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        logging.error(f"Failed to save pickle file {file_path}: {e}")
        return False


@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing operations.
    
    Args:
        description: Description of the operation being timed
        logger: Logger instance to use
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info(f"Starting {description}")
    
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Completed {description} in {elapsed_time:.2f} seconds")


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        
        # Start profiling
        tracemalloc.start()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Get profiling results
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Log results
            logger = logging.getLogger(__name__)
            logger.info(f"Function {func.__name__} profiling:")
            logger.info(f"  Execution time: {end_time - start_time:.4f} seconds")
            logger.info(f"  Current memory: {current / 1024 / 1024:.2f} MB")
            logger.info(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    return wrapper


class MemoryMonitor:
    """Memory usage monitoring utility."""
    
    def __init__(self, threshold_mb: float = 100.0):
        """
        Initialize memory monitor.
        
        Args:
            threshold_mb: Memory threshold in MB for warnings
        """
        self.threshold_mb = threshold_mb
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self, interval_seconds: float = 10.0) -> None:
        """
        Start continuous memory monitoring.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self._monitoring:
            return
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self._monitoring = False
        self.logger.info("Memory monitoring stopped")
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _monitor_loop(self, interval_seconds: float) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(interval_seconds):
            usage = self.get_current_usage()
            
            if 'error' not in usage:
                rss_mb = usage['rss_mb']
                
                if rss_mb > self.threshold_mb:
                    self.logger.warning(f"High memory usage: {rss_mb:.1f} MB")
                
                self.logger.debug(f"Memory usage: RSS={rss_mb:.1f}MB, "
                                f"VMS={usage['vms_mb']:.1f}MB, "
                                f"Percent={usage['percent']:.1f}%")


class ProgressReporter:
    """Advanced progress reporting utility."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress reporter.
        
        Args:
            total_items: Total number of items to process
            description: Description of the process
        """
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Starting {description}: {total_items} items")
    
    def update(self, increment: int = 1, message: Optional[str] = None) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items completed
            message: Optional status message
        """
        self.current_item += increment
        progress_percent = (self.current_item / self.total_items) * 100
        
        elapsed_time = time.time() - self.start_time
        if self.current_item > 0:
            estimated_total_time = elapsed_time * (self.total_items / self.current_item)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        status = f"{self.description}: {self.current_item}/{self.total_items} "
        status += f"({progress_percent:.1f}%) "
        status += f"ETA: {remaining_time:.0f}s"
        
        if message:
            status += f" - {message}"
        
        self.logger.info(status)
    
    def finish(self, message: Optional[str] = None) -> None:
        """
        Mark process as finished.
        
        Args:
            message: Optional completion message
        """
        total_time = time.time() - self.start_time
        final_message = f"Completed {self.description}: {self.total_items} items in {total_time:.1f}s"
        
        if message:
            final_message += f" - {message}"
        
        self.logger.info(final_message)


def retry_on_failure(max_retries: int = 3, 
                    delay_seconds: float = 1.0,
                    backoff_factor: float = 2.0,
                    exceptions: Tuple = (Exception,)) -> Callable:
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay_seconds
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger = logging.getLogger(__name__)
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            # Re-raise the last exception if all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def validate_file_integrity(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """
    Validate file integrity using hash comparison.
    
    Args:
        file_path: Path to file to validate
        expected_hash: Expected SHA256 hash (if None, just checks if file exists and is readable)
        
    Returns:
        True if file is valid, False otherwise
    """
    file_path = Path(file_path)
    
    # Check if file exists and is readable
    if not file_path.exists():
        logging.error(f"File does not exist: {file_path}")
        return False
    
    if not file_path.is_file():
        logging.error(f"Path is not a file: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            f.read(1)  # Try to read one byte
    except Exception as e:
        logging.error(f"File is not readable: {file_path}, error: {e}")
        return False
    
    # If no expected hash provided, basic validation passed
    if expected_hash is None:
        return True
    
    # Calculate and compare hash
    try:
        actual_hash = calculate_file_hash(str(file_path))
        
        if actual_hash.lower() == expected_hash.lower():
            return True
        else:
            logging.error(f"Hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to calculate hash for {file_path}: {e}")
        return False


def cleanup_temp_files(temp_dir: str = "/tmp", pattern: str = "shadowai_*", max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Temporary directory to clean
        pattern: File pattern to match
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of files cleaned up
    """
    import glob
    
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    files_cleaned = 0
    
    try:
        pattern_path = temp_path / pattern
        for file_path in glob.glob(str(pattern_path)):
            file_path = Path(file_path)
            
            if file_path.is_file():
                # Check file age
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        files_cleaned += 1
                        logging.debug(f"Cleaned up temp file: {file_path}")
                    except Exception as e:
                        logging.warning(f"Failed to delete temp file {file_path}: {e}")
        
        if files_cleaned > 0:
            logging.info(f"Cleaned up {files_cleaned} temporary files")
        
    except Exception as e:
        logging.error(f"Error during temp file cleanup: {e}")
    
    return files_cleaned


def get_available_memory_mb() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available memory in MB, or -1 if unable to determine
    """
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 / 1024
    except ImportError:
        return -1
    except Exception as e:
        logging.warning(f"Failed to get memory info: {e}")
        return -1


def estimate_memory_usage(data_size_mb: float, processing_factor: float = 3.0) -> float:
    """
    Estimate memory usage for data processing.
    
    Args:
        data_size_mb: Size of data in MB
        processing_factor: Multiplication factor for processing overhead
        
    Returns:
        Estimated memory usage in MB
    """
    return data_size_mb * processing_factor


def check_disk_space(path: str, required_mb: float) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check disk space for
        required_mb: Required space in MB
        
    Returns:
        True if sufficient space available
    """
    try:
        import shutil
        free_bytes = shutil.disk_usage(path).free
        free_mb = free_bytes / 1024 / 1024
        
        return free_mb >= required_mb
        
    except Exception as e:
        logging.warning(f"Failed to check disk space: {e}")
        return True  # Assume sufficient space if check fails


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


# Global instances for convenience
_memory_monitor = None

def start_global_memory_monitoring(threshold_mb: float = 100.0) -> None:
    """Start global memory monitoring."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor(threshold_mb)
        _memory_monitor.start_monitoring()


def stop_global_memory_monitoring() -> None:
    """Stop global memory monitoring."""
    global _memory_monitor
    if _memory_monitor is not None:
        _memory_monitor.stop_monitoring()
        _memory_monitor = None