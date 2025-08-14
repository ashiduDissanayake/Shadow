"""
ShadowAI - Comprehensive TinyML Pipeline for Stress Detection

A modular, production-ready system for stress detection using BVP signals,
optimized for deployment on ESP32-S3 microcontrollers.

Author: Shadow AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Shadow AI Team"

# Core modules
from . import data
from . import models
from . import deployment
from . import simulation
from . import utils

__all__ = [
    'data',
    'models', 
    'deployment',
    'simulation',
    'utils'
]