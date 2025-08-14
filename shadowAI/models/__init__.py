"""
Models Module

This module contains the CNN architecture with QAT support,
training utilities, and model evaluation components.
"""

from .shadow_cnn import ShadowCNN
from .qat_trainer import QATTrainer
from .model_evaluator import ModelEvaluator

__all__ = [
    'ShadowCNN',
    'QATTrainer', 
    'ModelEvaluator'
]