"""
Simulation Module

This module provides MAX30102 sensor simulation capabilities
for testing and validation of the ShadowAI pipeline.
"""

from .max30102_simulator import MAX30102Simulator,create_test_scenarios

__all__ = [
    'MAX30102Simulator',
    'create_test_scenarios'
]