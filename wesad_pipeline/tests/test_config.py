#!/usr/bin/env python3
"""Test script for WESAD Configuration"""

import sys
import os
sys.path.append('wesad_pipeline')

# Import the config
from config.config import WESADConfig, create_default_config

def test_config():
    print("=== Testing WESAD Configuration ===")
    
    try:
        # Test default configuration
        print("1. Creating default configuration...")
        config = WESADConfig()
        print("✅ Default configuration created successfully")
        
        # Print some key config values
        print(f"   WESAD Path: {config.dataset.wesad_path}")
        print(f"   Subjects: {config.dataset.subjects}")
        print(f"   Window Size: {config.analysis.window_size_seconds}s")
        print(f"   Quality Threshold: {config.analysis.quality_threshold}")
        
        # Test configuration with custom parameters
        print("\n2. Creating custom configuration...")
        custom_config = create_default_config()
        custom_config.dataset.wesad_path = "data/raw/wesad/"
        custom_config.dataset.subjects = [2, 3, 4, 5]
        print("✅ Custom configuration created successfully")
        
        # Test validation
        print("\n3. Testing configuration validation...")
        config.validate()
        print("✅ Configuration validation passed")
        
        # Test directory creation
        print("\n4. Testing directory creation...")
        config.create_output_directories()
        print("✅ Output directories created successfully")
        
        # Test configuration conversion
        print("\n5. Testing configuration conversion...")
        config_dict = config.to_dict()
        print(f"✅ Configuration converted to dict ({len(config_dict)} sections)")
        
        print("\n🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config()