#!/usr/bin/env python3
"""
TEST STAGE 0: Quick verification with real WESAD data

This script tests Stage 0 with the actual WESAD dataset to verify
the integrity validation pipeline works correctly.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_stage0():
    """Test Stage 0 with real WESAD data"""
    print("🧪 TESTING STAGE 0: DATA INTEGRITY & SPLITTING")
    print("=" * 60)
    
    # Create test environment
    test_dir = Path("../test_outputs/")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify real data exists
    real_data_path = Path("/Users/ashidudissanayake/Dev/Shadow/model-development/data-input/")
    data_file = real_data_path / "flirt-wesad-acc-bvp-eda-temp-60-10.parquet"
    
    if not data_file.exists():
        print(f"❌ Real WESAD data not found: {data_file}")
        print("Available files:")
        for f in real_data_path.glob("*.parquet"):
            print(f"   - {f.name}")
        return False
    
    print(f"✅ Found real WESAD data: {data_file}")
    
    # Load and check real data
    real_data = pd.read_parquet(data_file)
    print(f"📊 Real data: {len(real_data)} samples, {len(real_data.columns)} features")
    print(f"   Subjects: {sorted(real_data['subject'].unique())}")
    print(f"   Labels: {sorted(real_data['label'].unique())}")
    
    # Import and run Stage 0
    sys.path.append('../stages')
    try:
        from stage0_data_integrity import Stage0DataIntegrity
        
        # Run Stage 0 with real data config
        stage0 = Stage0DataIntegrity(config_path="../config/pipeline_config.json")
        success = stage0.run_stage0()
        
        if success:
            print("\n🎉 STAGE 0 TEST PASSED!")
            
            # Show generated artifacts
            output_dir = Path("../outputs/stage0/")
            artifacts = list(output_dir.glob("*.json"))
            print(f"\n📋 Generated artifacts ({len(artifacts)}):")
            for artifact in artifacts:
                print(f"   - {artifact.name}")
                
            # Show summary
            summary_file = output_dir / "stage0_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print(f"\n📊 Summary:")
                print(f"   - Subjects: {summary['total_subjects']}")
                print(f"   - Samples: {summary['total_samples']:,}")
                print(f"   - LOSO folds: {summary['loso_folds']}")
                print(f"   - Integrity validated: {summary['integrity_validated']}")
                
                if 'temporal_analysis' in summary:
                    ta = summary['temporal_analysis']
                    print(f"   - Temporal leakage: {ta.get('overlap_percent', 'N/A')}%")
                    print(f"   - Leakage risk: {ta.get('temporal_leakage_risk', 'N/A')}")
            
            return True
        else:
            print("\n❌ STAGE 0 TEST FAILED!")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure required packages are installed:")
        print("  pip install pandas numpy scikit-learn")
        return False
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_stage0()
