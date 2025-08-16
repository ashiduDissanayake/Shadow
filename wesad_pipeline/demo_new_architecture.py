#!/usr/bin/env python3
"""
Demonstration of the new WESAD Signal Quality Architecture

This script demonstrates the new 3-step processing flow:
1. Single Signal Quality Assessment (signal_quality.py)
2. Window Creation (windowing.py) 
3. Window Quality Assessment (window_quality.py)

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.analysis import SignalQuality, WindowQuality, WindowAnalyzer

def create_test_signals():
    """Create test signals with different quality levels."""
    sampling_rate = 64
    duration = 300  # 5 minutes
    signal_length = duration * sampling_rate
    t = np.linspace(0, duration, signal_length)
    
    # High quality BVP-like signal
    high_quality_signal = (
        -2.0 * np.sin(2 * np.pi * 1.2 * t) +          # 72 BPM heart rate
        0.5 * np.sin(2 * np.pi * 0.25 * t) +          # Respiratory component
        0.1 * np.random.randn(signal_length)          # Low noise
    )
    
    # Low quality signal (mostly noise)
    low_quality_signal = 5.0 * np.random.randn(signal_length)
    
    # Labels for both signals
    labels = np.random.randint(1, 4, signal_length)
    
    return {
        'high_quality': (high_quality_signal, labels),
        'low_quality': (low_quality_signal, labels)
    }

def demonstrate_new_architecture():
    """Demonstrate the new 3-step architecture."""
    print("=" * 60)
    print("WESAD Signal Quality Architecture Demonstration")
    print("=" * 60)
    
    # Initialize components
    config = WESADConfig()
    signal_quality = SignalQuality(config)
    window_quality = WindowQuality(config)
    window_analyzer = WindowAnalyzer(config)
    
    print(f"\nInitialized components:")
    print(f"  • SignalQuality: Single signal assessment only")
    print(f"  • WindowQuality: Windowed quality assessment")
    print(f"  • WindowAnalyzer: Pure window creation")
    
    # Create test signals
    signals = create_test_signals()
    
    for signal_name, (bvp_signal, labels) in signals.items():
        print(f"\n{'='*40}")
        print(f"Processing {signal_name.replace('_', ' ').title()} Signal")
        print(f"{'='*40}")
        
        print(f"Signal length: {len(bvp_signal):,} samples")
        
        # Step 1: Single Signal Quality Assessment
        print(f"\n🔍 Step 1: Single Signal Quality Assessment")
        quality_result = signal_quality.assess_signal_quality(bvp_signal)
        
        print(f"  Overall quality score: {quality_result['overall_score']:.3f}")
        print(f"  Quality level: {quality_result['quality_level']}")
        print(f"  Individual metrics:")
        for metric, score in quality_result['metrics'].items():
            print(f"    - {metric}: {score:.3f}")
        
        # Quality Gating
        quality_threshold = config.analysis.quality_threshold
        signal_passes_quality = quality_result['overall_score'] >= quality_threshold
        
        print(f"\n🚪 Quality Gating")
        print(f"  Threshold: {quality_threshold:.3f}")
        print(f"  Signal passes: {'✅ YES' if signal_passes_quality else '❌ NO'}")
        
        if signal_passes_quality:
            # Step 2: Window Creation
            print(f"\n🪟 Step 2: Window Creation")
            windows_result = window_analyzer.create_windows(bvp_signal, labels)
            
            print(f"  Windows created: {len(windows_result['windows'])}")
            print(f"  Window size: {config.analysis.window_size_seconds}s")
            print(f"  Overlap: {config.analysis.overlap_seconds}s")
            
            # Step 3: Window Quality Assessment
            print(f"\n⚖️  Step 3: Window Quality Assessment")
            
            # 3a: Individual window quality filtering
            filtered_windows, filter_stats = window_quality.filter_windows_by_quality(
                windows_result['windows']
            )
            
            print(f"  Window filtering:")
            print(f"    - Total windows: {filter_stats['total_windows']}")
            print(f"    - Accepted: {filter_stats['accepted_windows']}")
            print(f"    - Acceptance rate: {filter_stats['acceptance_rate']:.1%}")
            print(f"    - Avg accepted quality: {filter_stats['avg_accepted_quality']:.3f}")
            
            # 3b: Windowed quality analysis
            windowed_quality_result = window_quality.assess_windowed_quality(bvp_signal)
            
            print(f"  Windowed analysis:")
            print(f"    - Average quality: {windowed_quality_result['avg_quality']:.3f}")
            print(f"    - Quality std: {windowed_quality_result['quality_std']:.3f}")
            print(f"    - Quality range: [{windowed_quality_result['min_quality']:.3f}, {windowed_quality_result['max_quality']:.3f}]")
            
            # 3c: Quality distribution
            quality_distribution = window_quality.analyze_quality_distribution(filtered_windows)
            
            print(f"  Quality distribution:")
            level_counts = quality_distribution['quality_level_counts']
            for level, count in level_counts.items():
                percentage = quality_distribution['quality_level_percentages'][level]
                print(f"    - {level}: {count} windows ({percentage:.1f}%)")
            
        else:
            print(f"  ⏭️  Skipping windowing due to poor signal quality")
    
    print(f"\n{'='*60}")
    print("Architecture Benefits Demonstrated:")
    print("✅ Separation of Concerns: Each module has single responsibility")
    print("✅ Quality Gating: Poor signals filtered out early")
    print("✅ Reusability: Components work independently")
    print("✅ Better Testability: Each step can be tested separately")
    print("✅ Cleaner Flow: Signal → Windows → Window Quality")
    print(f"{'='*60}")

if __name__ == "__main__":
    demonstrate_new_architecture()