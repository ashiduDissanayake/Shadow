#!/usr/bin/env python3
"""Comprehensive test script for WESAD Data Loader"""

import sys
import os
import traceback
sys.path.append('wesad_pipeline')
sys.path.append('.')  # For shadowAI imports

from config.config import WESADConfig
from data.loader import WESADDataLoader


def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")

def test_validate_subjects(loader):
    print_section("validate_subjects")
    try:
        valid_subjects = loader.validate_subjects()
        print(f"✅ Validated subjects: {valid_subjects}")
    except Exception as e:
        print(f"❌ validate_subjects failed: {e}")
        traceback.print_exc()

def test_load_subject_data(loader, subject_id):
    print_section(f"load_subject_data (Subject {subject_id})")
    try:
        data = loader.load_subject_data(subject_id)
        if data:
            print(f"✅ Loaded subject {subject_id} data. Keys: {list(data.keys())}")
        else:
            print(f"⚠️  No data loaded for subject {subject_id}")
    except Exception as e:
        print(f"❌ load_subject_data failed: {e}")
        traceback.print_exc()

def test_load_multiple_subjects(loader, subjects):
    print_section("load_multiple_subjects")
    try:
        data = loader.load_multiple_subjects(subjects)
        print(f"✅ Loaded data for subjects: {list(data.keys())}")
    except Exception as e:
        print(f"❌ load_multiple_subjects failed: {e}")
        traceback.print_exc()

def test_get_dataset_statistics(loader):
    print_section("get_dataset_statistics")
    try:
        stats = loader.get_dataset_statistics()
        print(f"✅ Dataset statistics loaded. Keys: {list(stats.keys())}")
        print(f"   Total subjects: {stats.get('total_subjects', 'N/A')}")
        print(f"   Pipeline stats: {stats.get('pipeline_stats', {})}")
    except Exception as e:
        print(f"❌ get_dataset_statistics failed: {e}")
        traceback.print_exc()

def test_reset_statistics(loader):
    print_section("reset_statistics")
    try:
        loader.reset_statistics()
        print(f"✅ Statistics reset. Current stats: {loader.stats}")
    except Exception as e:
        print(f"❌ reset_statistics failed: {e}")
        traceback.print_exc()

def test_get_subject_file_path(loader, subject_id):
    print_section(f"get_subject_file_path (Subject {subject_id})")
    try:
        path = loader.get_subject_file_path(subject_id)
        print(f"✅ Subject {subject_id} file path: {path}")
    except Exception as e:
        print(f"❌ get_subject_file_path failed: {e}")
        traceback.print_exc()

def test_check_subject_availability(loader, subject_id):
    print_section(f"check_subject_availability (Subject {subject_id})")
    try:
        available = loader.check_subject_availability(subject_id)
        print(f"✅ Subject {subject_id} availability: {available}")
    except Exception as e:
        print(f"❌ check_subject_availability failed: {e}")
        traceback.print_exc()

def test_get_available_subjects(loader):
    print_section("get_available_subjects")
    try:
        available_subjects = loader.get_available_subjects()
        print(f"✅ Available subjects: {available_subjects}")
    except Exception as e:
        print(f"❌ get_available_subjects failed: {e}")
        traceback.print_exc()

def run_all_tests():
    print("=== Comprehensive WESADDataLoader Test Suite ===")
    try:
        # Setup configuration
        config = WESADConfig()
        config.dataset.wesad_path = "data/raw/wesad/"  # Adjust as needed
        config.dataset.subjects = [2, 3]  # Use a small set for test
        loader = WESADDataLoader(config)

        # Test get_available_subjects (should work even if no data)
        test_get_available_subjects(loader)
        available_subjects = loader.get_available_subjects()
        test_subject = available_subjects[0] if available_subjects else config.dataset.subjects[0]

        # Test validate_subjects
        test_validate_subjects(loader)

        # Test get_subject_file_path
        test_get_subject_file_path(loader, test_subject)

        # Test check_subject_availability
        test_check_subject_availability(loader, test_subject)

        # Test load_subject_data
        test_load_subject_data(loader, test_subject)

        # Test load_multiple_subjects
        test_load_multiple_subjects(loader, [test_subject])

        # Test get_dataset_statistics
        test_get_dataset_statistics(loader)

        # Test reset_statistics
        test_reset_statistics(loader)

        print("\n🎉 All WESADDataLoader tests completed!")
        return True
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_all_tests()