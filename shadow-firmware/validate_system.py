#!/usr/bin/env python3
"""
ESP32-S3 Stress Monitor System Validation Script
Validates that the compiled firmware contains all required components and symbols
"""

import os
import subprocess
import re
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_binary_symbols():
    """Check if required symbols are present in the compiled binary"""
    print("🔍 Checking compiled binary for required symbols...")
    
    # Path to the built ELF file
    elf_file = "build/shadow-firmware.elf"
    
    if not os.path.exists(elf_file):
        print("❌ ELF file not found. Run 'idf.py build' first.")
        return False
    
    # Required symbols to check
    required_symbols = [
        # Stress FSM symbols
        "stress_fsm_init",
        "stress_fsm_process_inference", 
        "stress_fsm_get_current_state",
        "stress_fsm_state_to_string",
        
        # Event log symbols
        "event_log_init",
        "event_log_add_transition",
        "event_log_get_events",
        
        # BLE service symbols
        "ble_stress_service_init",
        "ble_stress_service_start_advertising",
        "ble_stress_service_notify_fsm_state",
        
        # BLE GAP symbols (the ones that were causing linker issues)
        "esp_ble_gap_config_adv_data",
        "esp_ble_gap_start_advertising", 
        "esp_ble_gap_stop_advertising",
    ]
    
    # Use objdump to check symbols
    cmd = f"xtensa-esp32s3-elf-objdump -t {elf_file}"
    returncode, output, error = run_command(cmd)
    
    if returncode != 0:
        print(f"❌ Failed to run objdump: {error}")
        return False
    
    found_symbols = []
    missing_symbols = []
    
    for symbol in required_symbols:
        if symbol in output:
            found_symbols.append(symbol)
            print(f"✅ {symbol}")
        else:
            missing_symbols.append(symbol)
            print(f"❌ {symbol}")
    
    print(f"\n📊 Symbol Check Results:")
    print(f"   Found: {len(found_symbols)}/{len(required_symbols)}")
    print(f"   Missing: {len(missing_symbols)}")
    
    if missing_symbols:
        print(f"❌ Missing symbols: {missing_symbols}")
        return False
    
    print("✅ All required symbols found in binary!")
    return True

def check_component_integration():
    """Check if all components are properly linked"""
    print("\n🔗 Checking component integration...")
    
    # Check CMake build files
    build_dir = "build"
    component_dirs = [
        "esp-idf/stress_fsm",
        "esp-idf/event_log", 
        "esp-idf/ble_stress_service"
    ]
    
    for comp_dir in component_dirs:
        comp_path = os.path.join(build_dir, comp_dir)
        if os.path.exists(comp_path):
            print(f"✅ Component built: {comp_dir}")
        else:
            print(f"❌ Component missing: {comp_dir}")
            return False
    
    return True

def check_memory_usage():
    """Check memory usage of the compiled firmware"""
    print("\n💾 Checking memory usage...")
    
    # Look for memory usage in build output
    map_file = "build/shadow-firmware.map"
    
    if not os.path.exists(map_file):
        print("❌ Map file not found")
        return False
    
    # Read memory usage from build log or map file
    # This is a simplified check
    with open(map_file, 'r') as f:
        content = f.read()
        
    if ".text" in content and ".rodata" in content:
        print("✅ Memory sections properly allocated")
        return True
    
    print("❌ Memory layout issues detected")
    return False

def check_ble_configuration():
    """Check BLE configuration in sdkconfig"""
    print("\n📡 Checking BLE configuration...")
    
    sdkconfig_file = "sdkconfig"
    
    if not os.path.exists(sdkconfig_file):
        print("❌ sdkconfig not found")
        return False
    
    required_configs = [
        "CONFIG_BT_ENABLED=y",
        "CONFIG_BT_BLUEDROID_ENABLED=y", 
        "CONFIG_BT_BLE_ENABLED=y",
        "CONFIG_BT_GATTS_ENABLE=y",
        "CONFIG_BT_BLE_42_FEATURES_SUPPORTED=y"
    ]
    
    with open(sdkconfig_file, 'r') as f:
        config_content = f.read()
    
    for config in required_configs:
        if config in config_content:
            print(f"✅ {config}")
        else:
            print(f"❌ {config}")
            return False
    
    # Check that BLE 5.0 is disabled
    if "# CONFIG_BT_BLE_50_FEATURES_SUPPORTED is not set" in config_content:
        print("✅ CONFIG_BT_BLE_50_FEATURES_SUPPORTED disabled")
    else:
        print("❌ BLE 5.0 features should be disabled")
        return False
    
    return True

def main():
    print("🧪 ESP32-S3 Stress Monitor System Validation")
    print("=" * 50)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    checks = [
        ("BLE Configuration", check_ble_configuration),
        ("Component Integration", check_component_integration), 
        ("Memory Usage", check_memory_usage),
        ("Binary Symbols", check_binary_symbols),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    print(f"\n{'='*50}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 SYSTEM VALIDATION SUCCESSFUL!")
        print("   Your stress monitoring system is ready for testing!")
        return 0
    else:
        print("⚠️  VALIDATION ISSUES DETECTED")
        print("   Please review the failed checks above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
