#!/usr/bin/env python3
"""
Script to enable PSRAM (SPIRAM) support in ESP-IDF sdkconfig
"""

import sys

def enable_psram(sdkconfig_path):
    """Enable PSRAM in sdkconfig file"""
    
    # Read the current config
    with open(sdkconfig_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace SPIRAM config
    new_lines = []
    spiram_section_found = False
    
    for i, line in enumerate(lines):
        if '# CONFIG_SPIRAM is not set' in line:
            # Replace with enabled config
            new_lines.append('CONFIG_SPIRAM=y\n')
            new_lines.append('CONFIG_SPIRAM_MODE_OCT=y\n')
            new_lines.append('CONFIG_SPIRAM_SPEED_80M=y\n')
            new_lines.append('CONFIG_SPIRAM_BOOT_INIT=y\n')
            new_lines.append('CONFIG_SPIRAM_USE_MALLOC=y\n')
            new_lines.append('CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=16384\n')
            new_lines.append('CONFIG_SPIRAM_MALLOC_RESERVE_INTERNAL=32768\n')
            spiram_section_found = True
            print("✅ Enabled PSRAM support")
        else:
            new_lines.append(line)
    
    if not spiram_section_found:
        print("❌ Could not find SPIRAM config line")
        return False
    
    # Write back
    with open(sdkconfig_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated {sdkconfig_path}")
    print("\nPSRAM Configuration:")
    print("  - SPIRAM: Enabled")
    print("  - Mode: Octal (OPI)")
    print("  - Speed: 80MHz")
    print("  - Boot Init: Yes")
    print("  - Use for malloc: Yes")
    print("\nNext steps:")
    print("  1. Run: idf.py reconfigure")
    print("  2. Run: idf.py build")
    print("  3. Run: idf.py flash monitor")
    
    return True

if __name__ == '__main__':
    sdkconfig_path = 'sdkconfig'
    if len(sys.argv) > 1:
        sdkconfig_path = sys.argv[1]
    
    success = enable_psram(sdkconfig_path)
    sys.exit(0 if success else 1)
