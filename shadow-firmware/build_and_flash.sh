#!/bin/bash

# Script to build and flash Shadow firmware with ESP-IDF environment
# Usage: ./build_and_flash.sh

set -e  # Exit on error

echo "============================================"
echo "  Shadow Firmware Build & Flash Script"
echo "============================================"
echo ""

# Step 1: Source ESP-IDF environment
echo "📦 Step 1: Sourcing ESP-IDF environment..."
if [ -f "$HOME/Dev/esp/esp-idf/export.sh" ]; then
    . $HOME/Dev/esp/esp-idf/export.sh
    echo "✅ ESP-IDF environment loaded"
else
    echo "❌ ERROR: ESP-IDF not found at $HOME/Dev/esp/esp-idf/"
    echo "Please install ESP-IDF or update the path in this script"
    exit 1
fi

echo ""

# Step 2: Navigate to project directory
echo "📂 Step 2: Navigating to project directory..."
cd /Users/ashidudissanayake/Dev/Shadow/shadow-firmware
echo "✅ Current directory: $(pwd)"
echo ""

# Step 3: Build the project
echo "🔨 Step 3: Building firmware..."
idf.py build
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""

# Step 4: Flash to device
echo "⚡ Step 4: Flashing to ESP32-S3..."
idf.py flash
if [ $? -eq 0 ]; then
    echo "✅ Flash successful!"
else
    echo "❌ Flash failed!"
    exit 1
fi

echo ""

# Step 5: Monitor serial output
echo "📡 Step 5: Starting serial monitor..."
echo "Press Ctrl+] to exit monitor"
echo ""
sleep 2
idf.py monitor
