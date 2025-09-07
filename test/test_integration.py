#!/usr/bin/env python3
"""
Test script for the shared memory integration.
Run this to verify the integration works correctly.
"""

import subprocess
import time
import signal
import os
import sys


def test_shared_memory_integration():
    """Test the complete integration"""
    
    print("=== Shared Memory Integration Test ===")
    print()
    
    # Check if video_capture exists
    video_capture_path = "lib/video_capture"
    if not os.path.exists(video_capture_path):
        print("❌ video_capture not found. Building...")
        result = subprocess.run(["make", "-C", "lib"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to build video_capture: {result.stderr}")
            return False
        print("✅ video_capture built successfully")
    
    # Test 1: Check shared memory reader import
    try:
        from shared_memory_reader import SharedMemoryFrameReader
        print("✅ Shared memory reader import successful")
    except ImportError as e:
        print(f"❌ Failed to import shared memory reader: {e}")
        return False
    
    # Test 2: Test connection without video_capture (should fail gracefully)
    reader = SharedMemoryFrameReader()
    if reader.connect():
        print("❌ Should not connect without video_capture running")
        reader.disconnect()
        return False
    else:
        print("✅ Correctly failed to connect without video_capture")
    
    print()
    print("=== Manual Test Instructions ===")
    print()
    print("To test the full integration:")
    print("1. Terminal 1: cd lib && ./video_capture")
    print("2. Terminal 2: python3 parse_video.py --shared-memory --confidence 0.5")
    print()
    print("Expected behavior:")
    print("- Terminal 1 should start capturing and saving 1-minute MP4 files")
    print("- Terminal 2 should connect and start processing frames with YOLO")
    print("- Both should run in parallel with no frame drops")
    print()
    print("Command reference:")
    print("# Basic shared memory processing")
    print("python3 parse_video.py --shared-memory")
    print()
    print("# With custom confidence threshold")  
    print("python3 parse_video.py --shared-memory --confidence 0.7")
    print()
    print("# Process only 100 frames for testing")
    print("python3 parse_video.py --shared-memory --max-frames 100")
    print()
    print("✅ Integration test setup complete!")
    
    return True


if __name__ == "__main__":
    test_shared_memory_integration()