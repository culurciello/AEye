#!/usr/bin/env python3
"""
Shared memory frame reader for integrating with C video capture system.
Reads frames from shared memory created by video_capture.c
"""

import mmap
import struct
import time
import threading
import numpy as np
from ctypes import Structure, c_int, c_size_t, c_ubyte, c_long, c_uint32
from typing import Optional, Tuple
import os


class TimeVal(Structure):
    """C struct timeval equivalent"""
    _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]


class SharedFrameHeader(Structure):
    """C shared_frame_header_t structure equivalent"""
    _fields_ = [
        ("width", c_int),
        ("height", c_int), 
        ("format", c_uint32),
        ("size", c_size_t),
        ("timestamp", TimeVal),
        ("frame_ready", c_int),
    ]


class SharedMemoryFrameReader:
    """Read frames from shared memory created by video_capture.c"""
    
    def __init__(self, shm_name="/video_capture_frame"):
        self.shm_name = shm_name
        self.shm_fd = None
        self.mmap_obj = None
        self.header_size = struct.calcsize("iiLLlli")  # SharedFrameHeader size
        self.frame_data_size = 1280 * 720 * 3
        self.running = False
        self.last_frame = None
        self.last_timestamp = None
        self.frame_count = 0
        
    def connect(self) -> bool:
        """Connect to shared memory"""
        try:
            # Open shared memory
            self.shm_fd = os.open(f"/dev/shm{self.shm_name}", os.O_RDWR)
            
            # Memory map the shared frame structure (header + frame data)
            total_size = self.header_size + self.frame_data_size
            self.mmap_obj = mmap.mmap(self.shm_fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ)
            
            print("Connected to shared memory frame source")
            return True
            
        except Exception as e:
            print(f"Failed to connect to shared memory: {e}")
            print("Make sure video_capture is running first")
            return False
    
    def disconnect(self):
        """Disconnect from shared memory"""
        self.running = False
        
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
            
        if self.shm_fd:
            os.close(self.shm_fd)
            self.shm_fd = None
    
    def read_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Read a frame from shared memory.
        Returns: (frame_rgb, timestamp) or None if no new frame
        """
        if not self.mmap_obj:
            return None
            
        try:
            # Read the header from shared memory
            self.mmap_obj.seek(0)
            header_data = self.mmap_obj.read(self.header_size)
            
            if len(header_data) < self.header_size:
                return None
                
            # Parse header: width(4) + height(4) + format(4) + size(8) + timestamp(16) + frame_ready(4)
            width, height, format_val, size, tv_sec, tv_usec, frame_ready = struct.unpack("iiLLlli", header_data)
            
            if width != 1280 or height != 720:
                print(f"Unexpected frame size: {width}x{height}")
                return None
            
            if not frame_ready:
                return None
            
            # Convert timestamp to float seconds
            timestamp = float(tv_sec) + float(tv_usec) / 1000000.0
            
            # Check if this is a new frame
            if self.last_timestamp and timestamp <= self.last_timestamp:
                return None  # Same frame as before
            
            # Read frame data
            self.mmap_obj.seek(self.header_size)
            frame_data = self.mmap_obj.read(self.frame_data_size)
            
            if len(frame_data) < self.frame_data_size:
                return None
            
            # Convert bytes to numpy array (RGB24)
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame_rgb = frame_array.reshape((height, width, 3))
            
            # Convert RGB to BGR for OpenCV compatibility
            frame_bgr = frame_rgb[:, :, ::-1].copy()
            
            self.last_frame = frame_bgr
            self.last_timestamp = timestamp
            self.frame_count += 1
            
            return frame_bgr, timestamp
            
        except Exception as e:
            print(f"Error reading frame from shared memory: {e}")
            return None
    
    def wait_for_frame(self, timeout=1.0) -> Optional[Tuple[np.ndarray, float]]:
        """
        Wait for a new frame with timeout.
        Returns: (frame_bgr, timestamp) or None on timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame_data = self.read_frame()
            if frame_data:
                return frame_data
            time.sleep(0.001)  # 1ms sleep
            
        return None
    
    def start_reading(self):
        """Start continuous frame reading"""
        self.running = True
        
    def stop_reading(self):
        """Stop continuous frame reading"""
        self.running = False
    
    def is_connected(self) -> bool:
        """Check if connected to shared memory"""
        return self.mmap_obj is not None
    
    def get_frame_info(self) -> dict:
        """Get information about the current frame source"""
        return {
            "connected": self.is_connected(),
            "frame_count": self.frame_count,
            "last_timestamp": self.last_timestamp,
            "width": 1280,
            "height": 720,
            "format": "RGB24->BGR"
        }


# Test function
if __name__ == "__main__":
    import cv2
    
    reader = SharedMemoryFrameReader()
    
    if not reader.connect():
        print("Failed to connect. Make sure video_capture is running.")
        exit(1)
    
    print("Connected! Reading frames... Press 'q' to quit")
    
    try:
        while True:
            frame_data = reader.wait_for_frame(timeout=1.0)
            
            if frame_data:
                frame, timestamp = frame_data
                print(f"Frame {reader.frame_count}: {frame.shape} at {timestamp:.3f}s")
                
                # Display frame (optional)
                cv2.imshow("Shared Memory Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received (timeout)")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.disconnect()
        cv2.destroyAllWindows()