#!/usr/bin/env python3

import mmap
import struct
import threading
import time
import cv2
import numpy as np
from ctypes import *

# Shared memory structure matching C code
class SharedFrame(Structure):
    _fields_ = [
        ("width", c_int),
        ("height", c_int), 
        ("format", c_int),
        ("size", c_size_t),
        ("data", c_ubyte * (640 * 480 * 3)),  # RGB24 format from video_capture
        ("timestamp_sec", c_long),
        ("timestamp_usec", c_long),
        ("frame_ready", c_int),
        # pthread_mutex_t and pthread_cond_t are platform specific
        # We'll use Python threading instead
    ]

class FrameReader:
    def __init__(self, shared_memory_name="/video_capture_frame"):
        self.shared_memory_name = shared_memory_name
        self.running = False
        self.frame_callback = None
        self.thread = None
        self.shm_fd = None
        self.shared_frame = None
        self.last_timestamp = (0, 0)
        
    def connect(self):
        """Connect to shared memory created by C program"""
        try:
            # Open existing shared memory object
            self.shm_fd = os.open(f"/dev/shm{self.shared_memory_name}", os.O_RDWR)
            
            # Memory map the shared frame structure
            self.mmap_obj = mmap.mmap(self.shm_fd, sizeof(SharedFrame), 
                                     mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            
            # Cast to SharedFrame structure
            self.shared_frame = SharedFrame.from_buffer(self.mmap_obj)
            
            print(f"Connected to shared memory: {self.shared_memory_name}")
            print(f"Frame format: {self.shared_frame.width}x{self.shared_frame.height}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to shared memory: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from shared memory"""
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.shm_fd:
            os.close(self.shm_fd)
        print("Disconnected from shared memory")
    
    def rgb_to_bgr(self, rgb_data, width, height):
        """Convert RGB format to BGR for OpenCV"""
        # Reshape RGB data
        rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape((height, width, 3))
        
        # Convert RGB to BGR (OpenCV uses BGR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    
    def get_latest_frame(self, timeout=1.0):
        """Get the latest frame from shared memory"""
        if not self.shared_frame:
            return None
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if new frame is ready
            current_timestamp = (self.shared_frame.timestamp_sec, 
                               self.shared_frame.timestamp_usec)
            
            if (self.shared_frame.frame_ready and 
                current_timestamp != self.last_timestamp):
                
                # Copy frame data
                frame_data = bytes(self.shared_frame.data[:self.shared_frame.size])
                self.last_timestamp = current_timestamp
                
                # Convert RGB to BGR
                bgr_frame = self.rgb_to_bgr(frame_data, 
                                          self.shared_frame.width,
                                          self.shared_frame.height)
                
                return {
                    'frame': bgr_frame,
                    'timestamp': current_timestamp,
                    'width': self.shared_frame.width,
                    'height': self.shared_frame.height
                }
            
            time.sleep(0.01)  # 10ms polling interval
        
        return None
    
    def start_continuous_reading(self, callback):
        """Start continuous frame reading in a separate thread"""
        self.frame_callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._continuous_read_loop)
        self.thread.start()
        print("Started continuous frame reading")
    
    def stop_continuous_reading(self):
        """Stop continuous frame reading"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("Stopped continuous frame reading")
    
    def _continuous_read_loop(self):
        """Main loop for continuous frame reading"""
        while self.running:
            frame_data = self.get_latest_frame(timeout=0.1)
            if frame_data and self.frame_callback:
                self.frame_callback(frame_data)
            time.sleep(0.01)

# Example usage and testing
if __name__ == "__main__":
    import os
    
    def frame_processor(frame_data):
        """Example frame processing callback"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        
        print(f"Processing frame: {frame.shape}, timestamp: {timestamp}")
        
        # Example: Save frame as image every 30 frames
        if hasattr(frame_processor, 'frame_count'):
            frame_processor.frame_count += 1
        else:
            frame_processor.frame_count = 1
            
        if frame_processor.frame_count % 30 == 0:
            filename = f"frame_{timestamp[0]}_{timestamp[1]:06d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame: {filename}")
    
    # Test the frame reader
    reader = FrameReader()
    
    if reader.connect():
        print("Testing single frame capture...")
        frame_data = reader.get_latest_frame(timeout=5.0)
        if frame_data:
            print(f"Got frame: {frame_data['frame'].shape}")
            cv2.imwrite("test_frame.jpg", frame_data['frame'])
            print("Saved test_frame.jpg")
        else:
            print("No frame received (make sure video_capture is running)")
        
        print("\nTesting continuous capture (5 seconds)...")
        reader.start_continuous_reading(frame_processor)
        time.sleep(5)
        reader.stop_continuous_reading()
        
        reader.disconnect()
    else:
        print("Failed to connect. Run 'make run' in another terminal first.")