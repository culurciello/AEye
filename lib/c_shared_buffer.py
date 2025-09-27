"""
C Shared Buffer Interface for Video Capture

This module provides a Python interface to the C shared memory buffer
used by the video_capture C programs for cross-process frame sharing.
"""

import os
import mmap
import struct
import time
import ctypes
import numpy as np
import logging
from typing import Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FrameHeader:
    """Represents the shared frame header structure from C code."""
    width: int
    height: int
    format: int
    size: int
    timestamp_sec: int
    timestamp_usec: int
    frame_ready: int

class CSharedBuffer:
    """Interface to C shared memory buffer for video frames."""

    # Constants from C code
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # RGB format
    SHM_NAME = "/video_capture_frame"

    # Shared frame header structure (matching C struct)
    # struct { int width, height, format; size_t size; struct timeval timestamp; int frame_ready; }
    HEADER_FORMAT = "iiiq qq i"  # 7 fields: 4 ints, 1 size_t, 2 longs (timeval), 1 int
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    TOTAL_SIZE = HEADER_SIZE + FRAME_SIZE

    def __init__(self):
        """Initialize the shared buffer interface."""
        self.shm_fd = None
        self.mmap_obj = None
        self.is_connected = False
        self.last_frame_ready = 0

    def connect(self) -> bool:
        """
        Connect to the existing shared memory buffer created by C process.

        Returns:
            True if successfully connected, False otherwise
        """
        try:
            # Open existing shared memory object (read-only)
            self.shm_fd = os.open(self.SHM_NAME, os.O_RDONLY)

            # Memory map the shared memory
            self.mmap_obj = mmap.mmap(
                self.shm_fd,
                self.TOTAL_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ
            )

            self.is_connected = True
            logger.info(f"Connected to C shared buffer: {self.SHM_NAME}")
            return True

        except (OSError, IOError) as e:
            logger.error(f"Failed to connect to shared buffer {self.SHM_NAME}: {e}")
            self.cleanup()
            return False

    def disconnect(self):
        """Disconnect from the shared buffer."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except:
                pass
            self.mmap_obj = None

        if self.shm_fd is not None:
            try:
                os.close(self.shm_fd)
            except:
                pass
            self.shm_fd = None

        self.is_connected = False

    def read_header(self) -> Optional[FrameHeader]:
        """
        Read the frame header from shared memory.

        Returns:
            FrameHeader object or None if error
        """
        if not self.is_connected or not self.mmap_obj:
            return None

        try:
            # Read header from beginning of shared memory
            self.mmap_obj.seek(0)
            header_data = self.mmap_obj.read(self.HEADER_SIZE)

            if len(header_data) != self.HEADER_SIZE:
                return None

            # Unpack header structure
            unpacked = struct.unpack(self.HEADER_FORMAT, header_data)

            return FrameHeader(
                width=unpacked[0],
                height=unpacked[1],
                format=unpacked[2],
                size=unpacked[3],
                timestamp_sec=unpacked[4],
                timestamp_usec=unpacked[5],
                frame_ready=unpacked[6]
            )

        except Exception as e:
            logger.error(f"Error reading frame header: {e}")
            return None

    def read_frame_data(self) -> Optional[np.ndarray]:
        """
        Read the frame data from shared memory.

        Returns:
            NumPy array containing frame data (H, W, 3) or None if error
        """
        if not self.is_connected or not self.mmap_obj:
            return None

        try:
            # Read frame data after header
            self.mmap_obj.seek(self.HEADER_SIZE)
            frame_data = self.mmap_obj.read(self.FRAME_SIZE)

            if len(frame_data) != self.FRAME_SIZE:
                return None

            # Convert to numpy array and reshape
            # Assuming RGB format from C code
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame_array = frame_array.reshape((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3))

            # Convert RGB to BGR for OpenCV compatibility
            frame_bgr = frame_array[:, :, ::-1].copy()

            return frame_bgr

        except Exception as e:
            logger.error(f"Error reading frame data: {e}")
            return None

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Get the latest frame if available.

        Returns:
            Tuple of (frame_array, timestamp) or (None, None) if no new frame
        """
        header = self.read_header()
        if not header:
            return None, None

        # Check if new frame is available
        if header.frame_ready <= self.last_frame_ready:
            return None, None

        # Read frame data
        frame = self.read_frame_data()
        if frame is None:
            return None, None

        # Create timestamp from header
        timestamp = datetime.fromtimestamp(
            header.timestamp_sec + header.timestamp_usec / 1_000_000
        )

        # Update last frame ready counter
        self.last_frame_ready = header.frame_ready

        return frame, timestamp

    def wait_for_frame(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Wait for a new frame to become available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Tuple of (frame_array, timestamp) or (None, None) if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            frame, timestamp = self.get_latest_frame()
            if frame is not None:
                return frame, timestamp

            # Small sleep to avoid busy waiting
            time.sleep(0.001)  # 1ms

        return None, None

    def is_c_process_running(self) -> bool:
        """
        Check if the C video capture process is running.

        Returns:
            True if C process appears to be running, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            # Try to read header - if it fails, process might be down
            header = self.read_header()
            return header is not None
        except:
            return False


class CSharedBufferManager:
    """Manager class for handling C shared buffer lifecycle."""

    def __init__(self, c_program_path: Optional[str] = None):
        """
        Initialize the manager.

        Args:
            c_program_path: Optional path to the C video capture program
        """
        self.c_program_path = c_program_path
        self.buffer = CSharedBuffer()
        self.c_process = None

    def start_c_process(self, device: str = "/dev/video0") -> bool:
        """
        Start the C video capture process.

        Args:
            device: Video device path (Linux) or camera index (macOS)

        Returns:
            True if process started successfully
        """
        if not self.c_program_path or not os.path.exists(self.c_program_path):
            logger.error(f"C program not found: {self.c_program_path}")
            return False

        try:
            import subprocess

            # Start C process with webcam input
            cmd = [self.c_program_path, "-w", device]
            self.c_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Give process time to initialize shared memory
            time.sleep(2)

            # Try to connect to shared buffer
            if self.buffer.connect():
                logger.info(f"Started C video capture process: PID {self.c_process.pid}")
                return True
            else:
                logger.error("Failed to connect to shared buffer after starting C process")
                self.stop_c_process()
                return False

        except Exception as e:
            logger.error(f"Failed to start C process: {e}")
            return False

    def stop_c_process(self):
        """Stop the C video capture process."""
        if self.c_process:
            try:
                self.c_process.terminate()
                self.c_process.wait(timeout=5)
                logger.info("C video capture process terminated")
            except subprocess.TimeoutExpired:
                self.c_process.kill()
                logger.warning("C video capture process killed")
            except Exception as e:
                logger.error(f"Error stopping C process: {e}")
            finally:
                self.c_process = None

        self.buffer.disconnect()

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Get the latest frame from the shared buffer."""
        return self.buffer.get_latest_frame()

    def wait_for_frame(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Wait for a new frame from the shared buffer."""
        return self.buffer.wait_for_frame(timeout)

    def is_running(self) -> bool:
        """Check if the C process and buffer are running."""
        return (self.c_process is not None and
                self.c_process.poll() is None and
                self.buffer.is_c_process_running())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_c_process()