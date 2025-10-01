import cv2
import numpy as np
import os
import logging
import subprocess
import threading
import queue
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """Represents a video segment with motion data."""
    start_time: datetime
    end_time: datetime
    file_path: str
    motion_detected: bool
    processed: bool = False
    detection_count: int = 0

class CircularVideoBuffer:
    """Maintains a circular buffer of video frames for pre/post motion recording."""
    
    def __init__(self, buffer_duration: int = 60, fps: int = 30):
        """
        Initialize circular buffer.
        
        Args:
            buffer_duration: Buffer duration in seconds
            fps: Frames per second
        """
        self.buffer_duration = buffer_duration
        self.fps = fps
        self.max_frames = buffer_duration * fps
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
    def add_frame(self, frame: np.ndarray, timestamp: datetime):
        """Add a frame to the circular buffer."""
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
    
    def get_frames_around_time(self, trigger_time: datetime, 
                              before_seconds: int = 30, 
                              after_seconds: int = 30) -> List[Tuple[np.ndarray, datetime]]:
        """
        Get frames around a specific trigger time.
        
        Args:
            trigger_time: The trigger timestamp
            before_seconds: Seconds before trigger
            after_seconds: Seconds after trigger
            
        Returns:
            List of (frame, timestamp) tuples
        """
        if not self.timestamps:
            return []
        
        start_time = trigger_time - timedelta(seconds=before_seconds)
        end_time = trigger_time + timedelta(seconds=after_seconds)
        
        selected_frames = []
        for frame, timestamp in zip(self.frames, self.timestamps):
            if start_time <= timestamp <= end_time:
                selected_frames.append((frame, timestamp))
        
        return selected_frames


class RTSPStreamReader:
    """Robust RTSP reader using FFmpeg subprocess for better HEVC handling."""
    
    def __init__(self, rtsp_url: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3  # BGR24
        self.ffmpeg_process = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.is_running = False
        self.read_thread = None
        self.error_count = 0
        self.max_errors = 10
        
    def start(self):
        """Start the FFmpeg subprocess for RTSP reading."""
        # FFmpeg command with HEVC-friendly parameters
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            # RTSP input parameters - these fix POC errors
            '-rtsp_transport', 'tcp',  # Use TCP instead of UDP
            '-max_delay', '5000000',  # 5 seconds max delay
            '-reorder_queue_size', '4096',  # Large reorder queue for HEVC
            '-buffer_size', '2097152',  # 2MB buffer
            '-fflags', '+genpts+discardcorrupt+nobuffer',  # Generate PTS, discard corrupt frames
            '-flags', 'low_delay',  # Low delay mode
            '-strict', 'experimental',
            '-err_detect', 'ignore_err',  # Ignore errors and continue
            '-i', self.rtsp_url,
            # Video processing
            '-vf', f'scale={self.width}:{self.height}',  # Scale to target resolution
            '-r', str(self.fps),  # Output framerate
            # Output raw video
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR
            '-an',  # No audio
            '-'  # Output to stdout
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer
            )
            
            self.is_running = True
            self.error_count = 0
            
            # Start reading thread
            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
            
            logger.info(f"Started RTSP stream reader for {self.rtsp_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RTSP reader: {e}")
            self.is_running = False
            return False
    
    def _read_frames(self):
        """Read frames from FFmpeg process in separate thread."""
        while self.is_running and self.ffmpeg_process:
            try:
                # Read raw frame data
                raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)
                
                if len(raw_frame) != self.frame_size:
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error("Too many read errors, stopping RTSP reader")
                        self.is_running = False
                        break
                    continue
                
                # Reset error count on successful read
                self.error_count = 0
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))
                
                # Add to queue with timestamp
                try:
                    # Remove old frame if queue is full
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    
                    self.frame_queue.put((frame, datetime.now()), timeout=0.1)
                    
                except queue.Full:
                    pass  # Drop frame if queue is full
                    
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    logger.error(f"Error reading frame: {e}")
                    self.error_count += 1
                    
                    if self.error_count > self.max_errors:
                        logger.error("Too many errors, stopping RTSP reader")
                        self.is_running = False
                        break
    
    def get_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Get a frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def stop(self):
        """Stop the RTSP reader."""
        self.is_running = False
        
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg process: {e}")
            
            self.ffmpeg_process = None
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=5)
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        logger.info("RTSP stream reader stopped")
    
    def restart(self):
        """Restart the RTSP reader (useful for error recovery)."""
        logger.info("Restarting RTSP reader...")
        self.stop()
        return self.start()


class OpenCVRTSPReader:
    """Alternative RTSP reader using OpenCV with optimized settings."""
    
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.is_opened = False
        
    def start(self):
        """Initialize OpenCV VideoCapture with RTSP-optimized settings."""
        try:
            # Set environment variable for TCP transport
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;5000000|reorder_queue_size;4096"
            
            # Create VideoCapture with specific backend
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False
            
            # Set capture properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # Try to set other properties (may not work with all cameras)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            
            self.is_opened = True
            
            # Get actual properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"OpenCV RTSP reader initialized: {width}x{height} @ {fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OpenCV RTSP reader: {e}")
            return False
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Read a frame from the RTSP stream."""
        if not self.cap or not self.is_opened:
            return None, None
        
        ret, frame = self.cap.read()
        if ret:
            return frame, datetime.now()
        else:
            # Try to reconnect on read failure
            logger.warning("Failed to read frame, attempting reconnect...")
            self.restart()
            return None, None
    
    def stop(self):
        """Release the VideoCapture."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        
    def restart(self):
        """Restart the capture."""
        self.stop()
        return self.start()


class VideoProcessor:
    def __init__(self, videos_dir: str, fps: int, rtsp_url: str = None,
                 pre_motion_seconds: int = 30, post_motion_seconds: int = 30, 
                 db_manager=None, camera_device=None, use_ffmpeg_reader: bool = True):
        self.videos_dir = videos_dir
        self.fps = fps
        self.rtsp_url = rtsp_url
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.db_manager = db_manager
        self.camera_device = camera_device or rtsp_url
        self.use_ffmpeg_reader = use_ffmpeg_reader
        
        self.video_buffer = CircularVideoBuffer(60, fps)
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None
        
        # RTSP reader (FFmpeg or OpenCV)
        self.rtsp_reader = None
        
        # Local camera capture (if no RTSP URL)
        self.local_cap = None
        
        # Initialize capture
        if self.rtsp_url:
            self.init_rtsp_capture()
        elif camera_device is not None:
            self.init_local_capture()

    def init_rtsp_capture(self):
        """Initialize RTSP capture with selected backend."""
        if not self.rtsp_url:
            return False
        
        try:
            if self.use_ffmpeg_reader:
                # Use FFmpeg subprocess for better HEVC handling
                self.rtsp_reader = RTSPStreamReader(self.rtsp_url, fps=self.fps)
                success = self.rtsp_reader.start()
            else:
                # Use OpenCV with optimized settings
                self.rtsp_reader = OpenCVRTSPReader(self.rtsp_url)
                success = self.rtsp_reader.start()
            
            if success:
                logger.info(f"Initialized RTSP capture from {self.rtsp_url} using {'FFmpeg' if self.use_ffmpeg_reader else 'OpenCV'}")
            else:
                logger.error("Failed to initialize RTSP capture")
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing RTSP capture: {e}")
            return False
    
    def init_local_capture(self):
        """Initialize local camera capture."""
        try:
            # Try to parse camera_device as integer first
            try:
                device_id = int(self.camera_device)
            except (ValueError, TypeError):
                device_id = self.camera_device
            
            self.local_cap = cv2.VideoCapture(device_id)
            
            if not self.local_cap.isOpened():
                logger.error(f"Failed to open local camera: {self.camera_device}")
                return False
            
            # Set capture properties
            self.local_cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.local_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info(f"Initialized local camera: {self.camera_device}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing local camera: {e}")
            return False
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Get a frame from the current capture source."""
        if self.rtsp_reader:
            if self.use_ffmpeg_reader:
                return self.rtsp_reader.get_frame()
            else:
                return self.rtsp_reader.get_frame()
        elif self.local_cap and self.local_cap.isOpened():
            ret, frame = self.local_cap.read()
            if ret:
                return frame, datetime.now()
        
        return None, None
    
    def process_frames(self):
        """Generator that yields frames from the capture source."""
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            frame, timestamp = self.get_frame()
            
            if frame is not None:
                consecutive_failures = 0
                
                # Add to circular buffer
                self.video_buffer.add_frame(frame, timestamp)
                
                # Write to active recording if needed
                if self.is_recording:
                    self.write_frame(frame)
                
                yield frame, timestamp
            else:
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    logger.error(f"Too many consecutive frame read failures ({max_failures})")
                    
                    # Try to restart RTSP connection
                    if self.rtsp_reader:
                        logger.info("Attempting to restart RTSP connection...")
                        if self.rtsp_reader.restart():
                            consecutive_failures = 0
                            logger.info("RTSP connection restarted successfully")
                        else:
                            logger.error("Failed to restart RTSP connection")
                            break
                    else:
                        break
                
                yield None, None

    def get_daily_directory(self, base_dir: str, date: datetime) -> str:
        """Get the daily directory path for a given date and ensure it exists."""
        date_str = date.strftime("%Y_%m_%d")
        daily_dir = os.path.join(base_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def _warmup_video_writer(self):
        """Pre-initialize video codec and writer to avoid delays during recording."""
        logger.info("Warming up video writer...")
        
        try:
            # Create a temporary test video file
            test_path = os.path.join(self.videos_dir, "test_warmup.mp4")
            
            # Default resolution for warmup
            width, height = 640, 480
            
            # Initialize video writer with the same settings we'll use for recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            test_writer = cv2.VideoWriter(test_path, fourcc, self.fps, (width, height))
            
            if test_writer.isOpened():
                # Write a few dummy frames to initialize the codec
                dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
                dummy_frame.fill(128)
                
                for i in range(5):
                    test_writer.write(dummy_frame)
                
                test_writer.release()
                
                # Clean up test file
                if os.path.exists(test_path):
                    os.remove(test_path)
                
                logger.info("Video writer warm-up completed")
            else:
                logger.warning("Could not initialize video writer during warm-up")
                
        except Exception as e:
            logger.warning(f"Video writer warm-up failed: {e}")

    def start_recording(self, trigger_time: datetime) -> str:
        """Start recording a motion-triggered video segment."""
        # Get daily directory for videos
        daily_video_dir = self.get_daily_directory(self.videos_dir, trigger_time)

        timestamp_str = trigger_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}.mp4"
        file_path = os.path.join(daily_video_dir, filename)
        
        # Get frame dimensions from buffer if available
        if self.video_buffer.frames:
            height, width = self.video_buffer.frames[-1].shape[:2]
        else:
            # Default resolution
            width, height = 1280, 720
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recording_writer = cv2.VideoWriter(file_path, fourcc, self.fps, (width, height))

        if not self.recording_writer.isOpened():
            logger.error(f"Failed to open video writer for {file_path}")
            return None

        self.current_recording_path = file_path
        self.recording_frame_size = (height, width)  # Store frame dimensions
        self.is_recording = True
        
        # Write pre-motion frames from buffer
        frames_around = self.video_buffer.get_frames_around_time(
            trigger_time, 
            self.pre_motion_seconds, 
            0  # Don't get post frames yet
        )
        
        for frame, _ in frames_around:
            if frame.shape[:2] == self.recording_frame_size:
                self.recording_writer.write(frame)
        
        logger.info(f"Started recording: {file_path}")
        return file_path
    
    def stop_recording(self, end_time: datetime, start_time: datetime) -> Optional[VideoSegment]:
        """Stop recording and create video segment metadata."""
        if not self.is_recording or not self.recording_writer:
            return None
        
        # Write post-motion frames from buffer
        frames_around = self.video_buffer.get_frames_around_time(
            end_time, 
            0,  # Don't get pre frames again
            self.post_motion_seconds
        )
        
        for frame, _ in frames_around:
            if self.recording_writer and hasattr(self, 'recording_frame_size') and frame.shape[:2] == self.recording_frame_size:
                self.recording_writer.write(frame)
        
        # Close video writer
        self.recording_writer.release()
        self.recording_writer = None
        self.is_recording = False
        
        # Create video segment
        segment = VideoSegment(
            start_time=start_time,
            end_time=end_time,
            file_path=self.current_recording_path,
            motion_detected=True,
            processed=False
        )
        
        # Store in database
        if self.db_manager:
            self.db_manager.store_motion_event(segment)
        
        logger.info(f"Stopped recording: {self.current_recording_path}")
        self.current_recording_path = None

        return segment

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the current recording if active."""
        if self.is_recording and self.recording_writer and self.recording_writer.isOpened():
            # Ensure frame dimensions match the recording dimensions
            if hasattr(self, 'recording_frame_size') and frame.shape[:2] == self.recording_frame_size:
                self.recording_writer.write(frame)
            else:
                logger.warning(f"Frame size mismatch: {frame.shape[:2]} vs {getattr(self, 'recording_frame_size', 'unknown')}")
    
    def cleanup(self):
        """Clean up resources."""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording(datetime.now(), datetime.now())
        
        # Stop RTSP reader
        if self.rtsp_reader:
            self.rtsp_reader.stop()
            self.rtsp_reader = None
        
        # Release local capture
        if self.local_cap:
            self.local_cap.release()
            self.local_cap = None
        
        logger.info("Video processor cleaned up")


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize processor with RTSP URL
    # Use FFmpeg reader for better HEVC handling
    processor = VideoProcessor(
        videos_dir="./videos",
        fps=30,
        rtsp_url="rtsp://192.168.6.244:554/11",
        use_ffmpeg_reader=True  # Set to False to use OpenCV instead
    )
    
    try:
        # Warm up the video writer
        processor._warmup_video_writer()
        
        # Process frames
        motion_active = False
        motion_start_time = None
        
        for frame, timestamp in processor.process_frames():
            if frame is not None:
                # Here you would do motion detection
                # For demo, just log that we got a frame
                logger.debug(f"Got frame at {timestamp}")
                
                # Example motion detection simulation
                # In real use, replace this with actual motion detection
                # motion_detected = your_motion_detection_function(frame)
                
                # Example recording logic:
                # if motion_detected and not motion_active:
                #     motion_active = True
                #     motion_start_time = timestamp
                #     processor.start_recording(timestamp)
                # elif not motion_detected and motion_active:
                #     motion_active = False
                #     processor.stop_recording(timestamp, motion_start_time)
                
    except KeyboardInterrupt:
        logger.info("Stopping video processor...")
    finally:
        processor.cleanup()