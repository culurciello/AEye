import cv2
import numpy as np
import os
import logging
import subprocess
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
from dataclasses import dataclass

try:
    from .c_shared_buffer import CSharedBufferManager
    C_SHARED_BUFFER_AVAILABLE = True
except ImportError:
    C_SHARED_BUFFER_AVAILABLE = False
    CSharedBufferManager = None

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


class VideoProcessor:
    def __init__(self, videos_dir: str, fps: int, pre_motion_seconds: int = 30, post_motion_seconds: int = 30, db_manager=None, video_backend: str = "opencv", c_program_path: str = None, camera_device: str = None):
        self.videos_dir = videos_dir
        self.fps = fps
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.db_manager = db_manager
        self.video_backend = video_backend
        self.video_buffer = CircularVideoBuffer(self.pre_motion_seconds + self.post_motion_seconds, fps)
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None

        # FFmpeg specific attributes
        self.ffmpeg_proc = None

        # C shared buffer specific attributes
        self.c_buffer_manager = None
        self.c_program_path = c_program_path
        self.camera_device = camera_device or self._get_default_camera_device()

        # Initialize C shared buffer if selected
        if self.video_backend == "c_shared_buffer":
            self._init_c_shared_buffer()

    def _get_default_camera_device(self) -> str:
        """Get default camera device based on platform."""
        import platform
        if platform.system() == "Darwin":  # macOS
            return "0"  # Camera index
        else:  # Linux
            return "/dev/video0"

    def _init_c_shared_buffer(self):
        """Initialize C shared buffer backend."""
        if not C_SHARED_BUFFER_AVAILABLE:
            logger.error("C shared buffer not available - missing c_shared_buffer module")
            raise RuntimeError("C shared buffer backend not available")

        if not self.c_program_path:
            # Try to find C program in libc directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            import platform
            if platform.system() == "Darwin":  # macOS
                self.c_program_path = os.path.join(script_dir, "libc", "video_capture")
            else:  # Linux
                self.c_program_path = os.path.join(script_dir, "libc", "video_capture")

        if not os.path.exists(self.c_program_path):
            logger.error(f"C video capture program not found: {self.c_program_path}")
            raise FileNotFoundError(f"C program not found: {self.c_program_path}")

        self.c_buffer_manager = CSharedBufferManager(self.c_program_path)
        logger.info(f"Initialized C shared buffer backend with program: {self.c_program_path}")

    def start_c_capture(self) -> bool:
        """Start the C video capture process."""
        if self.video_backend != "c_shared_buffer" or not self.c_buffer_manager:
            return False

        success = self.c_buffer_manager.start_c_process(self.camera_device)
        if success:
            logger.info(f"Started C video capture with device: {self.camera_device}")
        else:
            logger.error("Failed to start C video capture process")

        return success

    def stop_c_capture(self):
        """Stop the C video capture process."""
        if self.c_buffer_manager:
            self.c_buffer_manager.stop_c_process()
            logger.info("Stopped C video capture process")

    def get_frame_from_c_buffer(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """Get a frame from the C shared buffer."""
        if self.video_backend != "c_shared_buffer" or not self.c_buffer_manager:
            return None, None

        return self.c_buffer_manager.wait_for_frame(timeout=0.1)

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

        self.current_recording_path = file_path
        self.recording_frame_size = (height, width)
        self.is_recording = True

        if self.video_backend == "opencv":
            # Initialize OpenCV video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recording_writer = cv2.VideoWriter(file_path, fourcc, self.fps, (width, height))

            if not self.recording_writer.isOpened():
                logger.error(f"Failed to open video writer for {file_path}")
                return None

            # Write pre-motion frames from buffer
            frames_around = self.video_buffer.get_frames_around_time(
                trigger_time,
                self.pre_motion_seconds,
                0
            )

            for frame, _ in frames_around:
                if frame.shape[:2] == self.recording_frame_size:
                    self.recording_writer.write(frame)

        elif self.video_backend == "ffmpeg":
            # Initialize FFmpeg writer process
            ffmpeg_cmd = [
                "ffmpeg",
                "-hwaccel", "videotoolbox",
                "-f", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-y",
                file_path
            ]

            try:
                self.ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

                # Write pre-motion frames from buffer
                frames_around = self.video_buffer.get_frames_around_time(
                    trigger_time,
                    self.pre_motion_seconds,
                    0
                )

                for frame, _ in frames_around:
                    if frame.shape[:2] == self.recording_frame_size:
                        try:
                            self.ffmpeg_proc.stdin.write(frame.tobytes())
                        except Exception as e:
                            logger.error(f"Error writing frame to FFmpeg: {e}")
                            break

            except Exception as e:
                logger.error(f"Failed to start FFmpeg process: {e}")
                return None

        elif self.video_backend == "c_shared_buffer":
            # For C shared buffer, we use OpenCV writer but get frames from C process
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recording_writer = cv2.VideoWriter(file_path, fourcc, self.fps, (width, height))

            if not self.recording_writer.isOpened():
                logger.error(f"Failed to open video writer for {file_path}")
                return None

            # Write pre-motion frames from buffer
            frames_around = self.video_buffer.get_frames_around_time(
                trigger_time,
                self.pre_motion_seconds,
                0
            )

            for frame, _ in frames_around:
                if frame.shape[:2] == self.recording_frame_size:
                    self.recording_writer.write(frame)

        logger.info(f"Started recording with {self.video_backend}: {file_path}")
        return file_path
    
    def stop_recording(self, end_time: datetime, start_time: datetime) -> Optional[VideoSegment]:
        """Stop recording and create video segment metadata."""
        if not self.is_recording:
            return None

        # Write post-motion frames from buffer
        frames_around = self.video_buffer.get_frames_around_time(
            end_time,
            0,  # Don't get pre frames again
            self.post_motion_seconds
        )

        if (self.video_backend == "opencv" or self.video_backend == "c_shared_buffer") and self.recording_writer:
            for frame, _ in frames_around:
                if frame.shape[:2] == self.recording_frame_size:
                    self.recording_writer.write(frame)

            # Close OpenCV video writer
            self.recording_writer.release()
            self.recording_writer = None

        elif self.video_backend == "ffmpeg" and self.ffmpeg_proc:
            for frame, _ in frames_around:
                if frame.shape[:2] == self.recording_frame_size:
                    try:
                        self.ffmpeg_proc.stdin.write(frame.tobytes())
                    except Exception as e:
                        logger.error(f"Error writing frame to FFmpeg: {e}")
                        break

            # Close FFmpeg process
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
                logger.warning("FFmpeg process killed due to timeout")
            except Exception as e:
                logger.error(f"Error closing FFmpeg process: {e}")
            finally:
                self.ffmpeg_proc = None

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

        logger.info(f"Stopped recording with {self.video_backend}: {self.current_recording_path}")
        self.current_recording_path = None

        return segment

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the active recording using the selected backend."""
        if not self.is_recording or not hasattr(self, 'recording_frame_size'):
            return False

        if frame.shape[:2] != self.recording_frame_size:
            return False

        try:
            if (self.video_backend == "opencv" or self.video_backend == "c_shared_buffer") and self.recording_writer:
                self.recording_writer.write(frame)
                return True
            elif self.video_backend == "ffmpeg" and self.ffmpeg_proc:
                self.ffmpeg_proc.stdin.write(frame.tobytes())
                return True
        except Exception as e:
            logger.error(f"Error writing frame with {self.video_backend}: {e}")
            return False

        return False