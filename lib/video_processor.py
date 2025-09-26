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
        self.frame_shape = None

    def add_frame(self, frame: np.ndarray, timestamp: datetime):
        """Add a frame to the circular buffer with memory optimization."""
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        # Store frame as bytes to reduce memory usage
        frame_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        self.frames.append(frame_bytes)
        self.timestamps.append(timestamp)

    def _decode_frame(self, frame_bytes: bytes) -> np.ndarray:
        """Decode frame from stored bytes."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
        for frame_bytes, timestamp in zip(self.frames, self.timestamps):
            if start_time <= timestamp <= end_time:
                decoded_frame = self._decode_frame(frame_bytes)
                selected_frames.append((decoded_frame, timestamp))

        return selected_frames


class FFmpegVideoWriter:
    """FFmpeg-based video writer for better performance and codec support."""

    def __init__(self, output_path: str, width: int, height: int, fps: int, crf: int = 23):
        """
        Initialize FFmpeg video writer.

        Args:
            output_path: Output video file path
            width: Video width
            height: Video height
            fps: Frames per second
            crf: Constant Rate Factor (0-51, lower = better quality)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.process = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.writer_thread = None
        self.is_writing = False
        self.error_occurred = False

    def start(self) -> bool:
        """Start the FFmpeg process and writer thread."""
        try:
            # FFmpeg command for H.264 encoding with hardware acceleration if available
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', '-',  # Input from stdin
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', str(self.crf),
                '-pix_fmt', 'yuv420p',
                self.output_path
            ]

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL
            )

            self.is_writing = True
            self.writer_thread = threading.Thread(target=self._writer_worker)
            self.writer_thread.daemon = True
            self.writer_thread.start()

            logger.info(f"Started FFmpeg writer for {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg writer: {e}")
            self.error_occurred = True
            return False

    def _writer_worker(self):
        """Worker thread that writes frames to FFmpeg process."""
        try:
            while self.is_writing or not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:  # Sentinel value to stop
                        break

                    # Resize frame if necessary
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))

                    # Write frame to FFmpeg stdin
                    self.process.stdin.write(frame.tobytes())
                    self.frame_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error writing frame to FFmpeg: {e}")
                    self.error_occurred = True
                    break

        except Exception as e:
            logger.error(f"FFmpeg writer thread error: {e}")
            self.error_occurred = True

    def write_frame(self, frame: np.ndarray) -> bool:
        """Add a frame to the writing queue."""
        if not self.is_writing or self.error_occurred:
            return False

        try:
            self.frame_queue.put(frame, timeout=0.1)
            return True
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
            return False

    def stop(self) -> bool:
        """Stop the FFmpeg writer and close the process."""
        if not self.process:
            return True

        try:
            self.is_writing = False

            # Send sentinel value to stop writer thread
            try:
                self.frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

            # Wait for writer thread to finish
            if self.writer_thread and self.writer_thread.is_alive():
                self.writer_thread.join(timeout=5.0)

            # Close FFmpeg stdin and wait for process to finish
            if self.process.stdin:
                self.process.stdin.close()

            self.process.wait(timeout=10.0)

            if self.process.returncode == 0:
                logger.info(f"FFmpeg writer finished successfully: {self.output_path}")
                return True
            else:
                stderr_output = self.process.stderr.read().decode('utf-8') if self.process.stderr else ""
                logger.error(f"FFmpeg process failed with return code {self.process.returncode}: {stderr_output}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg process timeout, terminating")
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            return False
        except Exception as e:
            logger.error(f"Error stopping FFmpeg writer: {e}")
            return False
        finally:
            self.process = None


class VideoProcessor:
    def __init__(self, videos_dir: str, fps: int, pre_motion_seconds: int = 30, post_motion_seconds: int = 30, db_manager=None):
        self.videos_dir = videos_dir
        self.fps = fps
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.db_manager = db_manager
        self.video_buffer = CircularVideoBuffer(60, fps)
        self.is_recording = False
        self.current_recording_path = None
        self.ffmpeg_writer = None
        self.recording_frame_size = None

    def get_daily_directory(self, base_dir: str, date: datetime) -> str:
        """Get the daily directory path for a given date and ensure it exists."""
        date_str = date.strftime("%Y_%m_%d")
        daily_dir = os.path.join(base_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def _warmup_ffmpeg_writer(self):
        """Test FFmpeg availability and basic functionality."""
        logger.info("Warming up FFmpeg writer...")

        try:
            # Test FFmpeg availability
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.error("FFmpeg not available or not working")
                return False

            logger.info("FFmpeg writer warm-up completed")
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"FFmpeg writer warm-up failed: {e}")
            return False


    def start_recording(self, trigger_time: datetime) -> str:
        """Start recording a motion-triggered video segment."""
        # Get daily directory for videos
        daily_video_dir = self.get_daily_directory(self.videos_dir, trigger_time)

        timestamp_str = trigger_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}.mp4"
        file_path = os.path.join(daily_video_dir, filename)

        # Get frame dimensions from buffer if available
        if self.video_buffer.frames and self.video_buffer.frame_shape:
            height, width = self.video_buffer.frame_shape[:2]
        else:
            # Default resolution
            width, height = 1280, 720

        # Initialize FFmpeg video writer
        self.ffmpeg_writer = FFmpegVideoWriter(file_path, width, height, self.fps)

        if not self.ffmpeg_writer.start():
            logger.error(f"Failed to start FFmpeg writer for {file_path}")
            self.ffmpeg_writer = None
            return None

        self.current_recording_path = file_path
        self.recording_frame_size = (height, width)
        self.is_recording = True

        # Write pre-motion frames from buffer
        frames_around = self.video_buffer.get_frames_around_time(
            trigger_time,
            self.pre_motion_seconds,
            0  # Don't get post frames yet
        )

        for frame, _ in frames_around:
            if frame.shape[:2] == self.recording_frame_size:
                self.ffmpeg_writer.write_frame(frame)

        logger.info(f"Started recording: {file_path}")
        return file_path

    def stop_recording(self, end_time: datetime, start_time: datetime) -> Optional[VideoSegment]:
        """Stop recording and create video segment metadata."""
        if not self.is_recording or not self.ffmpeg_writer:
            return None

        # Write post-motion frames from buffer
        frames_around = self.video_buffer.get_frames_around_time(
            end_time,
            0,  # Don't get pre frames again
            self.post_motion_seconds
        )

        for frame, _ in frames_around:
            if self.ffmpeg_writer and hasattr(self, 'recording_frame_size') and frame.shape[:2] == self.recording_frame_size:
                self.ffmpeg_writer.write_frame(frame)

        # Stop FFmpeg writer
        success = self.ffmpeg_writer.stop()
        if not success:
            logger.warning(f"FFmpeg writer may not have closed cleanly: {self.current_recording_path}")

        self.ffmpeg_writer = None
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

    def add_live_frame(self, frame: np.ndarray, timestamp: datetime):
        """Add a live frame to the buffer and current recording if active."""
        # Add to circular buffer
        self.video_buffer.add_frame(frame, timestamp)

        # If currently recording, also write to the video file
        if self.is_recording and self.ffmpeg_writer:
            if frame.shape[:2] == self.recording_frame_size:
                self.ffmpeg_writer.write_frame(frame)


