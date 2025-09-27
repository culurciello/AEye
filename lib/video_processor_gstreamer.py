import os
import sys

# Set up proper library paths for macOS before importing GStreamer
if sys.platform == 'darwin':
    homebrew_lib = '/opt/homebrew/lib'
    if os.path.exists(homebrew_lib):
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        if homebrew_lib not in current_dyld:
            os.environ['DYLD_LIBRARY_PATH'] = f"{homebrew_lib}:{current_dyld}"

        # Set GStreamer plugin paths
        os.environ['GST_PLUGIN_SYSTEM_PATH'] = '/opt/homebrew/lib/gstreamer-1.0'
        os.environ['GST_PLUGIN_PATH'] = '/opt/homebrew/lib/gstreamer-1.0'

        # Set GI typelib path
        gi_path = '/opt/homebrew/lib/girepository-1.0'
        current_gi = os.environ.get('GI_TYPELIB_PATH', '')
        if gi_path not in current_gi:
            os.environ['GI_TYPELIB_PATH'] = f"{gi_path}:{current_gi}"

try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst, GstApp, GLib
except Exception as e:
    raise ImportError(f"GStreamer not properly installed or configured: {e}")

import numpy as np
import logging
import threading
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


class GStreamerVideoWriter:
    """GStreamer-based video writer for RTSP compatibility."""

    def __init__(self, file_path: str, fps: int, width: int, height: int):
        self.file_path = file_path
        self.fps = fps
        self.width = width
        self.height = height
        self.pipeline = None
        self.appsrc = None
        self.is_opened = False
        self.main_loop = None
        self.loop_thread = None
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup GStreamer pipeline for video recording."""
        try:
            Gst.init(None)

            # Create pipeline string with proper MP4 muxing for OpenCV compatibility
            pipeline_str = (
                f"appsrc name=source ! "
                f"videoconvert ! "
                f"video/x-raw,format=I420,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                f"x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
                f"mp4mux fragment-duration=1000 ! "
                f"filesink location={self.file_path}"
            )

            logger.info(f"GStreamer pipeline: {pipeline_str}")
            self.pipeline = Gst.parse_launch(pipeline_str)

            if not self.pipeline:
                logger.error("Failed to create GStreamer pipeline")
                return

            # Get appsrc element
            self.appsrc = self.pipeline.get_by_name("source")
            if not self.appsrc:
                logger.error("Failed to get appsrc element")
                return

            # Configure appsrc
            caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1")
            self.appsrc.set_property("caps", caps)
            self.appsrc.set_property("format", Gst.Format.TIME)
            self.appsrc.set_property("is-live", True)

            # Start the pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start GStreamer pipeline")
                return

            self.is_opened = True

            # Start GLib main loop in separate thread
            self.main_loop = GLib.MainLoop()
            self.loop_thread = threading.Thread(target=self.main_loop.run, daemon=True)
            self.loop_thread.start()

        except Exception as e:
            logger.error(f"Error setting up GStreamer pipeline: {e}")
            self.is_opened = False

    def isOpened(self) -> bool:
        """Check if the video writer is opened."""
        return self.is_opened

    def write(self, frame: np.ndarray):
        """Write a frame to the video file."""
        if not self.is_opened or not self.appsrc:
            return

        try:
            # Ensure frame is the right size
            if frame.shape[:2] != (self.height, self.width):
                # Resize frame if needed
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))

            # Convert frame to GStreamer buffer
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.pts = self.appsrc.get_clock().get_time() - self.pipeline.get_base_time()

            # Push buffer to pipeline
            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"Failed to push buffer: {ret}")

        except Exception as e:
            logger.error(f"Error writing frame: {e}")

    def release(self):
        """Release the video writer."""
        if self.appsrc:
            # Send EOS to finish the file properly
            logger.info("Sending EOS to finalize video file...")
            ret = self.appsrc.emit("end-of-stream")
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"EOS signal returned: {ret}")

        if self.pipeline:
            # Wait for EOS to propagate through the pipeline
            bus = self.pipeline.get_bus()
            if bus:
                # Wait up to 3 seconds for EOS message
                msg = bus.timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        logger.error(f"Pipeline error during finalization: {err}")
                    else:
                        logger.info("EOS received, file properly finalized")
                else:
                    logger.warning("Timeout waiting for EOS, file may be incomplete")

            # Stop the pipeline
            self.pipeline.set_state(Gst.State.NULL)

        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()

        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=3.0)

        self.is_opened = False
        logger.info(f"Released GStreamer video writer: {self.file_path}")


class VideoProcessor:
    def __init__(self, videos_dir: str, fps: int, pre_motion_seconds: int = 30, post_motion_seconds: int = 30, db_manager=None, camera_device=None):
        self.videos_dir = videos_dir
        self.fps = fps
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.db_manager = db_manager
        self.camera_device = camera_device
        self.video_buffer = CircularVideoBuffer(60, fps)
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None

        # Initialize GStreamer
        Gst.init(None)

    def get_daily_directory(self, base_dir: str, date: datetime) -> str:
        """Get the daily directory path for a given date and ensure it exists."""
        date_str = date.strftime("%Y_%m_%d")
        daily_dir = os.path.join(base_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir

    def _warmup_video_writer(self):
        """Pre-initialize GStreamer pipeline to avoid delays during recording."""
        logger.info("Warming up GStreamer video writer...")

        try:
            # Create a temporary test video file
            test_path = os.path.join(self.videos_dir, "test_warmup_gst.mp4")

            # Default resolution for warmup
            width, height = 640, 480

            # Initialize GStreamer video writer with the same settings we'll use for recording
            test_writer = GStreamerVideoWriter(test_path, self.fps, width, height)

            if test_writer.isOpened():
                # Write a few dummy frames to initialize the pipeline
                dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
                dummy_frame.fill(128)

                for i in range(5):
                    test_writer.write(dummy_frame)

                test_writer.release()

                # Clean up test file
                if os.path.exists(test_path):
                    os.remove(test_path)

                logger.info("GStreamer video writer warm-up completed")
            else:
                logger.warning("Could not initialize GStreamer video writer during warm-up")

        except Exception as e:
            logger.warning(f"GStreamer video writer warm-up failed: {e}")

    def start_recording(self, trigger_time: datetime) -> str:
        """Start recording a motion-triggered video segment."""
        # Get daily directory for videos
        daily_video_dir = self.get_daily_directory(self.videos_dir, trigger_time)

        timestamp_str = trigger_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_gst.mp4"
        file_path = os.path.join(daily_video_dir, filename)

        # Get frame dimensions from buffer if available
        if self.video_buffer.frames:
            height, width = self.video_buffer.frames[-1].shape[:2]
        else:
            # Default resolution
            width, height = 1280, 720

        # Initialize GStreamer video writer
        self.recording_writer = GStreamerVideoWriter(file_path, self.fps, width, height)

        if not self.recording_writer.isOpened():
            logger.error(f"Failed to open GStreamer video writer for {file_path}")
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
                self.recording_writer.write(frame)

        logger.info(f"Started GStreamer recording: {file_path}")
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

        logger.info(f"Stopped GStreamer recording: {self.current_recording_path}")
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