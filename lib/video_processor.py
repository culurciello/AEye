import cv2
import numpy as np
import os
import logging
import sqlite3
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


class VideoProcessor:
    def __init__(self, videos_dir: str, fps: int, pre_motion_seconds: int = 30, post_motion_seconds: int = 30):
        self.videos_dir = videos_dir
        self.fps = fps
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.video_buffer = CircularVideoBuffer(60, fps)
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None

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
        self.store_motion_event(segment)
        
        logger.info(f"Stopped recording: {self.current_recording_path}")
        self.current_recording_path = None
        
        return segment
    

    def process_recorded_video(self, segment: VideoSegment):
        """Process a recorded video for face detection."""
        if not os.path.exists(segment.file_path):
            logger.error(f"Video file not found: {segment.file_path}")
            return
        
        logger.info(f"Processing video: {segment.file_path}")
        
        # Get motion event ID from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id FROM motion_events 
            WHERE video_file = ?
        ''', (segment.file_path,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            logger.error(f"Motion event not found in database for: {segment.file_path}")
            return
        
        motion_event_id = result[0]
        
        # Open video file
        cap = cv2.VideoCapture(segment.file_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {segment.file_path}")
            return
        
        frame_count = 0
        total_faces = 0
        total_objects = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 15th frame to reduce computational load
            if frame_count % 15 == 0:
                frame_time = segment.start_time + timedelta(seconds=frame_count / fps)

                # COMMENTED OUT FOR TESTING - Face detection causing delays
                # faces_detected = self.detect_faces_in_frame(frame, frame_time, motion_event_id)
                # total_faces += faces_detected

                # Object detection
                objects_detected = self.detect_objects_in_frame(frame, frame_time, motion_event_id)
                total_objects += objects_detected

        cap.release()

        # Update motion event with counts
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE motion_events
            SET processed = ?, face_count = ?, object_count = ?
            WHERE id = ?
        ''', (True, total_faces, total_objects, motion_event_id))
        conn.commit()
        conn.close()

        logger.info(f"Processed video: {segment.file_path} - Found {total_faces} faces, {total_objects} objects")
    