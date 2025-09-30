#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import argparse
import logging
import threading
import glob
import subprocess
import signal
import sys
import re
from collections import deque
from datetime import datetime, timedelta


# Import our existing motion detection
from lib.motion_detector import AdaptiveMotionDetector
from lib.object_detector import ObjectDetector
from lib.video_processor import CircularVideoBuffer, VideoProcessor # Original import if using OpenCV backend
from lib.database import DatabaseManager


# Setup logger
logger = logging.getLogger(__name__)


class MotionTriggeredProcessor:
    """
    Main processor that combines motion detection, video recording, and face detection.
    """
    
    def __init__(self,
                 video_source: str = 0,
                 videos_dir: str = "data/videos",
                 images_dir: str = "data/images",
                 db_path: str = "data/db/detections.db",
                 buffer_duration: int = 120,
                 pre_motion_seconds: int = 30,
                 post_motion_seconds: int = 60,
                 fps: int = 15,
                 use_gpu: bool = True,
                 image_capture_interval: int = 600,  # 10 minutes in seconds
                 headless: bool = False,
                 camera_device: str = None,
                 enable_face_detection: bool = True,
                 learning_rate: float = 0.003,
                 history_frames: int = 3,
                 min_contour_area: int = 300,
                 noise_reduction_kernel: int = 7,
                 min_motion_confidence: float = 0.3,
                 motion_timeout: float = 5.0):
        """
        Initialize the motion triggered processor.

        Args:
            video_source: Video source (camera index, file path, or directory path)
            videos_dir: Directory to save video recordings
            images_dir: Directory to save periodic images
            db_path: Database path for storing detections
            buffer_duration: Circular buffer duration in seconds
            pre_motion_seconds: Seconds to save before motion
            post_motion_seconds: Seconds to save after motion
            fps: Target FPS for processing
            use_gpu: Whether to use GPU for face detection
            image_capture_interval: Seconds between periodic image captures (default: 600 = 10 minutes)
            headless: Skip visualization and display for server/headless mode
            enable_face_detection: Whether to enable face detection (default: True)
            learning_rate: Background model learning rate (0.001-0.1)
            history_frames: Number of frames for temporal consistency
            min_contour_area: Minimum contour area for motion detection
            noise_reduction_kernel: Kernel size for morphological operations
            min_motion_confidence: Minimum motion confidence threshold (0.0-1.0)
            motion_timeout: Seconds without motion before stopping recording
        """
        self.video_source = video_source
        self.videos_dir = videos_dir
        self.images_dir = images_dir
        self.db_path = db_path
        self.buffer_duration = buffer_duration
        self.pre_motion_seconds = pre_motion_seconds
        self.post_motion_seconds = post_motion_seconds
        self.fps = fps
        self.use_gpu = use_gpu
        self.image_capture_interval = image_capture_interval
        self.headless = headless
        self.enable_face_detection = enable_face_detection

        # Create required directories
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize and warm up motion detection
        logger.info("Initializing motion detection...")
        self.motion_detector = AdaptiveMotionDetector(
            learning_rate=learning_rate,
            history_frames=history_frames,
            min_contour_area=min_contour_area,
            noise_reduction_kernel=noise_reduction_kernel
        )
        
        # Warm up motion detector
        self._warmup_motion_detector()

        # Initialize motion detection parameters
        self.min_motion_confidence = min_motion_confidence  # Minimum motion confidence threshold
        
        self.video_buffer = CircularVideoBuffer(buffer_duration, fps)

        # Initialize database manager
        self.db_manager = DatabaseManager(self.db_path)

        # Initialize object detection
        self.object_detector = ObjectDetector(self.db_manager)
        self.object_detector.init_yolo_detector()
        self.yolo_model = self.object_detector.yolo_model

        # Initialize face detection (conditionally)
        if self.enable_face_detection:
            try:
                from lib.face_detector import FaceDetector
                self.face_detector = FaceDetector(self.use_gpu, self.db_manager)
                self.face_detector.init_face_detector()
            except ImportError as e:
                logger.warning(f"Face detection dependencies not available: {e}")
                logger.info("Face detection disabled due to missing dependencies")
                self.face_detector = None
        else:
            self.face_detector = None
            logger.info("Face detection disabled")

        # Initialize video processor
        self.video_processor = VideoProcessor(
            self.videos_dir,
            self.fps,
            pre_motion_seconds=self.pre_motion_seconds,
            post_motion_seconds=self.post_motion_seconds,
            db_manager=self.db_manager,
            camera_device=camera_device,
        )

        self.video_processor.video_buffer = self.video_buffer

        # Pre-initialize video codec
        self.video_processor._warmup_video_writer()
        
        # Motion state tracking
        self.motion_active = False
        self.motion_start_time = None
        self.last_motion_time = None
        self.motion_timeout = motion_timeout  # Seconds without motion before stopping recording
        
        # Recording state
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None
        
        # Remove processing queue - process recordings directly

        # Periodic image capture
        self.last_image_capture_time = None

    def get_daily_directory(self, base_dir: str, date: datetime) -> str:
        """Get the daily directory path for a given date and ensure it exists."""
        date_str = date.strftime("%Y_%m_%d")
        daily_dir = os.path.join(base_dir, date_str)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir
    
    def _warmup_motion_detector(self):
        """Warm up motion detection with dummy frames."""
        logger.info("Warming up motion detection...")
        
        # Create a sequence of dummy frames with slight variations
        for i in range(10):
            # Create base frame
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame.fill(100 + i * 5)  # Gradually change brightness
            
            # Add some random noise
            noise = np.random.randint(-20, 20, dummy_frame.shape, dtype=np.int16)
            dummy_frame = np.clip(dummy_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Add some "motion" in later frames
            if i > 5:
                cv2.rectangle(dummy_frame, (200 + i * 10, 200), (250 + i * 10, 250), (255, 255, 255), -1)
            
            # Process frame through motion detector
            _, _, _ = self.motion_detector.process_frame(dummy_frame)
            
        logger.info("Motion detection warm-up completed")


    def start_recording(self, trigger_time: datetime) -> str:
        """Start recording a motion-triggered video segment."""
        return self.video_processor.start_recording(trigger_time)

    def stop_recording(self, end_time: datetime, start_time: datetime):
        """Stop recording and create video segment metadata."""
        return self.video_processor.stop_recording(end_time, start_time)

    def detect_objects_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
        """Detect objects in a single frame using YOLO and store results."""
        return self.object_detector.detect_objects_in_frame(frame, frame_time, motion_event_id)

    def detect_faces_in_person_crops(self, frame: np.ndarray, person_bboxes: list, frame_time: datetime, motion_event_id: int):
        """Detect faces within person bounding boxes."""
        if self.face_detector:
            return self.face_detector.detect_faces_in_person_crops(frame, person_bboxes, frame_time, motion_event_id)
        return 0

    def process_video_recording(self, segment):
        """Process a completed video recording for object and face detection."""
        if not segment or not os.path.exists(segment.file_path):
            logger.warning(f"Video file not found: {segment.file_path if segment else 'None'}")
            return

        logger.info(f"Processing video: {segment.file_path}")

        try:
            # Open video for processing
            cap = cv2.VideoCapture(segment.file_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {segment.file_path}")
                return

            # Get motion event ID for this recording
            motion_event_id = self.db_manager.store_motion_event(segment)

            total_faces = 0
            total_objects = 0
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Process every 5th frame to reduce processing time
                if frame_count % 5 != 0:
                    continue

                # Calculate frame timestamp
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_time = segment.start_time + timedelta(seconds=frame_count / fps)

                # Detect objects in frame
                object_count, person_bboxes = self.detect_objects_in_frame(frame, frame_time, motion_event_id)
                total_objects += object_count

                # Detect faces in person crops
                if person_bboxes:
                    faces_count = self.detect_faces_in_person_crops(frame, person_bboxes, frame_time, motion_event_id)
                    total_faces += faces_count

            cap.release()

            # Mark segment as processed
            segment.processed = True
            segment.detection_count = total_objects

            # Update the database with processing results
            self.db_manager.update_motion_event_counts(motion_event_id, total_faces, total_objects)

            logger.info(f"Processed video: {segment.file_path} - Found {total_faces} faces, {total_objects} total objects")

        except Exception as e:
            logger.error(f"Error processing video {segment.file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def save_periodic_image(self, frame: np.ndarray, current_time: datetime):
        """Save a periodic image from the camera feed."""
        try:
            # Get daily directory for images
            daily_image_dir = self.get_daily_directory(self.images_dir, current_time)

            # Create filename with timestamp
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{timestamp_str}.jpg"
            file_path = os.path.join(daily_image_dir, filename)

            # Save image
            success = cv2.imwrite(file_path, frame)
            if success:
                logger.info(f"Periodic image saved: {filename}")
            else:
                logger.error(f"Failed to save periodic image: {filename}")

        except Exception as e:
            logger.error(f"Error saving periodic image: {e}")


    def _run_single_source(self):
        """Run processing for a single video source (camera or file)."""
        cap = None
        camera_retry_count = 0
        max_camera_retries = 10
        camera_retry_delay = 5

        # Initialize video capture with retry logic
        while camera_retry_count < max_camera_retries:
            try:
                logger.info(f"Attempting to connect to video source: {self.video_source}")
                cap = cv2.VideoCapture(self.video_source)

                # Set RTSP timeout properties for better handling
                if str(self.video_source).startswith('rtsp://'):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second open timeout
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 second read timeout
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag

                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info("Camera connection successful")
                        break
                    else:
                        logger.warning("Camera opened but cannot read frames")
                        cap.release()
                        cap = None

                camera_retry_count += 1
                if camera_retry_count < max_camera_retries:
                    logger.warning(f"Camera connection failed, retrying in {camera_retry_delay}s (attempt {camera_retry_count}/{max_camera_retries})")
                    time.sleep(camera_retry_delay)

            except Exception as e:
                logger.error(f"Camera connection error: {e}")
                camera_retry_count += 1
                if camera_retry_count < max_camera_retries:
                    time.sleep(camera_retry_delay)

        if not cap or not cap.isOpened():
            logger.error(f"Failed to connect to camera after {max_camera_retries} attempts")
            return

        self._process_video_stream(cap)

    def _process_video_stream(self, cap, is_file=False, file_path=None):
        """Process frames from a video stream (camera or file)."""


        frame_count = 0
        failed_reads = 0
        max_failed_reads = 30 if not is_file else 5  # Files should fail faster
        last_successful_frame_time = datetime.now()
        connection_check_interval = 60

        try:
            while True:
                try:
                    ret, frame = cap.read()


                except Exception as e:
                    logger.warning(f"Error reading frame: {e}")
                    ret, frame = False, None

                if not ret or frame is None:
                    # For files, end of file is expected
                    if is_file:
                        logger.info(f"Finished processing file: {os.path.basename(file_path) if file_path else 'unknown'}")
                        break

                    # For streams, handle connection issues
                    failed_reads += 1
                    logger.warning(f"Failed to read frame (attempt {failed_reads}/{max_failed_reads})")

                    # Check if connection has been down too long
                    time_since_last_frame = (datetime.now() - last_successful_frame_time).total_seconds()
                    if time_since_last_frame > connection_check_interval:
                        logger.warning(f"No successful frames for {time_since_last_frame:.1f} seconds")

                    if failed_reads >= max_failed_reads:
                        if not is_file:
                            logger.error("Too many failed frame reads, attempting camera reconnection...")
                            cap.release()

                            # Attempt to reconnect (only for streams, not files)
                            reconnect_attempts = 0
                            while reconnect_attempts < 5:
                                time.sleep(5)  # Wait before reconnecting
                                try:
                                    cap = cv2.VideoCapture(self.video_source)

                                    # Set RTSP timeout properties for reconnection
                                    if str(self.video_source).startswith('rtsp://'):
                                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                                    if cap.isOpened():
                                        ret, test_frame = cap.read()
                                        if ret and test_frame is not None:
                                            logger.info("Camera reconnected successfully")
                                            failed_reads = 0
                                            break
                                        else:
                                            cap.release()
                                    reconnect_attempts += 1
                                    logger.warning(f"Reconnection attempt {reconnect_attempts}/5 failed")
                                except Exception as e:
                                    logger.error(f"Reconnection error: {e}")
                                    reconnect_attempts += 1

                            if reconnect_attempts >= 5:
                                logger.error("Camera reconnection failed, exiting...")
                                break
                        else:
                            # For files, just break on too many failed reads
                            break
                    else:
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                else:
                    failed_reads = 0  # Reset counter on successful read
                    last_successful_frame_time = datetime.now()  # Update last successful frame time

                frame_count += 1

                # Use current time for all processing
                current_time = datetime.now()

                # Add frame to circular buffer
                self.video_buffer.add_frame(frame, current_time)

                # Check for periodic image capture (less frequent for file processing)
                if not is_file and (self.last_image_capture_time is None or
                    (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_interval):
                    self.save_periodic_image(frame, current_time)
                    self.last_image_capture_time = current_time

                # Process frame for motion detection
                motion_detected, motion_mask, motion_regions = self.motion_detector.process_frame(frame)

                # Calculate motion confidence based on motion area and intensity
                motion_confidence = 0.0
                if motion_detected and motion_regions:
                    # Calculate confidence based on number and size of motion regions
                    total_motion_area = 0
                    for region in motion_regions:
                        try:
                            # Validate contour before calculating area
                            if len(region) >= 3:  # Need at least 3 points for a valid contour
                                area = cv2.contourArea(region)
                                if area > 0:  # Only add positive areas
                                    total_motion_area += area
                        except Exception:
                            # Skip invalid contours
                            continue

                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = total_motion_area / frame_area if frame_area > 0 else 0

                    # Confidence based on area ratio and number of regions
                    motion_confidence = min(1.0, area_ratio * 10 + len(motion_regions) * 0.1)

                # Apply motion confidence threshold
                if motion_detected and motion_confidence < self.min_motion_confidence:
                    motion_detected = False  # Filter out low-confidence motion

                # Handle motion state transitions
                if motion_detected:
                    if not self.motion_active:
                        # Motion started
                        self.motion_active = True
                        self.motion_start_time = current_time
                        self.current_recording_path = self.start_recording(current_time)
                        self.is_recording = True

                        # Run object detection on first motion frame to log what's detected
                        if self.yolo_model:
                            try:
                                results = self.yolo_model(frame, verbose=False)
                                detected_objects = []
                                for result in results:
                                    boxes = result.boxes
                                    if boxes is not None:
                                        for box in boxes:
                                            confidence = float(box.conf[0])
                                            if confidence > 0.5:
                                                class_id = int(box.cls[0])
                                                class_name = self.yolo_model.names[class_id]
                                                detected_objects.append(f"{class_name} ({confidence:.2f})")

                                if detected_objects:
                                    source_info = f"in {os.path.basename(file_path)}" if is_file else ""
                                    logger.info(f"Motion detected {source_info} with objects: {', '.join(detected_objects)} - Started recording")
                                else:
                                    source_info = f"in {os.path.basename(file_path)}" if is_file else ""
                                    logger.info(f"Motion detected {source_info} - Started recording")
                            except Exception as e:
                                logger.warning(f"Object detection failed: {e}")
                                source_info = f"in {os.path.basename(file_path)}" if is_file else ""
                                logger.info(f"Motion detected {source_info} - Started recording")
                        else:
                            source_info = f"in {os.path.basename(file_path)}" if is_file else ""
                            logger.info(f"Motion detected {source_info} - Started recording")

                    self.last_motion_time = current_time

                    # Continue recording
                    if self.is_recording:
                        self.video_processor.write_frame(frame)

                else:
                    # No motion detected
                    if self.motion_active and self.last_motion_time:
                        # Check if motion timeout exceeded
                        time_since_motion = (current_time - self.last_motion_time).total_seconds()

                        if time_since_motion > self.motion_timeout:
                            # Motion ended
                            self.motion_active = False
                            segment = self.stop_recording(current_time, self.motion_start_time)
                            self.is_recording = False

                            if segment:
                                logger.info("Motion ended - Processing recording directly")
                                self.process_video_recording(segment)

                    # Continue recording for a bit after motion stops
                    if self.is_recording:
                        self.video_processor.write_frame(frame)

                # Create visualization and display only if not in headless mode
                if not self.headless:
                    output_frame = frame.copy()

                    # Run object detection and draw bounding boxes
                    if self.yolo_model:
                        try:
                            results = self.yolo_model(frame, verbose=False)
                            for result in results:
                                boxes = result.boxes
                                if boxes is not None:
                                    for box in boxes:
                                        confidence = float(box.conf[0])
                                        if confidence > 0.5:
                                            # Get box coordinates
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                                            # Get class name
                                            class_id = int(box.cls[0])
                                            class_name = self.yolo_model.names[class_id]

                                            # Draw bounding box
                                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                            # Draw label with confidence
                                            label = f"{class_name} {confidence:.2f}"
                                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10),
                                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                                            cv2.putText(output_frame, label, (x1, y1 - 5),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        except Exception as e:
                            logger.warning(f"Object detection visualization failed: {e}")

                    # Add recording indicator
                    if self.is_recording:
                        cv2.circle(output_frame, (output_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(output_frame, "REC", (output_frame.shape[1] - 60, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Add file info for directory processing
                    if is_file and file_path:
                        filename = os.path.basename(file_path)
                        cv2.putText(output_frame, f"File: {filename}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Display results
                    cv2.imshow('Motion-Triggered Processor', output_frame)

                # Handle key presses only if not in headless mode
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        filename_suffix = f"_{os.path.basename(file_path)}" if is_file and file_path else ""
                        cv2.imwrite(f'frame_{timestamp}{filename_suffix}.jpg', output_frame)
                        logger.info(f"Frame saved as frame_{timestamp}{filename_suffix}.jpg")
                else:
                    # In headless mode, add a small delay to prevent excessive CPU usage
                    time.sleep(1.0 / self.fps)

        finally:
            # Cleanup for this stream/file only if not directory processing
            self._cleanup_processing()

    def _cleanup_processing(self):
        """Clean up processing resources."""
        # Cleanup
        if self.is_recording:
            segment = self.stop_recording(datetime.now(), self.motion_start_time)
            self.is_recording = False
            if segment:
                logger.info("Cleanup - Processing final recording directly")
                self.process_video_recording(segment)

        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Motion-triggered processor stopped")

def main():
    parser = argparse.ArgumentParser(description='Motion-triggered video recording and face detection')
    parser.add_argument('--video-source', default=0,
                       help='Video source (camera index, IP camera path)')
    parser.add_argument('--process-directory', action='store_true',
                       help='Process all video files in the specified directory')
    parser.add_argument('--output-dir', default='data/',
                       help='Base output directory (videos/, images/, db/ will be created inside)')
    parser.add_argument('--buffer-duration', type=int, default=90,
                       help='Circular buffer duration in seconds (default: 120)')
    parser.add_argument('--pre-motion', type=int, default=30,
                       help='Seconds to record before motion (default: 30)')
    parser.add_argument('--post-motion', type=int, default=60,
                       help='Seconds to record after motion (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for processing (default: 30)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage for face detection')
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection entirely')
    parser.add_argument('--image-interval', type=int, default=600,
                       help='Seconds between periodic image captures (default: 600 = 10 minutes)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without video display (for servers)')
    parser.add_argument('--camera-device',
                       help='Camera device (e.g., /dev/video0 for Linux, 0 for macOS)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')

    # Motion detection parameters
    parser.add_argument('--learning-rate', type=float, default=0.003,
                       help='Background model learning rate (0.001-0.1, default: 0.003)')
    parser.add_argument('--history-frames', type=int, default=3,
                       help='Number of frames for temporal consistency (default: 3)')
    parser.add_argument('--min-contour-area', type=int, default=300,
                       help='Minimum contour area for motion detection in pixels (default: 300)')
    parser.add_argument('--noise-kernel', type=int, default=7,
                       help='Noise reduction kernel size (default: 7)')
    parser.add_argument('--min-motion-confidence', type=float, default=0.3,
                       help='Minimum motion confidence threshold (0.0-1.0, default: 0.3)')
    parser.add_argument('--motion-timeout', type=float, default=5.0,
                       help='Seconds without motion before stopping recording (default: 5.0)')

    args = parser.parse_args()

    # Create derived paths from base output directory
    base_output_dir = args.output_dir.rstrip('/')
    videos_dir = os.path.join(base_output_dir, 'videos')
    images_dir = os.path.join(base_output_dir, 'images')
    db_path = os.path.join(base_output_dir, 'db', 'detections.db')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/aeye.log')
        ]
    )
    
    # Initialize processor
    try:
        # Convert video source to int if it's a number
        try:
            video_source = int(args.video_source)
        except ValueError:
            video_source = args.video_source


        processor = MotionTriggeredProcessor(
            video_source=video_source,
            videos_dir=videos_dir,
            images_dir=images_dir,
            db_path=db_path,
            buffer_duration=args.buffer_duration,
            pre_motion_seconds=args.pre_motion,
            post_motion_seconds=args.post_motion,
            fps=args.fps,
            use_gpu=not args.no_gpu,
            image_capture_interval=args.image_interval,
            headless=args.headless,
            camera_device=args.camera_device,
            enable_face_detection=not args.no_face_detection,
            learning_rate=args.learning_rate,
            history_frames=args.history_frames,
            min_contour_area=args.min_contour_area,
            noise_reduction_kernel=args.noise_kernel,
            min_motion_confidence=args.min_motion_confidence,
            motion_timeout=args.motion_timeout,
        )
        
        # Run processor
        processor._run_single_source()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())