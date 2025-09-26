#!/usr/bin/env python3

import cv2
import numpy as np
import os
import time
import argparse
import logging
import threading
from collections import deque
from datetime import datetime, timedelta


# Import our existing motion detection
from lib.motion_detector import AdaptiveMotionDetector
from lib.object_detector import ObjectDetector
from lib.video_processor import CircularVideoBuffer, VideoSegment, VideoProcessor
from lib.face_detector import FaceDetector
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
                 fps: int = 30,
                 use_gpu: bool = True,
                 image_capture_interval: int = 600,  # 10 minutes in seconds
                 headless: bool = False):
        """
        Initialize the motion triggered processor.

        Args:
            video_source: Video source (camera index or file path)
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

        # Create required directories
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize and warm up motion detection
        logger.info("Initializing motion detection...")
        self.motion_detector = AdaptiveMotionDetector(
            learning_rate=0.005,
            history_frames=3,
            min_contour_area=800,
            noise_reduction_kernel=7
        )
        
        # Warm up motion detector
        self._warmup_motion_detector()
        
        self.video_buffer = CircularVideoBuffer(buffer_duration, fps)

        # Initialize database manager
        self.db_manager = DatabaseManager(self.db_path)

        # Initialize object detection
        self.object_detector = ObjectDetector(self.db_manager)
        self.object_detector.init_yolo_detector()
        self.yolo_model = self.object_detector.yolo_model

        # Initialize face detection
        self.face_detector = FaceDetector(self.use_gpu, self.db_manager)
        self.face_detector.init_face_detector()

        # Initialize video processor
        self.video_processor = VideoProcessor(self.videos_dir, self.fps, self.pre_motion_seconds, self.post_motion_seconds, self.db_manager)
        self.video_processor.video_buffer = self.video_buffer

        # Pre-initialize video codec
        self.video_processor._warmup_video_writer()
        
        # Motion state tracking
        self.motion_active = False
        self.motion_start_time = None
        self.last_motion_time = None
        self.motion_timeout = 5.0  # Seconds without motion before stopping recording
        
        # Recording state
        self.is_recording = False
        self.current_recording_path = None
        self.recording_writer = None
        self.recording_frame_size = None
        
        # Processing queue
        self.processing_queue = deque()
        self.processing_thread = None
        self.stop_processing = False

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

    def process_recorded_video(self, segment: VideoSegment):
        """Process a recorded video with person-triggered face detection."""
        if not os.path.exists(segment.file_path):
            logger.error(f"Video file not found: {segment.file_path}")
            return

        logger.info(f"Processing video: {segment.file_path}")

        # Get motion event ID from database
        motion_event_data = self.db_manager.get_motion_event_by_video_file(segment.file_path)

        if not motion_event_data:
            logger.error(f"Motion event not found in database for: {segment.file_path}")
            return

        motion_event_id, _ = motion_event_data

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

                # Object detection - now returns person bboxes too
                objects_detected, person_bboxes = self.detect_objects_in_frame(frame, frame_time, motion_event_id)
                total_objects += objects_detected

                # Person-triggered face detection - only run if persons detected
                if person_bboxes and self.face_detector:
                    faces_detected = self.detect_faces_in_person_crops(frame, person_bboxes, frame_time, motion_event_id)
                    total_faces += faces_detected
                    if faces_detected > 0:
                        logger.debug(f"Detected {faces_detected} faces in {len(person_bboxes)} person crops")

        cap.release()

        # Update motion event with counts
        self.db_manager.update_motion_event_counts(motion_event_id, total_faces, total_objects)

        logger.info(f"Processed video: {segment.file_path} - Found {total_faces} faces in {len(person_bboxes) if person_bboxes else 0} person detections, {total_objects} total objects")

    def start_processing_thread(self):
        """Start background thread for processing recorded videos."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        logger.info("Processing thread started")
    
    def _processing_worker(self):
        """Background worker for processing recorded videos."""
        while not self.stop_processing:
            if self.processing_queue:
                try:
                    segment = self.processing_queue.popleft()
                    self.process_recorded_video(segment)
                except Exception as e:
                    logger.error(f"Error processing video segment: {e}")
            else:
                time.sleep(1)  # Wait if no segments to process

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

    def run(self):
        """Main processing loop with camera reconnection handling."""
        cap = None
        camera_retry_count = 0
        max_camera_retries = 10
        camera_retry_delay = 5

        # Initialize video capture with retry logic
        while camera_retry_count < max_camera_retries:
            try:
                logger.info(f"Attempting to connect to video source: {self.video_source}")
                cap = cv2.VideoCapture(self.video_source)

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
        
        # Start processing thread
        self.start_processing_thread()
        
        logger.info("Motion-triggered processor started")
        if not self.headless:
            logger.info("Press 'q' to quit, 's' to save current frame")
        else:
            logger.info("Running in headless mode - use Ctrl+C to quit")
        
        frame_count = 0
        failed_reads = 0
        max_failed_reads = 30  # Allow 30 consecutive failed reads before reconnecting

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    failed_reads += 1
                    logger.warning(f"Failed to read frame (attempt {failed_reads}/{max_failed_reads})")

                    if failed_reads >= max_failed_reads:
                        logger.error("Too many failed frame reads, attempting camera reconnection...")
                        cap.release()

                        # Attempt to reconnect
                        reconnect_attempts = 0
                        while reconnect_attempts < 5:
                            time.sleep(5)  # Wait before reconnecting
                            try:
                                cap = cv2.VideoCapture(self.video_source)
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
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                else:
                    failed_reads = 0  # Reset counter on successful read
                
                frame_count += 1
                current_time = datetime.now()
                
                # Add frame to circular buffer
                self.video_buffer.add_frame(frame, current_time)

                # Check for periodic image capture
                if (self.last_image_capture_time is None or
                    (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_interval):
                    self.save_periodic_image(frame, current_time)
                    self.last_image_capture_time = current_time

                # Process frame for motion detection
                motion_detected, motion_mask, motion_regions = self.motion_detector.process_frame(frame)
                
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
                                    logger.info(f"Motion detected with objects: {', '.join(detected_objects)} - Started recording")
                                else:
                                    logger.info("Motion detected - Started recording")
                            except Exception as e:
                                logger.warning(f"Object detection failed in live stream: {e}")
                                logger.info("Motion detected - Started recording")
                        else:
                            logger.info("Motion detected - Started recording")

                    self.last_motion_time = current_time

                    # Continue recording
                    if self.is_recording and self.video_processor.recording_writer:
                        self.video_processor.recording_writer.write(frame)
                
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
                                self.processing_queue.append(segment)
                                logger.info("Motion ended - Recording queued for processing")

                    # Continue recording for a bit after motion stops
                    if self.is_recording and self.video_processor.recording_writer:
                        self.video_processor.recording_writer.write(frame)
                
                # Create visualization and display only if not in headless mode
                if not self.headless:
                    output_frame = self.motion_detector.visualize_results(
                        frame, motion_mask, motion_regions, motion_detected
                    )

                    # Add recording indicator
                    if self.is_recording:
                        cv2.circle(output_frame, (output_frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(output_frame, "REC", (output_frame.shape[1] - 60, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
                        cv2.imwrite(f'frame_{timestamp}.jpg', output_frame)
                        logger.info(f"Frame saved as frame_{timestamp}.jpg")
                else:
                    # In headless mode, add a small delay to prevent excessive CPU usage
                    time.sleep(1.0 / self.fps)
        
        finally:
            # Cleanup
            if self.is_recording:
                segment = self.stop_recording(datetime.now(), self.motion_start_time)
                self.is_recording = False
                if segment:
                    self.processing_queue.append(segment)
            
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            logger.info("Motion-triggered processor stopped")

def main():
    parser = argparse.ArgumentParser(description='Motion-triggered video recording and face detection')
    parser.add_argument('--video-source', default=0,
                       help='Video source (camera index or file path)')
    parser.add_argument('--output-dir', default='data/',
                       help='Base output directory (videos/, images/, db/ will be created inside)')
    parser.add_argument('--buffer-duration', type=int, default=120,
                       help='Circular buffer duration in seconds (default: 120)')
    parser.add_argument('--pre-motion', type=int, default=30,
                       help='Seconds to record before motion (default: 30)')
    parser.add_argument('--post-motion', type=int, default=60,
                       help='Seconds to record after motion (default: 60)')
    parser.add_argument('--fps', type=int, default=15,
                       help='Target FPS for processing (default: 15)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage for face detection')
    parser.add_argument('--image-interval', type=int, default=600,
                       help='Seconds between periodic image captures (default: 600 = 10 minutes)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode without video display (for servers)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')
    
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
            headless=args.headless
        )
        
        # Run processor
        processor.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())