#!/usr/bin/env python3

import cv2
import numpy as np
import sqlite3
import pickle
import os
import time
import argparse
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# YOLO object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# COMMENTED OUT FOR TESTING - Face detection
# from insightface.app import FaceAnalysis

# Import our existing motion detection
from motion_detection import AdaptiveMotionDetector

# Setup logger
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

class MotionTriggeredProcessor:
    """
    Main processor that combines motion detection, video recording, and face detection.
    """
    
    def __init__(self,
                 video_source: str = 0,
                 videos_dir: str = "data/videos",
                 images_dir: str = "data/images",
                 db_path: str = "data/db/detections.db",
                 base_output_dir: str = "data",
                 buffer_duration: int = 60,
                 pre_motion_seconds: int = 30,
                 post_motion_seconds: int = 30,
                 fps: int = 30,
                 use_gpu: bool = True,
                 image_capture_interval: int = 600):  # 10 minutes in seconds
        """
        Initialize the motion triggered processor.

        Args:
            video_source: Video source (camera index or file path)
            output_dir: Directory to save video recordings
            db_path: Database path for storing detections
            base_output_dir: Base output directory for derived paths
            buffer_duration: Circular buffer duration in seconds
            pre_motion_seconds: Seconds to save before motion
            post_motion_seconds: Seconds to save after motion
            fps: Target FPS for processing
            use_gpu: Whether to use GPU for face detection
            image_capture_interval: Seconds between periodic image captures (default: 600 = 10 minutes)
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
        
        # Initialize YOLO object detection
        self.init_yolo_detector()

        # Initialize face detection (COMMENTED OUT FOR TESTING)
        # self.init_face_detector()
        self.face_app = None

        # Initialize database
        self.init_database()
        
        # Pre-initialize video codec
        self._warmup_video_writer()
        
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
        
    # COMMENTED OUT FOR TESTING - Face detector initialization causing delays
    # def init_face_detector(self):
    #     """Initialize and warm up face detection using InsightFace."""
    #     try:
    #         logger.info("Initializing InsightFace neural networks...")
    #         self.face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    #         ctx_id = 0 if self.use_gpu else -1
    #         self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    #
    #         # Warm up the neural network with dummy data
    #         logger.info("Warming up face detection neural networks...")
    #         dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    #         dummy_frame.fill(128)  # Fill with gray color
    #
    #         # Add some noise to make it more realistic
    #         noise = np.random.randint(0, 50, dummy_frame.shape, dtype=np.uint8)
    #         dummy_frame = cv2.add(dummy_frame, noise)
    #
    #         # Run several warm-up inferences
    #         for i in range(3):
    #             _ = self.face_app.get(dummy_frame)
    #             logger.debug(f"Face detection warm-up iteration {i+1}/3 completed")
    #
    #         logger.info(f"InsightFace initialized and warmed up (ctx_id: {ctx_id})")
    #
    #     except Exception as e:
    #         logger.error(f"Failed to initialize face detection: {e}")
    #         self.face_app = None
    
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

    def init_yolo_detector(self):
        """Initialize YOLO object detection model."""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - object detection disabled")
            self.yolo_model = None
            return

        try:
            logger.info("Initializing YOLO object detection...")
            # Use YOLOv8n (nano) for speed, or YOLOv8s/m/l/x for better accuracy
            self.yolo_model = YOLO('models/yolov8n.pt')

            # Warm up the model
            logger.info("Warming up YOLO model...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            dummy_frame.fill(128)

            # Run a few warm-up inferences
            for i in range(3):
                _ = self.yolo_model(dummy_frame, verbose=False)
                logger.debug(f"YOLO warm-up iteration {i+1}/3 completed")

            logger.info("YOLO object detection initialized and warmed up")

        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.yolo_model = None

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
    
    def init_database(self):
        """Initialize database for storing motion and face detections."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create motion events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS motion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                video_file TEXT,
                duration_seconds REAL,
                processed BOOLEAN DEFAULT FALSE,
                face_count INTEGER DEFAULT 0,
                object_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create face detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                motion_event_id INTEGER,
                frame_timestamp TIMESTAMP,
                face_crop BLOB,
                face_embedding BLOB,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (motion_event_id) REFERENCES motion_events(id)
            )
        ''')

        # Create object detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                motion_event_id INTEGER,
                frame_timestamp TIMESTAMP,
                class_name TEXT,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (motion_event_id) REFERENCES motion_events(id)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
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
    
    def store_motion_event(self, segment: VideoSegment):
        """Store motion event in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        duration = (segment.end_time - segment.start_time).total_seconds()
        
        cursor.execute('''
            INSERT INTO motion_events 
            (start_time, end_time, video_file, duration_seconds, processed)
            VALUES (?, ?, ?, ?, ?)
        ''', (segment.start_time, segment.end_time, segment.file_path, 
              duration, segment.processed))
        
        conn.commit()
        conn.close()
    
    # COMMENTED OUT FOR TESTING - Face detection causing delays
    # def detect_faces_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
    #     """Detect faces in a single frame and store results."""
    #     if not self.face_app:
    #         return []
    #
    #     try:
    #         faces = self.face_app.get(frame)
    #         face_count = 0
    #
    #         for face in faces:
    #             bbox = face.bbox.astype(int)
    #             confidence = face.det_score
    #
    #             # Skip low confidence detections
    #             if confidence < 0.7:
    #                 continue
    #
    #             x1, y1, x2, y2 = bbox
    #             bbox_w = x2 - x1
    #             bbox_h = y2 - y1
    #
    #             # Skip very small faces
    #             if bbox_w < 30 or bbox_h < 30:
    #                 continue
    #
    #             # Extract face crop
    #             face_crop = frame[y1:y2, x1:x2]
    #
    #             # Convert to bytes
    #             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    #             _, face_buffer = cv2.imencode('.jpg', face_crop, encode_param)
    #             face_bytes = face_buffer.tobytes()
    #
    #             # Get normalized embedding
    #             embedding = face.embedding
    #             norm = np.linalg.norm(embedding)
    #             if norm > 0:
    #                 embedding = embedding / norm
    #
    #             # Store in database
    #             self.store_face_detection(
    #                 motion_event_id, frame_time, face_bytes, embedding,
    #                 confidence, x1, y1, bbox_w, bbox_h
    #             )
    #
    #             face_count += 1
    #
    #         return face_count
    #
    #     except Exception as e:
    #         logger.error(f"Error detecting faces: {e}")
    #         return 0
    
    # COMMENTED OUT FOR TESTING - Face detection causing delays
    # def store_face_detection(self, motion_event_id: int, frame_time: datetime,
    #                        face_bytes: bytes, embedding: np.ndarray, confidence: float,
    #                        x: int, y: int, w: int, h: int):
    #     """Store face detection in database."""
    #     conn = sqlite3.connect(self.db_path)
    #     cursor = conn.cursor()
    #
    #     embedding_bytes = pickle.dumps(embedding)
    #
    #     cursor.execute('''
    #         INSERT INTO face_detections
    #         (motion_event_id, frame_timestamp, face_crop, face_embedding, confidence,
    #          bbox_x, bbox_y, bbox_width, bbox_height)
    #         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    #     ''', (motion_event_id, frame_time, face_bytes, embedding_bytes, confidence,
    #           x, y, w, h))
    #
    #     conn.commit()
    #     conn.close()

    def detect_objects_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
        """Detect objects in a single frame using YOLO and store results."""
        if not self.yolo_model:
            return []

        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            object_count = 0

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]

                        # Skip low confidence detections
                        if confidence < 0.5:
                            continue

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox_x = int(x1)
                        bbox_y = int(y1)
                        bbox_w = int(x2 - x1)
                        bbox_h = int(y2 - y1)

                        # Skip very small objects
                        if bbox_w < 20 or bbox_h < 20:
                            continue

                        # Store in database
                        self.store_object_detection(
                            motion_event_id, frame_time, class_name, confidence,
                            bbox_x, bbox_y, bbox_w, bbox_h
                        )

                        object_count += 1

            return object_count

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return 0

    def store_object_detection(self, motion_event_id: int, frame_time: datetime,
                               class_name: str, confidence: float,
                               x: int, y: int, w: int, h: int):
        """Store object detection in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO object_detections
            (motion_event_id, frame_timestamp, class_name, confidence,
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, class_name, confidence,
              x, y, w, h))

        conn.commit()
        conn.close()

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
        """Main processing loop."""
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Could not open video source: {self.video_source}")
            return
        
        # Start processing thread
        self.start_processing_thread()
        
        logger.info("Motion-triggered processor started")
        logger.info("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                        self.start_recording(current_time)

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
                    if self.is_recording and self.recording_writer:
                        self.recording_writer.write(frame)
                
                else:
                    # No motion detected
                    if self.motion_active and self.last_motion_time:
                        # Check if motion timeout exceeded
                        time_since_motion = (current_time - self.last_motion_time).total_seconds()
                        
                        if time_since_motion > self.motion_timeout:
                            # Motion ended
                            self.motion_active = False
                            segment = self.stop_recording(current_time, self.motion_start_time)
                            
                            if segment:
                                self.processing_queue.append(segment)
                                logger.info("Motion ended - Recording queued for processing")
                    
                    # Continue recording for a bit after motion stops
                    if self.is_recording and self.recording_writer:
                        self.recording_writer.write(frame)
                
                # Create visualization
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
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    cv2.imwrite(f'frame_{timestamp}.jpg', output_frame)
                    logger.info(f"Frame saved as frame_{timestamp}.jpg")
                
                # Limit FPS
                time.sleep(1.0 / self.fps)
        
        finally:
            # Cleanup
            if self.is_recording:
                segment = self.stop_recording(datetime.now(), self.motion_start_time)
                if segment:
                    self.processing_queue.append(segment)
            
            self.stop_processing = True
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Motion-triggered processor stopped")

def main():
    parser = argparse.ArgumentParser(description='Motion-triggered video recording and face detection')
    parser.add_argument('--video-source', default=0,
                       help='Video source (camera index or file path)')
    parser.add_argument('--output-dir', default='data/',
                       help='Base output directory (videos/, images/, db/ will be created inside)')
    parser.add_argument('--buffer-duration', type=int, default=60,
                       help='Circular buffer duration in seconds (default: 60)')
    parser.add_argument('--pre-motion', type=int, default=30,
                       help='Seconds to record before motion (default: 30)')
    parser.add_argument('--post-motion', type=int, default=30,
                       help='Seconds to record after motion (default: 30)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for processing (default: 30)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage for face detection')
    parser.add_argument('--image-interval', type=int, default=600,
                       help='Seconds between periodic image captures (default: 600 = 10 minutes)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('motion_processor.log')
        ]
    )
    
    # Initialize processor
    try:
        # Convert video source to int if it's a number
        try:
            video_source = int(args.video_source)
        except ValueError:
            video_source = args.video_source
        
        # Create derived paths from base output directory
        base_output_dir = args.output_dir.rstrip('/')
        videos_dir = os.path.join(base_output_dir, 'videos')
        images_dir = os.path.join(base_output_dir, 'images')
        db_path = os.path.join(base_output_dir, 'db', 'detections.db')
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

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
            image_capture_interval=args.image_interval
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