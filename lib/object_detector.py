import numpy as np
import cv2
import logging
from datetime import datetime

# YOLO object detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: (x, y, w, h) format bounding boxes

    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2) format
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

class ObjectDetector:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.yolo_model = None
        self.active_tracks = {}  # Dict to store active tracks per motion event
        self.track_iou_threshold = 0.3  # IoU threshold for track association
        self.max_track_age = 5  # Maximum frames without detection before track ends

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

    def start_tracking_session(self, motion_event_id: int):
        """Start a new tracking session for a motion event."""
        self.active_tracks[motion_event_id] = {
            'tracks': [],  # List of active track dictionaries
            'next_track_id': 1,
            'frame_count': 0
        }

    def end_tracking_session(self, motion_event_id: int):
        """End tracking session and finalize all tracks."""
        if motion_event_id in self.active_tracks:
            session = self.active_tracks[motion_event_id]

            # Finalize all remaining tracks
            for track in session['tracks']:
                if track['detection_count'] > 1:  # Only keep tracks with multiple detections
                    self._finalize_track(track)

            # Update motion event track count
            self.db_manager.update_motion_event_track_count(motion_event_id)

            # Clear the session
            del self.active_tracks[motion_event_id]

    def _finalize_track(self, track):
        """Finalize a track by updating it in the database."""
        avg_confidence = sum(track['confidences']) / len(track['confidences'])

        self.db_manager.update_object_track(
            track['track_id'],
            track['end_time'],
            track['last_bbox'],
            track['detection_count'],
            avg_confidence
        )

    def _find_matching_track(self, detection, active_tracks, class_name):
        """Find the best matching track for a detection."""
        best_match = None
        best_iou = 0

        detection_bbox = (detection['bbox_x'], detection['bbox_y'],
                         detection['bbox_w'], detection['bbox_h'])

        for track in active_tracks:
            if track['class_name'] != class_name:
                continue

            if track['age'] > self.max_track_age:
                continue

            iou = calculate_iou(detection_bbox, track['last_bbox'])
            if iou > self.track_iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = track

        return best_match

    def _create_new_track(self, motion_event_id: int, detection, frame_time, object_crop_bytes):
        """Create a new track for a detection."""
        session = self.active_tracks[motion_event_id]

        bbox = (detection['bbox_x'], detection['bbox_y'],
               detection['bbox_w'], detection['bbox_h'])

        # Create track in database
        track_id = self.db_manager.create_object_track(
            motion_event_id,
            detection['class_name'],
            frame_time,
            bbox,
            detection['confidence'],
            object_crop_bytes
        )

        # Create track object
        track = {
            'track_id': track_id,
            'class_name': detection['class_name'],
            'start_time': frame_time,
            'end_time': frame_time,
            'last_bbox': bbox,
            'detection_count': 1,
            'confidences': [detection['confidence']],
            'age': 0  # Frames since last detection
        }

        session['tracks'].append(track)
        return track

    def _update_track(self, track, detection, frame_time):
        """Update an existing track with a new detection."""
        track['end_time'] = frame_time
        track['last_bbox'] = (detection['bbox_x'], detection['bbox_y'],
                             detection['bbox_w'], detection['bbox_h'])
        track['detection_count'] += 1
        track['confidences'].append(detection['confidence'])
        track['age'] = 0  # Reset age since we have a new detection

    def detect_objects_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
        """Detect objects in a single frame using YOLO and store results with tracking."""
        if not self.yolo_model:
            return 0, []

        # Initialize tracking session if not exists
        if motion_event_id not in self.active_tracks:
            self.start_tracking_session(motion_event_id)

        session = self.active_tracks[motion_event_id]
        session['frame_count'] += 1

        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            current_detections = []
            person_bboxes = []

            # Process YOLO results to create detection list
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

                        # Extract object crop
                        object_crop_bytes = None
                        try:
                            # Ensure coordinates are within frame bounds
                            frame_h, frame_w = frame.shape[:2]
                            bbox_x = max(0, min(bbox_x, frame_w - 1))
                            bbox_y = max(0, min(bbox_y, frame_h - 1))
                            bbox_x2 = min(bbox_x + bbox_w, frame_w)
                            bbox_y2 = min(bbox_y + bbox_h, frame_h)

                            # Extract crop
                            crop = frame[bbox_y:bbox_y2, bbox_x:bbox_x2]

                            if crop.size > 0:
                                # Resize crop to standardized size (max 200x200, maintain aspect ratio)
                                h, w = crop.shape[:2]
                                if h > 200 or w > 200:
                                    scale = min(200/w, 200/h)
                                    new_w = int(w * scale)
                                    new_h = int(h * scale)
                                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

                                # Encode as JPEG bytes
                                success, crop_encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if success:
                                    object_crop_bytes = crop_encoded.tobytes()

                        except Exception as crop_e:
                            logger.warning(f"Failed to extract object crop: {crop_e}")

                        detection = {
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox_x': bbox_x,
                            'bbox_y': bbox_y,
                            'bbox_w': bbox_w,
                            'bbox_h': bbox_h,
                            'crop_bytes': object_crop_bytes
                        }
                        current_detections.append(detection)

                        # If this is a person, save the bounding box for face detection
                        if class_name.lower() == 'person':
                            person_bboxes.append({
                                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
                                'confidence': confidence
                            })

            # Track objects
            self._process_detections_with_tracking(motion_event_id, current_detections, frame_time)

            # Age existing tracks
            for track in session['tracks']:
                track['age'] += 1

            # Remove old tracks (finalize tracks that haven't been seen for too long)
            tracks_to_remove = []
            for i, track in enumerate(session['tracks']):
                if track['age'] > self.max_track_age:
                    if track['detection_count'] > 1:  # Only finalize tracks with multiple detections
                        self._finalize_track(track)
                    tracks_to_remove.append(i)

            # Remove finalized tracks in reverse order to maintain indices
            for i in reversed(tracks_to_remove):
                session['tracks'].pop(i)

            return len(current_detections), person_bboxes

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return 0, []

    def _process_detections_with_tracking(self, motion_event_id: int, detections, frame_time):
        """Process detections with tracking algorithm."""
        if motion_event_id not in self.active_tracks:
            return

        session = self.active_tracks[motion_event_id]
        active_tracks = session['tracks']

        for detection in detections:
            class_name = detection['class_name']

            # Find matching track
            matching_track = self._find_matching_track(detection, active_tracks, class_name)

            if matching_track:
                # Update existing track
                self._update_track(matching_track, detection, frame_time)
                track_id = matching_track['track_id']
            else:
                # Create new track
                new_track = self._create_new_track(motion_event_id, detection, frame_time, detection['crop_bytes'])
                track_id = new_track['track_id']

            # Store detection with track association
            self.db_manager.store_object_detection(
                motion_event_id, frame_time, detection['class_name'], detection['confidence'],
                detection['bbox_x'], detection['bbox_y'], detection['bbox_w'], detection['bbox_h'],
                detection['crop_bytes'], track_id
            )

