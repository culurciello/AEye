import cv2
import numpy as np
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

class ObjectDetector:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.yolo_model = None

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

    def detect_objects_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
        """Detect objects in a single frame using YOLO and store results."""
        if not self.yolo_model:
            return 0, []

        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False)
            object_count = 0
            person_bboxes = []

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
                        self.db_manager.store_object_detection(
                            motion_event_id, frame_time, class_name, confidence,
                            bbox_x, bbox_y, bbox_w, bbox_h
                        )

                        object_count += 1

                        # If this is a person, save the bounding box for face detection
                        if class_name.lower() == 'person':
                            person_bboxes.append({
                                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
                                'confidence': confidence
                            })

            return object_count, person_bboxes

        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return 0, []

