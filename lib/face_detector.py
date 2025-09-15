import cv2
import numpy as np
import logging
import sqlite3
import pickle
from datetime import datetime
from typing import Optional
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, use_gpu: bool = True, db_path: str = "data/db/detections.db"):
        self.use_gpu = use_gpu
        self.db_path = db_path
        self.face_app = None

    def init_face_detector(self):
        """Initialize and warm up face detection using InsightFace."""
        try:
            logger.info("Initializing InsightFace neural networks...")
            self.face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            ctx_id = 0 if self.use_gpu else -1
            self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
            # Warm up the neural network with dummy data
            logger.info("Warming up face detection neural networks...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame.fill(128)  # Fill with gray color
    
            # Add some noise to make it more realistic
            noise = np.random.randint(0, 50, dummy_frame.shape, dtype=np.uint8)
            dummy_frame = cv2.add(dummy_frame, noise)
    
            # Run several warm-up inferences
            for i in range(3):
                _ = self.face_app.get(dummy_frame)
                logger.debug(f"Face detection warm-up iteration {i+1}/3 completed")
    
            logger.info(f"InsightFace initialized and warmed up (ctx_id: {ctx_id})")
    
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_app = None



    def detect_faces_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
        """Detect faces in a single frame and store results."""
        if not self.face_app:
            return 0

        try:
            faces = self.face_app.get(frame)
            face_count = 0

            for face in faces:
                bbox = face.bbox.astype(int)
                confidence = face.det_score

                # Skip low confidence detections
                if confidence < 0.7:
                    continue

                x1, y1, x2, y2 = bbox
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                # Skip very small faces
                if bbox_w < 30 or bbox_h < 30:
                    continue

                # Extract face crop
                face_crop = frame[y1:y2, x1:x2]

                # Convert to bytes
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, face_buffer = cv2.imencode('.jpg', face_crop, encode_param)
                face_bytes = face_buffer.tobytes()

                # Get normalized embedding
                embedding = face.embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Store in database
                self.store_face_detection(
                    motion_event_id, frame_time, face_bytes, embedding,
                    confidence, x1, y1, bbox_w, bbox_h
                )

                face_count += 1

            return face_count

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return 0

    def detect_faces_in_person_crops(self, frame: np.ndarray, person_bboxes: list, frame_time: datetime, motion_event_id: int):
        """Detect faces within person bounding boxes and store results."""
        if not self.face_app or not person_bboxes:
            return 0

        total_face_count = 0

        try:
            for person_data in person_bboxes:
                bbox_x, bbox_y, bbox_w, bbox_h = person_data['bbox']

                # Add some padding to the person crop for better face detection
                padding = 20
                crop_x1 = max(0, bbox_x - padding)
                crop_y1 = max(0, bbox_y - padding)
                crop_x2 = min(frame.shape[1], bbox_x + bbox_w + padding)
                crop_y2 = min(frame.shape[0], bbox_y + bbox_h + padding)

                # Extract person crop
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # Skip if crop is too small
                if person_crop.shape[0] < 60 or person_crop.shape[1] < 60:
                    continue

                # Run face detection on the person crop
                faces = self.face_app.get(person_crop)

                for face in faces:
                    bbox = face.bbox.astype(int)
                    confidence = face.det_score

                    # Skip low confidence detections
                    if confidence < 0.7:
                        continue

                    # Adjust face coordinates to full frame coordinates
                    face_x1, face_y1, face_x2, face_y2 = bbox

                    # Convert crop coordinates back to full frame coordinates
                    full_frame_x1 = face_x1 + crop_x1
                    full_frame_y1 = face_y1 + crop_y1
                    full_frame_x2 = face_x2 + crop_x1
                    full_frame_y2 = face_y2 + crop_y1

                    bbox_w = full_frame_x2 - full_frame_x1
                    bbox_h = full_frame_y2 - full_frame_y1

                    # Skip very small faces
                    if bbox_w < 30 or bbox_h < 30:
                        continue

                    # Extract face crop from original frame
                    face_crop = frame[full_frame_y1:full_frame_y2, full_frame_x1:full_frame_x2]

                    # Convert to bytes
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    _, face_buffer = cv2.imencode('.jpg', face_crop, encode_param)
                    face_bytes = face_buffer.tobytes()

                    # Get normalized embedding
                    embedding = face.embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    # Store in database with full frame coordinates
                    self.store_face_detection(
                        motion_event_id, frame_time, face_bytes, embedding,
                        confidence, full_frame_x1, full_frame_y1, bbox_w, bbox_h
                    )

                    total_face_count += 1

            return total_face_count

        except Exception as e:
            logger.error(f"Error detecting faces in person crops: {e}")
            return 0
    
    def store_face_detection(self, motion_event_id: int, frame_time: datetime,
                           face_bytes: bytes, embedding: np.ndarray, confidence: float,
                           x: int, y: int, w: int, h: int):
        """Store face detection in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        embedding_bytes = pickle.dumps(embedding)
    
        cursor.execute('''
            INSERT INTO face_detections
            (motion_event_id, frame_timestamp, face_crop, face_embedding, confidence,
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, face_bytes, embedding_bytes, confidence,
              x, y, w, h))
    
        conn.commit()
        conn.close()