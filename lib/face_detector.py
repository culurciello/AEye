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
            return []
    
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