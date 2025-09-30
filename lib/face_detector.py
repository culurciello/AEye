import cv2
import numpy as np
import logging
import os
import glob
from datetime import datetime

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, use_gpu: bool = True, db_manager=None, known_faces_dir: str = "data/faces-known"):
        self.use_gpu = use_gpu
        self.db_manager = db_manager
        self.face_app = None
        self.known_faces_dir = known_faces_dir
        self.known_face_embeddings = {}
        self.recognition_threshold = 0.4  # Similarity threshold for face recognition

    def init_face_detector(self):
        """Initialize and warm up face detection using InsightFace."""
        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace library not available. Face detection disabled.")
            self.face_app = None
            return

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

            # Load known faces after initializing the face detector
            self.load_known_faces()

        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
            self.face_app = None

    def load_known_faces(self):
        """Load known faces from the faces-known directory and compute embeddings."""
        if not self.face_app:
            logger.warning("Face detector not initialized, cannot load known faces")
            return

        if not os.path.exists(self.known_faces_dir):
            logger.warning(f"Known faces directory not found: {self.known_faces_dir}")
            return

        self.known_face_embeddings = {}

        # Iterate through person directories
        for person_dir in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_dir)

            if not os.path.isdir(person_path) or person_dir.startswith('.'):
                continue

            logger.info(f"Loading known faces for: {person_dir}")
            person_embeddings = []

            # Load all images for this person
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for pattern in image_patterns:
                for image_path in glob.glob(os.path.join(person_path, pattern)):
                    try:
                        # Load and process image
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning(f"Could not load image: {image_path}")
                            continue

                        # Detect faces in the image
                        faces = self.face_app.get(image)

                        if len(faces) == 0:
                            logger.warning(f"No face detected in: {image_path}")
                            continue
                        elif len(faces) > 1:
                            logger.warning(f"Multiple faces detected in: {image_path}, using the first one")

                        # Use the first (or only) face
                        face = faces[0]
                        embedding = face.embedding

                        # Normalize embedding
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                            person_embeddings.append(embedding)
                            logger.debug(f"Loaded embedding for {person_dir} from {os.path.basename(image_path)}")

                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")

            if person_embeddings:
                # Store average embedding for this person
                self.known_face_embeddings[person_dir] = np.mean(person_embeddings, axis=0)
                logger.info(f"Loaded {len(person_embeddings)} face(s) for {person_dir}")
            else:
                logger.warning(f"No valid faces found for {person_dir}")

        logger.info(f"Loaded known faces for {len(self.known_face_embeddings)} people: {list(self.known_face_embeddings.keys())}")

    def recognize_face(self, embedding: np.ndarray):
        """Recognize a face by comparing its embedding to known faces.

        Args:
            embedding: Normalized face embedding

        Returns:
            Tuple of (person_name, confidence) or (None, 0) if no match
        """
        if not self.known_face_embeddings:
            return None, 0

        best_match = None
        best_similarity = 0

        for person_name, known_embedding in self.known_face_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, known_embedding)

            if similarity > best_similarity and similarity > self.recognition_threshold:
                best_similarity = similarity
                best_match = person_name

        return best_match, best_similarity


    # def detect_faces_in_frame(self, frame: np.ndarray, frame_time: datetime, motion_event_id: int):
    #     """Detect faces in a single frame and store results."""
    #     if not self.face_app:
    #         return 0

    #     try:
    #         faces = self.face_app.get(frame)
    #         face_count = 0

    #         for face in faces:
    #             bbox = face.bbox.astype(int)
    #             confidence = face.det_score

    #             # Skip low confidence detections
    #             if confidence < 0.7:
    #                 continue

    #             x1, y1, x2, y2 = bbox
    #             bbox_w = x2 - x1
    #             bbox_h = y2 - y1

    #             # Skip very small faces
    #             if bbox_w < 30 or bbox_h < 30:
    #                 continue

    #             # Extract face crop
    #             face_crop = frame[y1:y2, x1:x2]

    #             # Convert to bytes
    #             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    #             _, face_buffer = cv2.imencode('.jpg', face_crop, encode_param)
    #             face_bytes = face_buffer.tobytes()

    #             # Get normalized embedding
    #             embedding = face.embedding
    #             norm = np.linalg.norm(embedding)
    #             if norm > 0:
    #                 embedding = embedding / norm

    #             # Try to recognize the face
    #             known_person, recognition_confidence = self.recognize_face(embedding)

    #             if known_person:
    #                 logger.info(f"Recognized face: {known_person} (confidence: {recognition_confidence:.3f})")

    #             # Store in database
    #             if self.db_manager:
    #                 self.db_manager.store_face_detection(
    #                     motion_event_id, frame_time, face_bytes, embedding,
    #                     confidence, x1, y1, bbox_w, bbox_h, known_person, recognition_confidence
    #                 )

    #             face_count += 1

    #         return face_count

    #     except Exception as e:
    #         logger.error(f"Error detecting faces: {e}")
    #         return 0
    

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
                    confidence = float(face.det_score)

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

                    # Try to recognize the face
                    known_person, recognition_confidence = self.recognize_face(embedding)

                    if known_person:
                        logger.info(f"Recognized face: {known_person} (confidence: {recognition_confidence:.3f})")

                    # Store in database with full frame coordinates
                    if self.db_manager:
                        self.db_manager.store_face_detection(
                            motion_event_id, frame_time, face_bytes, embedding,
                            confidence, full_frame_x1, full_frame_y1, bbox_w, bbox_h, known_person, recognition_confidence
                        )

                    total_face_count += 1

            return total_face_count

        except Exception as e:
            logger.error(f"Error detecting faces in person crops: {e}")
            return 0
    
