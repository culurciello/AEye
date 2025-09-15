#!/usr/bin/env python3

import sqlite3
import cv2
import numpy as np
import os
import pickle
import argparse
import logging
from typing import List, Dict
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Setup logger
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, source_db_path: str, target_db_path: str, use_gpu: bool = True):
        """
        Initialize face detector.
        
        Args:
            source_db_path: Path to source detections database
            target_db_path: Path to target faces database
            use_gpu: Whether to use GPU if available
        """
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path
        self.use_gpu = use_gpu
        
        # Configure GPU usage
        self.configure_gpu()
        
        # Initialize face detection using PyTorch RetinaFace
        self.init_face_detector()
        
        # Load known faces for naming groups
        self.known_faces = self.load_known_faces()
        
        # Initialize face database
        self.init_face_database()
    
    def load_known_faces(self) -> Dict[str, str]:
        """Load known faces from data/faces-known/ directory for group naming."""
        known_faces = {}
        known_faces_dir = "data/faces-known"
        
        if not os.path.exists(known_faces_dir):
            logger.info(f"Known faces directory not found: {known_faces_dir}")
            return known_faces
        
        try:
            for person_dir in os.listdir(known_faces_dir):
                person_path = os.path.join(known_faces_dir, person_dir)
                if not os.path.isdir(person_path) or person_dir.startswith('.'):
                    continue
                
                # Find the first image file in the person's directory
                for filename in os.listdir(person_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(person_path, filename)
                        known_faces[person_dir] = image_path
                        logger.debug(f"Loaded known face: {person_dir} -> {image_path}")
                        break
            
            logger.info(f"Loaded {len(known_faces)} known faces: {list(known_faces.keys())}")
        except Exception as e:
            logger.warning(f"Error loading known faces: {e}")
        
        return known_faces
    
    def configure_gpu(self):
        """Configure GPU usage for InsightFace."""
        try:
            if self.use_gpu:
                # InsightFace will automatically detect and use available GPU
                self.ctx_id = 0  # GPU context
                logger.info("GPU enabled for InsightFace")
                self.gpu_available = True
            else:
                self.ctx_id = -1  # CPU context
                logger.info("Using CPU for InsightFace")
                self.gpu_available = False
        except Exception as e:
            logger.warning(f"GPU configuration failed, using CPU: {e}")
            self.ctx_id = -1
            self.gpu_available = False
    
    def init_face_database(self):
        """Initialize the faces database with required schema."""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                face_crop BLOB NOT NULL,
                face_embeddings BLOB,
                similarity_group INTEGER,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        ''')
        
        # Create face groups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT,
                representative_face_id INTEGER,
                face_count INTEGER DEFAULT 0,
                avg_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (representative_face_id) REFERENCES faces(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized face database: {self.target_db_path}")
    
    def init_face_detector(self):
        """Initialize face detection using InsightFace."""
        # Initialize InsightFace with default models (includes detection)
        try:
            self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            logger.info(f"InsightFace initialized with ctx_id: {self.ctx_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize with default models: {e}")
            # Try with buffalo_l model which is more commonly available
            try:
                self.app = FaceAnalysis(name="buffalo_l")
                self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
                logger.info(f"InsightFace initialized with buffalo_l model, ctx_id: {self.ctx_id}")
            except Exception as e2:
                logger.error(f"Failed to initialize InsightFace: {e2}")
                raise e2
    
    def extract_faces_from_detections(self):
        """Extract faces from all person detections in the source database."""
        logger.info("Starting face extraction from detections...")
        
        # Connect to source database
        source_conn = sqlite3.connect(self.source_db_path)
        source_cursor = source_conn.cursor()
        
        # Get all person detections
        source_cursor.execute('''
            SELECT id, crop_of_object, confidence
            FROM detections 
            WHERE object_type = 'person'
        ''')
        
        person_detections = source_cursor.fetchall()
        logger.info(f"Found {len(person_detections)} person detections")
        
        faces_extracted = 0
        
        # Process each detection with progress bar
        for detection in tqdm(person_detections, desc="Processing detections"):
            detection_id = detection[0]
            crop_bytes = detection[1]
            confidence = detection[2]
            
            # Convert crop bytes to image and detect faces
            faces = self.detect_faces_in_crop(crop_bytes, debug_save_path="debug")
            
            # Save all faces from this detection
            for face_data in faces:
                self.save_face(detection_id, face_data, confidence)
                faces_extracted += 1
        
        source_conn.close()
        logger.info(f"Extracted {faces_extracted} faces from {len(person_detections)} detections")
        
        return faces_extracted
    
    def detect_faces_in_crop(self, crop_bytes: bytes, debug_save_path: str = None) -> List[Dict]:
        """Detect faces in a single crop image using InsightFace."""
        # Convert bytes to image
        nparr = np.frombuffer(crop_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        # Debug: save first few images to see what we're processing
        if debug_save_path and not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if debug_save_path and self._debug_count < 5:
            debug_path = f"{debug_save_path}_crop_{self._debug_count}.jpg"
            cv2.imwrite(debug_path, image)
            logger.info(f"Debug: saved crop to {debug_path}")
            self._debug_count += 1
        
        face_data = []
        
        try:
            # Use InsightFace for face detection and recognition
            logger.debug(f"Processing image of shape: {image.shape}")
            faces = self.app.get(image)
            
            if not faces:
                logger.debug("No faces detected in this image")
                return []
            
            logger.debug(f"Found {len(faces)} faces in image")
            
            for face in faces:
                # Extract face information
                bbox = face.bbox  # [x1, y1, x2, y2]
                confidence = face.det_score  # Detection confidence
                
                # Filter out low confidence detections (lowered threshold for debugging)
                if confidence < 0.75:  # Lower threshold to catch more faces
                    logger.debug(f"Skipping low confidence face: {confidence:.3f}")
                    continue
                else:
                    logger.info(f"Found face with confidence: {confidence:.3f}")
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = bbox.astype(int)
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                
                # Additional quality check - skip very small faces
                if bbox_w < 30 or bbox_h < 30:
                    logger.debug(f"Skipping small face: {bbox_w}x{bbox_h}")
                    continue
                
                # Extract face crop from original image
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # Convert face crop to bytes with high quality JPEG (no resizing)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, face_buffer = cv2.imencode('.jpg', face_crop, encode_param)
                face_bytes = face_buffer.tobytes()
                
                # Get the raw embedding and normalize it manually to ensure proper normalization
                face_embedding = face.embedding
                # Normalize the embedding manually
                norm = np.linalg.norm(face_embedding)
                if norm > 0:
                    face_embedding = face_embedding / norm
                
                logger.debug(f"Embedding norm after normalization: {np.linalg.norm(face_embedding):.3f}")
                
                face_data.append({
                    'crop': face_bytes,
                    'embedding': face_embedding,
                    'bbox': (x1, y1, bbox_w, bbox_h),
                    'confidence': confidence
                })
                
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
        
        return face_data
    
    # def get_face_embedding(self, face_img: np.ndarray) -> np.ndarray:
    #     """Generate face embedding using InsightFace (this method is now unused as embeddings come from app.get)."""
    #     # This method is now unused since InsightFace provides embeddings directly
    #     # Keeping for compatibility but returning empty array
    #     logger.warning("get_face_embedding called but embeddings should come from InsightFace directly")
    #     return np.random.random(512).astype(np.float32)
    
    def save_face(self, detection_id: int, face_data: Dict, confidence: float):
        """Save a face to the database."""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()

        # Serialize face embedding
        embedding_bytes = pickle.dumps(face_data['embedding'])

        bbox = face_data['bbox']
        cursor.execute('''
            INSERT INTO faces 
            (detection_id, face_crop, face_embeddings, confidence, 
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (detection_id, face_data['crop'], embedding_bytes, confidence,
              bbox[0], bbox[1], bbox[2], bbox[3]))
        
        conn.commit()
        conn.close()
    
    def group_faces_by_similarity(self, min_samples: int = 2):
        """Group faces by similarity using cosine distance between embeddings."""
        logger.info("Starting face grouping...")
        
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Get all faces with embeddings
        cursor.execute('SELECT id, face_embeddings, confidence FROM faces WHERE face_embeddings IS NOT NULL')
        faces = cursor.fetchall()
        
        if not faces:
            logger.warning("No faces with embeddings found to group")
            return
        
        logger.info(f"Grouping {len(faces)} faces...")
        
        # Load embeddings
        face_embeddings = []
        face_ids = []
        
        for face_id, embedding_bytes, confidence in faces:
            try:
                embedding = pickle.loads(embedding_bytes)
                face_embeddings.append(embedding)
                face_ids.append(face_id)
            except Exception as e:
                logger.debug(f"Error loading embedding for face {face_id}: {e}")
                continue
        
        if not face_embeddings:
            logger.warning("No valid embeddings found")
            return
        
        face_embeddings = np.array(face_embeddings)
        
        # Group faces using distance-based similarity (similar to the example code pattern)
        face_groups = []
        processed_faces = set()
        distance_threshold = 1.0 # Distance threshold for normalized embeddings (0.0=identical, 2.0=opposite)
        
        for i, embedding1 in enumerate(tqdm(face_embeddings, desc="Grouping faces")):
            face_id1 = face_ids[i]
            
            if face_id1 in processed_faces:
                continue
            
            current_group = [face_id1]
            processed_faces.add(face_id1)
            
            # Compare with remaining faces
            for j in range(i + 1, len(face_embeddings)):
                face_id2 = face_ids[j]
                
                if face_id2 in processed_faces:
                    continue
                
                embedding2 = face_embeddings[j]
                
                # Calculate distance between embeddings
                distance = np.linalg.norm(embedding1 - embedding2)
                
                # Debug: Print some embedding stats for first few comparisons
                if i < 3 and j < i + 3:
                    print(f"Face {face_id1} vs {face_id2}: distance={distance:.3f}")
                    print(f"  Embedding1 norm: {np.linalg.norm(embedding1):.3f}")
                    print(f"  Embedding2 norm: {np.linalg.norm(embedding2):.3f}")
                    print(f"  Embedding1 shape: {embedding1.shape}")
                    print(f"  Embedding1 mean: {np.mean(embedding1):.3f}")
                    print("---")
                
                if distance < distance_threshold:
                    current_group.append(face_id2)
                    processed_faces.add(face_id2)
            
            # Only keep groups with minimum samples
            if len(current_group) >= min_samples:
                face_groups.append(current_group)
                logger.debug(f"Created group {len(face_groups)} with {len(current_group)} faces")
        
        # Update database with groups
        self.save_face_groups(face_groups)
        
        logger.info(f"Created {len(face_groups)} face groups")
    
    def cosine_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        normalized1 = embedding1 / norm1
        normalized2 = embedding2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(normalized1, normalized2)
        
        # Convert to distance (0 = identical, 2 = opposite)
        distance = 1 - similarity
        
        return distance
    
    def verify_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """Verify if two face embeddings belong to the same person."""
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance < 1.0  # Same threshold as grouping
    
    def identify_group_with_known_faces(self, representative_face_id: int) -> str:
        """Try to identify a face group by matching against known faces."""
        if not self.known_faces:
            return None
        
        # Get the representative face embedding from database
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT face_embeddings FROM faces WHERE id = ?', (representative_face_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result or not result[0]:
            return None
        
        try:
            representative_embedding = pickle.loads(result[0])
        except:
            return None
        
        # Try to match against each known face
        for person_name, known_face_path in self.known_faces.items():
            try:
                # Load and process known face
                known_img = cv2.imread(known_face_path)
                if known_img is None:
                    continue
                
                # Detect faces in known image
                known_faces = self.detect_faces_in_known_image(known_img)
                
                if not known_faces:
                    continue
                
                # Use the first detected face
                known_embedding = known_faces[0]['embedding']
                
                # Compare embeddings
                if self.verify_faces(representative_embedding, known_embedding):
                    distance = np.linalg.norm(representative_embedding - known_embedding)
                    logger.info(f"Face group matched to known person: {person_name} (distance: {distance:.3f})")
                    return person_name
                    
            except Exception as e:
                logger.debug(f"Error matching against {person_name}: {e}")
                continue
        
        return None
    
    def detect_faces_in_known_image(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in a known reference image."""
        try:
            faces = self.app.get(image)
            
            if not faces:
                return []
            
            result_faces = []
            for face in faces:
                # Use the normalized embedding from InsightFace
                embedding = face.normed_embedding
                result_faces.append({'embedding': embedding})
            
            return result_faces
            
        except Exception as e:
            logger.debug(f"Error detecting faces in known image: {e}")
            return []
    
    def save_face_groups(self, face_groups: List[List[int]]):
        """Save face groups to database."""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        for group_idx, group_faces in enumerate(face_groups):
            # Update faces with group labels
            for face_id in group_faces:
                cursor.execute(
                    'UPDATE faces SET similarity_group = ? WHERE id = ?',
                    (group_idx, face_id)
                )
            
            # Get group statistics
            cursor.execute('''
                SELECT id, confidence FROM faces 
                WHERE id IN ({}) ORDER BY confidence DESC
            '''.format(','.join(['?'] * len(group_faces))), group_faces)
            
            group_data = cursor.fetchall()
            representative_face_id = group_data[0][0]
            avg_confidence = sum(row[1] for row in group_data) / len(group_data)
            
            # Try to identify the group with known faces
            identified_name = self.identify_group_with_known_faces(representative_face_id)
            group_name = identified_name if identified_name else f"Unknown_Group_{group_idx}"
            
            # Create face group entry
            cursor.execute('''
                INSERT INTO face_groups 
                (group_name, representative_face_id, face_count, avg_confidence)
                VALUES (?, ?, ?, ?)
            ''', (group_name, representative_face_id, len(group_faces), avg_confidence))
        
        conn.commit()
        conn.close()
    
    def get_face_statistics(self) -> Dict:
        """Get statistics about faces and groups."""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Total faces
        cursor.execute('SELECT COUNT(*) FROM faces')
        total_faces = cursor.fetchone()[0]
        
        # Grouped faces
        cursor.execute('SELECT COUNT(*) FROM faces WHERE similarity_group IS NOT NULL')
        grouped_faces = cursor.fetchone()[0]
        
        # Face groups
        cursor.execute('SELECT COUNT(*) FROM face_groups')
        total_groups = cursor.fetchone()[0]
        
        # Group sizes
        cursor.execute('''
            SELECT face_count, COUNT(*) as group_count
            FROM face_groups 
            GROUP BY face_count 
            ORDER BY face_count DESC
        ''')
        group_sizes = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_faces': total_faces,
            'grouped_faces': grouped_faces,
            'ungrouped_faces': total_faces - grouped_faces,
            'total_groups': total_groups,
            'group_sizes': group_sizes
        }


def main():
    parser = argparse.ArgumentParser(description='Face detection and grouping for AEye database using InsightFace')
    parser.add_argument('--source-db', required=True,
                       help='Path to source detections database')
    parser.add_argument('--target-db', 
                       help='Path to target faces database (default: auto-generated from source DB)')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum samples for face grouping (default: 2)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip face extraction, only perform grouping')
    parser.add_argument('--skip-grouping', action='store_true',
                       help='Skip face grouping, only perform extraction')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage, force CPU-only processing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output and save sample images')

    args = parser.parse_args()
    
    # Auto-generate target database name if not provided
    if not args.target_db:
        source_dir = os.path.dirname(args.source_db)
        source_name = os.path.basename(args.source_db)
        name_without_ext = os.path.splitext(source_name)[0]
        args.target_db = os.path.join(source_dir, f"faces_{name_without_ext}.db")
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else args.log_level
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('facedetect2.log')
        ]
    )
    
    if not os.path.exists(args.source_db):
        logger.error(f"Source database not found: {args.source_db}")
        return 1
    
    # Initialize face detector
    use_gpu = not args.no_gpu  # Invert the no_gpu flag
    detector = FaceDetector(args.source_db, args.target_db, use_gpu=use_gpu)
    
    try:
        if not args.skip_extraction:
            # Extract faces from detections
            faces_count = detector.extract_faces_from_detections()
            logger.info(f"Successfully extracted {faces_count} faces")

            # Save all faces to a folder to see they are extracted correctly
            extracted_faces_dir = "extracted_faces2/"
            os.makedirs(extracted_faces_dir, exist_ok=True)
            conn = sqlite3.connect(args.target_db)
            cursor = conn.cursor()
            cursor.execute('SELECT id, face_crop FROM faces')
            faces = cursor.fetchall()
            for face_id, face_crop in faces:
                face_path = os.path.join(extracted_faces_dir, f"face_{face_id}.jpg")
                with open(face_path, 'wb') as f:
                    f.write(face_crop)
            conn.close()
            logger.info(f"Extracted faces saved to directory: {extracted_faces_dir}")

        if not args.skip_grouping:
            # Group faces by similarity
            detector.group_faces_by_similarity(min_samples=args.min_samples)

        # Print statistics
        stats = detector.get_face_statistics()
        logger.info("\n=== Face Detection Statistics ===")
        logger.info(f"Total faces: {stats['total_faces']}")
        logger.info(f"Grouped faces: {stats['grouped_faces']}")
        logger.info(f"Ungrouped faces: {stats['ungrouped_faces']}")
        logger.info(f"Total groups: {stats['total_groups']}")
        
        if stats['group_sizes']:
            logger.info("\nGroup size distribution:")
            for size, count in stats['group_sizes'][:10]:  # Top 10
                logger.info(f"  {size} faces: {count} groups")

        logger.info(f"Face detection and grouping completed successfully!")
        logger.info(f"Results saved to: {args.target_db}")

    except Exception as e:
        logger.error(f"Error during face detection: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())