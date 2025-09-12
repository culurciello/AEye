#!/usr/bin/env python3

import sqlite3
import cv2
import numpy as np
import os
import shutil
import pickle
import argparse
import logging
from typing import List, Dict
from deepface import DeepFace
from tqdm import tqdm
import tensorflow as tf

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
        
        # Initialize face detection using DeepFace
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
        """Configure GPU usage for TensorFlow/DeepFace."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.use_gpu:
                # Configure GPU memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available and configured: {len(gpus)} GPU(s) detected")
                self.gpu_available = True
            else:
                if not gpus:
                    logger.info("No GPU detected, using CPU")
                else:
                    logger.info("GPU available but disabled by user, using CPU")
                # Force CPU usage
                tf.config.set_visible_devices([], 'GPU')
                self.gpu_available = False
        except Exception as e:
            logger.warning(f"GPU configuration failed, using CPU: {e}")
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
                face_encoding BLOB,
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
        """Initialize face detection using DeepFace with optimized settings."""
        self.detector_backend = 'retinaface'  # Better quality than opencv
        self.model_name = 'Facenet'
        self.distance_metric = 'cosine'
        
        logger.info(f"DeepFace initialized: detector={self.detector_backend}, model={self.model_name}")
    
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
            faces = self.detect_faces_in_crop(crop_bytes)
            
            # Save all faces from this detection
            for face_data in faces:
                self.save_face(detection_id, face_data, confidence)
                faces_extracted += 1
        
        source_conn.close()
        logger.info(f"Extracted {faces_extracted} faces from {len(person_detections)} detections")
        
        return faces_extracted
    
    def detect_faces_in_crop(self, crop_bytes: bytes) -> List[Dict]:
        """Detect faces in a single crop image using DeepFace."""
        # Convert bytes to image
        nparr = np.frombuffer(crop_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        face_data = []
        
        try:
            # Use DeepFace.extract_faces to get face regions with better detector
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                # enforce_detection=True,
                grayscale=False,
                align=True  # Face alignment for better results
            )
            
            for face_obj in face_objs:
                # Extract the face array from the face_obj dictionary
                face_array = face_obj['face']
                facial_area = face_obj.get('facial_area', {})
                face_confidence = face_obj.get('confidence', 0.0)
                
                # Filter out low confidence detections to reduce false positives
                if face_confidence < 0.95:  # High threshold for quality
                    logger.debug(f"Skipping low confidence face: {face_confidence:.3f}")
                    continue
                
                # Convert normalized face back to uint8 with proper RGB ordering
                face_array_uint8 = (face_array * 255).astype(np.uint8)
                
                # Ensure proper color format (DeepFace uses RGB, OpenCV uses BGR)
                if len(face_array_uint8.shape) == 3 and face_array_uint8.shape[2] == 3:
                    face_array_bgr = cv2.cvtColor(face_array_uint8, cv2.COLOR_RGB2BGR)
                else:
                    face_array_bgr = face_array_uint8
                
                # Convert face crop to bytes with high quality JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, face_buffer = cv2.imencode('.jpg', face_array_bgr, encode_param)
                face_bytes = face_buffer.tobytes()
                
                # Generate face embedding
                face_embedding = self.get_face_embedding(face_array_uint8)

                # Get bounding box from facial_area
                bbox_x = facial_area.get('x', 0)
                bbox_y = facial_area.get('y', 0)
                bbox_w = facial_area.get('w', face_array_uint8.shape[1])
                bbox_h = facial_area.get('h', face_array_uint8.shape[0])
                
                # Additional quality check - skip very small faces
                if bbox_w < 30 or bbox_h < 30:
                    logger.debug(f"Skipping small face: {bbox_w}x{bbox_h}")
                    continue
                
                face_data.append({
                    'crop': face_bytes,
                    'encoding': face_embedding,
                    'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
                    'confidence': face_confidence
                })
                
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")


        
        return face_data
    
    def get_face_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Generate face embedding using DeepFace."""
        try:
            # Use DeepFace.represent to get face embedding
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                detector_backend='skip',  # Skip detection since we already have the face
                enforce_detection=False
            )[0]['embedding']
            
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Error generating face embedding: {e}")
            # Return random embedding as fallback
            return np.random.random(128).astype(np.float32)
    
    def save_face(self, detection_id: int, face_data: Dict, confidence: float):
        """Save a face to the database."""
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Serialize face encoding
        encoding_bytes = pickle.dumps(face_data['encoding'])
        
        bbox = face_data['bbox']
        cursor.execute('''
            INSERT INTO faces 
            (detection_id, face_crop, face_encoding, confidence, 
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (detection_id, face_data['crop'], encoding_bytes, confidence,
              bbox[0], bbox[1], bbox[2], bbox[3]))
        
        conn.commit()
        conn.close()
    
    def group_faces_by_similarity(self, min_samples: int = 2):
        """Group faces by similarity using DeepFace verification."""
        logger.info("Starting face grouping...")
        
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Get all faces
        cursor.execute('SELECT id, face_crop, confidence FROM faces')
        faces = cursor.fetchall()
        
        if not faces:
            logger.warning("No faces found to group")
            return
        
        logger.info(f"Grouping {len(faces)} faces...")
        
        # Create temporary directory for face images
        temp_dir = f"/tmp/faces_{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save all face images to temporary files
            face_paths = {}
            for face_id, face_crop, confidence in tqdm(faces, desc="Preparing faces"):
                temp_path = os.path.join(temp_dir, f"face_{face_id}.jpg")
                with open(temp_path, 'wb') as f:
                    f.write(face_crop)
                face_paths[face_id] = temp_path
            
            # Group faces using pairwise verification
            face_groups = []
            processed_faces = set()
            
            for i, (face_id1, _, _) in enumerate(tqdm(faces, desc="Grouping faces")):
                if face_id1 in processed_faces:
                    continue
                
                current_group = [face_id1]
                processed_faces.add(face_id1)
                
                # Compare with remaining faces
                for j in range(i + 1, len(faces)):
                    face_id2, _, _ = faces[j]
                    
                    if face_id2 in processed_faces:
                        continue
                    
                    # Verify if faces are similar
                    if self.verify_faces(face_paths[face_id1], face_paths[face_id2]):
                        current_group.append(face_id2)
                        processed_faces.add(face_id2)
                
                # Only keep groups with minimum samples
                if len(current_group) >= min_samples:
                    face_groups.append(current_group)
                    logger.debug(f"Created group {len(face_groups)} with {len(current_group)} faces")
            
            # Update database with groups
            self.save_face_groups(face_groups)
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        logger.info(f"Created {len(face_groups)} face groups")
    
    def verify_faces(self, path1: str, path2: str) -> bool:
        """Verify if two faces belong to the same person."""
        try:
            result = DeepFace.verify(
                img1_path=path1,
                img2_path=path2,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False
            )
            return result['verified']
        except Exception as e:
            logger.debug(f"Face verification failed: {e}")
            return False
    
    def identify_group_with_known_faces(self, representative_face_id: int) -> str:
        """Try to identify a face group by matching against known faces."""
        if not self.known_faces:
            return None
        
        # Get the representative face image from database
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT face_crop FROM faces WHERE id = ?', (representative_face_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        face_crop_bytes = result[0]
        
        # Save the face crop to a temporary file
        temp_face_path = f"/tmp/temp_face_{representative_face_id}.jpg"
        try:
            with open(temp_face_path, 'wb') as f:
                f.write(face_crop_bytes)
            
            # Try to match against each known face
            for person_name, known_face_path in self.known_faces.items():
                try:
                    result = DeepFace.verify(
                        img1_path=temp_face_path,
                        img2_path=known_face_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        distance_metric=self.distance_metric,
                        enforce_detection=False
                    )
                    
                    if result['verified']:
                        logger.info(f"Face group matched to known person: {person_name} (confidence: {result.get('distance', 'N/A')})")
                        return person_name
                except Exception as e:
                    logger.debug(f"Error matching against {person_name}: {e}")
                    continue
        
        except Exception as e:
            logger.debug(f"Error in face identification: {e}")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_face_path)
            except:
                pass
        
        return None
    
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
    parser = argparse.ArgumentParser(description='Face detection and grouping for AEye database')
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

    args = parser.parse_args()
    
    # Auto-generate target database name if not provided
    if not args.target_db:
        source_dir = os.path.dirname(args.source_db)
        source_name = os.path.basename(args.source_db)
        name_without_ext = os.path.splitext(source_name)[0]
        args.target_db = os.path.join(source_dir, f"faces_{name_without_ext}.db")
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('facedetect.log')
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
            extracted_faces_dir = "extracted_faces/"
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