import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Tuple


class DetectionDatabase:
    def __init__(self, db_path: str = "detections.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the required schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                crop_of_object BLOB NOT NULL,
                original_video_link TEXT NOT NULL,
                frame_num_original_video INTEGER NOT NULL,
                caption TEXT DEFAULT NULL,
                embeddings BLOB DEFAULT NULL,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create tracks table for object tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                original_video_link TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                track_data TEXT NOT NULL,  -- JSON of bounding boxes per frame
                best_crop_detection_id INTEGER,  -- ID of detection with highest confidence
                avg_confidence REAL,
                detection_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (best_crop_detection_id) REFERENCES detections(id)
            )
        ''')
        
        # Check if we need to add new columns to existing table
        cursor.execute("PRAGMA table_info(detections)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'original_video_link' not in columns:
            cursor.execute('ALTER TABLE detections ADD COLUMN original_video_link TEXT')
        
        if 'frame_num_original_video' not in columns:
            cursor.execute('ALTER TABLE detections ADD COLUMN frame_num_original_video INTEGER')
        
        conn.commit()
        conn.close()
    
    def save_detection(self, object_type: str, timestamp: datetime, 
                      crop_image: bytes,
                      original_video_link: str, frame_num: int,
                      confidence: float, bbox: Tuple[int, int, int, int],
                      caption: str = None) -> int:
        """Save a detection to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (object_type, time, crop_of_object,
             original_video_link, frame_num_original_video, confidence, 
             bbox_x, bbox_y, bbox_width, bbox_height, caption)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (object_type, timestamp, crop_image,
              original_video_link, frame_num, confidence, *bbox, caption))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return detection_id
    
    def get_detection_by_id(self, detection_id: int) -> Optional[Tuple]:
        """Get a specific detection by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, object_type, time, crop_of_object, original_video_link, 
                   frame_num_original_video, caption, embeddings, confidence, 
                   bbox_x, bbox_y, bbox_width, bbox_height, created_at
            FROM detections WHERE id = ?
        ''', (detection_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    
    @staticmethod
    def bytes_to_image(image_bytes: bytes):
        """Convert bytes back to image array."""
        import cv2
        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def image_to_bytes(image):
        """Convert image array to bytes."""
        import cv2
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    def save_track(self, object_type: str, original_video_link: str, 
                   start_frame: int, end_frame: int, start_time: datetime, 
                   end_time: datetime, track_data: dict, 
                   best_crop_detection_id: int, avg_confidence: float, 
                   detection_count: int) -> int:
        """Save a track to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tracks 
            (object_type, original_video_link, start_frame, end_frame, 
             start_time, end_time, track_data, best_crop_detection_id, 
             avg_confidence, detection_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (object_type, original_video_link, start_frame, end_frame,
              start_time, end_time, json.dumps(track_data), 
              best_crop_detection_id, avg_confidence, detection_count))
        
        track_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return track_id
    
    def get_all_tracks(self) -> List[Tuple]:
        """Get all tracks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tracks ORDER BY start_time DESC')
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_track_by_id(self, track_id: int) -> Optional[Tuple]:
        """Get a specific track by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tracks WHERE id = ?', (track_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result
    
