import sqlite3
import logging
import pickle
import os
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
from .video_processor import VideoSegment

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the AEye system."""

    def __init__(self, db_path: str):
        """Initialize database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir:  # Only create directory if there is one
            os.makedirs(db_dir, exist_ok=True)

        # Initialize database schema
        self.init_database()

    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database schema with all required tables."""
        conn = self._get_connection()
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

    def store_motion_event(self, segment: VideoSegment) -> int:
        """Store a motion event in the database.

        Args:
            segment: VideoSegment containing motion event data

        Returns:
            The ID of the created motion event
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        duration = (segment.end_time - segment.start_time).total_seconds()

        cursor.execute('''
            INSERT INTO motion_events
            (start_time, end_time, video_file, duration_seconds, processed)
            VALUES (?, ?, ?, ?, ?)
        ''', (segment.start_time, segment.end_time, segment.file_path,
              duration, segment.processed))

        motion_event_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Stored motion event {motion_event_id}: {segment.file_path}")
        return motion_event_id

    def store_face_detection(self, motion_event_id: int, frame_time: datetime,
                           face_bytes: bytes, embedding: np.ndarray, confidence: float,
                           x: int, y: int, w: int, h: int) -> int:
        """Store a face detection in the database.

        Args:
            motion_event_id: ID of the associated motion event
            frame_time: Timestamp of the frame
            face_bytes: Face crop image as bytes
            embedding: Face embedding vector
            confidence: Detection confidence score
            x, y, w, h: Bounding box coordinates

        Returns:
            The ID of the created face detection
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        embedding_bytes = pickle.dumps(embedding)

        cursor.execute('''
            INSERT INTO face_detections
            (motion_event_id, frame_timestamp, face_crop, face_embedding, confidence,
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, face_bytes, embedding_bytes, confidence,
              x, y, w, h))

        face_detection_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Stored face detection {face_detection_id} for motion event {motion_event_id}")
        return face_detection_id

    def store_object_detection(self, motion_event_id: int, frame_time: datetime,
                              class_name: str, confidence: float,
                              x: int, y: int, w: int, h: int) -> int:
        """Store an object detection in the database.

        Args:
            motion_event_id: ID of the associated motion event
            frame_time: Timestamp of the frame
            class_name: Detected object class name
            confidence: Detection confidence score
            x, y, w, h: Bounding box coordinates

        Returns:
            The ID of the created object detection
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO object_detections
            (motion_event_id, frame_timestamp, class_name, confidence,
             bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, class_name, confidence,
              x, y, w, h))

        object_detection_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Stored object detection {object_detection_id}: {class_name} for motion event {motion_event_id}")
        return object_detection_id

    def get_motion_event_by_video_file(self, video_file: str) -> Optional[Tuple[int, dict]]:
        """Get motion event by video file path.

        Args:
            video_file: Path to the video file

        Returns:
            Tuple of (motion_event_id, motion_event_data) or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, start_time, end_time, video_file, duration_seconds,
                   processed, face_count, object_count, created_at
            FROM motion_events
            WHERE video_file = ?
        ''', (video_file,))

        result = cursor.fetchone()
        conn.close()

        if result:
            motion_event_id = result[0]
            motion_event_data = {
                'id': result[0],
                'start_time': result[1],
                'end_time': result[2],
                'video_file': result[3],
                'duration_seconds': result[4],
                'processed': result[5],
                'face_count': result[6],
                'object_count': result[7],
                'created_at': result[8]
            }
            return motion_event_id, motion_event_data

        return None

    def update_motion_event_counts(self, motion_event_id: int, face_count: int, object_count: int):
        """Update motion event with detection counts and mark as processed.

        Args:
            motion_event_id: ID of the motion event
            face_count: Number of faces detected
            object_count: Number of objects detected
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE motion_events
            SET processed = ?, face_count = ?, object_count = ?
            WHERE id = ?
        ''', (True, face_count, object_count, motion_event_id))

        conn.commit()
        conn.close()

        logger.debug(f"Updated motion event {motion_event_id}: {face_count} faces, {object_count} objects")

    def get_motion_event_stats(self) -> dict:
        """Get motion event statistics.

        Returns:
            Dictionary with motion event statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total motion events
        cursor.execute('SELECT COUNT(*) FROM motion_events')
        total_events = cursor.fetchone()[0]

        # Processed events
        cursor.execute('SELECT COUNT(*) FROM motion_events WHERE processed = 1')
        processed_events = cursor.fetchone()[0]

        # Total duration
        cursor.execute('SELECT SUM(duration_seconds) FROM motion_events')
        total_duration = cursor.fetchone()[0] or 0

        # Average duration
        cursor.execute('SELECT AVG(duration_seconds) FROM motion_events')
        avg_duration = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_motion_events': total_events,
            'processed_events': processed_events,
            'total_duration_seconds': total_duration,
            'average_duration_seconds': avg_duration
        }

    def get_face_detection_stats(self) -> dict:
        """Get face detection statistics.

        Returns:
            Dictionary with face detection statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total face detections
        cursor.execute('SELECT COUNT(*) FROM face_detections')
        total_faces = cursor.fetchone()[0]

        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM face_detections')
        avg_confidence = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_face_detections': total_faces,
            'avg_confidence': avg_confidence
        }

    def get_object_detection_stats(self) -> dict:
        """Get object detection statistics.

        Returns:
            Dictionary with object detection statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total object detections
        cursor.execute('SELECT COUNT(*) FROM object_detections')
        total_objects = cursor.fetchone()[0]

        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM object_detections')
        avg_confidence = cursor.fetchone()[0] or 0

        # Class counts
        cursor.execute('''
            SELECT class_name, COUNT(*) as count
            FROM object_detections
            GROUP BY class_name
            ORDER BY count DESC
        ''')
        class_counts = dict(cursor.fetchall())

        conn.close()

        return {
            'total_object_detections': total_objects,
            'avg_confidence': avg_confidence,
            'class_counts': class_counts
        }

    def close(self):
        """Close database connections (if needed for cleanup)."""
        # SQLite connections are closed after each operation, so nothing to do here
        pass