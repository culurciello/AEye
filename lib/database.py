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
                known_person TEXT,
                recognition_confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (motion_event_id) REFERENCES motion_events(id)
            )
        ''')

        # Add columns to existing motion_events table if they don't exist
        try:
            cursor.execute('ALTER TABLE motion_events ADD COLUMN track_count INTEGER DEFAULT 0')
        except:
            pass  # Column already exists

        # Add columns to existing face_detections table if they don't exist
        try:
            cursor.execute('ALTER TABLE face_detections ADD COLUMN known_person TEXT')
        except:
            pass  # Column already exists

        try:
            cursor.execute('ALTER TABLE face_detections ADD COLUMN recognition_confidence REAL')
        except:
            pass  # Column already exists

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
                object_crop BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (motion_event_id) REFERENCES motion_events(id)
            )
        ''')

        # Add object_crop column to existing object_detections table if it doesn't exist
        try:
            cursor.execute('ALTER TABLE object_detections ADD COLUMN object_crop BLOB')
        except:
            pass  # Column already exists

        # Create object tracks table for organizing detections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                motion_event_id INTEGER,
                class_name TEXT,
                track_start_time TIMESTAMP,
                track_end_time TIMESTAMP,
                detection_count INTEGER DEFAULT 0,
                avg_confidence REAL,
                first_bbox_x INTEGER,
                first_bbox_y INTEGER,
                first_bbox_width INTEGER,
                first_bbox_height INTEGER,
                last_bbox_x INTEGER,
                last_bbox_y INTEGER,
                last_bbox_width INTEGER,
                last_bbox_height INTEGER,
                representative_crop BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (motion_event_id) REFERENCES motion_events(id)
            )
        ''')

        # Add track_id column to object_detections to link detections to tracks
        try:
            cursor.execute('ALTER TABLE object_detections ADD COLUMN track_id INTEGER')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_object_detections_track_id ON object_detections(track_id)')
        except:
            pass  # Column already exists

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
                           x: int, y: int, w: int, h: int, known_person: str = None,
                           recognition_confidence: float = None) -> int:
        """Store a face detection in the database.

        Args:
            motion_event_id: ID of the associated motion event
            frame_time: Timestamp of the frame
            face_bytes: Face crop image as bytes
            embedding: Face embedding vector
            confidence: Detection confidence score
            x, y, w, h: Bounding box coordinates
            known_person: Name of known person if recognized
            recognition_confidence: Confidence score for recognition

        Returns:
            The ID of the created face detection
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        embedding_bytes = pickle.dumps(embedding)

        cursor.execute('''
            INSERT INTO face_detections
            (motion_event_id, frame_timestamp, face_crop, face_embedding, confidence,
             known_person, recognition_confidence, bbox_x, bbox_y, bbox_width, bbox_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, face_bytes, embedding_bytes, confidence,
              known_person, recognition_confidence, x, y, w, h))

        face_detection_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Stored face detection {face_detection_id} for motion event {motion_event_id}")
        return face_detection_id

    def store_object_detection(self, motion_event_id: int, frame_time: datetime,
                              class_name: str, confidence: float,
                              x: int, y: int, w: int, h: int, object_crop: bytes = None, track_id: int = None) -> int:
        """Store an object detection in the database.

        Args:
            motion_event_id: ID of the associated motion event
            frame_time: Timestamp of the frame
            class_name: Detected object class name
            confidence: Detection confidence score
            x, y, w, h: Bounding box coordinates
            object_crop: Object crop image as bytes

        Returns:
            The ID of the created object detection
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO object_detections
            (motion_event_id, frame_timestamp, class_name, confidence,
             bbox_x, bbox_y, bbox_width, bbox_height, object_crop, track_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, frame_time, class_name, confidence,
              x, y, w, h, object_crop, track_id))

        object_detection_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Stored object detection {object_detection_id}: {class_name} for motion event {motion_event_id}")
        return object_detection_id

    def create_object_track(self, motion_event_id: int, class_name: str, start_time: datetime,
                           first_bbox: tuple, first_confidence: float, representative_crop: bytes = None) -> int:
        """Create a new object track.

        Args:
            motion_event_id: ID of the associated motion event
            class_name: Object class name
            start_time: Track start timestamp
            first_bbox: First bounding box (x, y, w, h)
            first_confidence: First detection confidence
            representative_crop: Representative crop image

        Returns:
            The ID of the created track
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        x, y, w, h = first_bbox

        cursor.execute('''
            INSERT INTO object_tracks
            (motion_event_id, class_name, track_start_time, track_end_time,
             detection_count, avg_confidence, first_bbox_x, first_bbox_y,
             first_bbox_width, first_bbox_height, last_bbox_x, last_bbox_y,
             last_bbox_width, last_bbox_height, representative_crop)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (motion_event_id, class_name, start_time, start_time,
              1, first_confidence, x, y, w, h, x, y, w, h, representative_crop))

        track_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.debug(f"Created object track {track_id}: {class_name} for motion event {motion_event_id}")
        return track_id

    def update_object_track(self, track_id: int, end_time: datetime, last_bbox: tuple,
                          detection_count: int, avg_confidence: float) -> None:
        """Update an existing object track.

        Args:
            track_id: ID of the track to update
            end_time: Track end timestamp
            last_bbox: Last bounding box (x, y, w, h)
            detection_count: Total number of detections in track
            avg_confidence: Average confidence across all detections
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        x, y, w, h = last_bbox

        cursor.execute('''
            UPDATE object_tracks
            SET track_end_time = ?, last_bbox_x = ?, last_bbox_y = ?,
                last_bbox_width = ?, last_bbox_height = ?,
                detection_count = ?, avg_confidence = ?
            WHERE id = ?
        ''', (end_time, x, y, w, h, detection_count, avg_confidence, track_id))

        conn.commit()
        conn.close()

        logger.debug(f"Updated object track {track_id}")

    def update_motion_event_track_count(self, motion_event_id: int) -> None:
        """Update the track count for a motion event."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Count tracks for this motion event
        cursor.execute('SELECT COUNT(*) FROM object_tracks WHERE motion_event_id = ?', (motion_event_id,))
        track_count = cursor.fetchone()[0]

        # Update motion event
        cursor.execute(
            'UPDATE motion_events SET track_count = ? WHERE id = ?',
            (track_count, motion_event_id)
        )

        conn.commit()
        conn.close()

        logger.debug(f"Updated motion event {motion_event_id} track count to {track_count}")

    def get_object_tracks(self, motion_event_id: int = None) -> list:
        """Get object tracks with optional filtering.

        Args:
            motion_event_id: Optional motion event ID to filter by

        Returns:
            List of track records
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = '''
            SELECT id, motion_event_id, class_name, track_start_time, track_end_time,
                   detection_count, avg_confidence, first_bbox_x, first_bbox_y,
                   first_bbox_width, first_bbox_height, last_bbox_x, last_bbox_y,
                   last_bbox_width, last_bbox_height, created_at
            FROM object_tracks
        '''
        params = []

        if motion_event_id:
            query += ' WHERE motion_event_id = ?'
            params.append(motion_event_id)

        query += ' ORDER BY track_start_time DESC'

        cursor.execute(query, params)
        tracks = cursor.fetchall()
        conn.close()

        return tracks

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

        # Recognized faces count
        cursor.execute('SELECT COUNT(*) FROM face_detections WHERE known_person IS NOT NULL')
        recognized_faces = cursor.fetchone()[0]

        # Count by person
        cursor.execute('''
            SELECT known_person, COUNT(*) as count
            FROM face_detections
            WHERE known_person IS NOT NULL
            GROUP BY known_person
            ORDER BY count DESC
        ''')
        person_counts = dict(cursor.fetchall())

        conn.close()

        return {
            'total_face_detections': total_faces,
            'recognized_faces': recognized_faces,
            'unknown_faces': total_faces - recognized_faces,
            'avg_confidence': avg_confidence,
            'person_counts': person_counts
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