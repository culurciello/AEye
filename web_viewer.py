#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import os
import sqlite3
import json
import cv2
import argparse
from datetime import datetime, timedelta
import glob
from typing import List, Dict, Optional

app = Flask(__name__)

class MotionViewer:
    def __init__(self, base_path: str = "data", date: str = None):
        """
        Initialize motion viewer with date-based organization.

        Args:
            base_path: Base directory for data storage
            date: Specific date to view (YYYY-MM-DD format). If None, uses latest date with data.
        """
        self.base_path = base_path
        self.videos_dir = os.path.join(base_path, "videos")
        self.images_dir = os.path.join(base_path, "images")
        self.db_dir = os.path.join(base_path, "db")

        # Ensure directories exist
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)

        if date:
            self.current_date = date
        else:
            # Auto-select the latest date with data
            available_dates = self.get_available_dates()
            if available_dates:
                self.current_date = available_dates[0]  # Most recent date
            else:
                self.current_date = datetime.now().strftime("%Y-%m-%d")

        # Database path
        self.db_path = os.path.join(self.db_dir, "detections.db")
    
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        # Ensure database has latest schema
        self._migrate_database(conn)
        return conn

    def _migrate_database(self, conn):
        """Migrate database to latest schema if needed."""
        cursor = conn.cursor()

        # Check if object_count column exists in motion_events
        try:
            cursor.execute("PRAGMA table_info(motion_events)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'object_count' not in columns:
                cursor.execute("ALTER TABLE motion_events ADD COLUMN object_count INTEGER DEFAULT 0")
                print("Added object_count column to motion_events table")

            if 'track_count' not in columns:
                cursor.execute("ALTER TABLE motion_events ADD COLUMN track_count INTEGER DEFAULT 0")
                print("Added track_count column to motion_events table")

        except Exception as e:
            print(f"Database migration info: {e}")

        # Ensure object_detections table exists
        try:
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
        except Exception as e:
            print(f"Object detections table creation info: {e}")

        # Check if object_crop column exists in object_detections
        try:
            cursor.execute("PRAGMA table_info(object_detections)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'object_crop' not in columns:
                cursor.execute("ALTER TABLE object_detections ADD COLUMN object_crop BLOB")
                print("Added object_crop column to object_detections table")

            if 'track_id' not in columns:
                cursor.execute("ALTER TABLE object_detections ADD COLUMN track_id INTEGER")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_object_detections_track_id ON object_detections(track_id)")
                print("Added track_id column to object_detections table")

        except Exception as e:
            print(f"Object crop column migration info: {e}")

        # Ensure object_tracks table exists
        try:
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
        except Exception as e:
            print(f"Object tracks table creation info: {e}")

        conn.commit()

    def get_available_dates(self) -> List[str]:
        """Get list of available dates with data (either videos or images)."""
        dates = set()

        # Check video directories
        if os.path.exists(self.videos_dir):
            for item in os.listdir(self.videos_dir):
                if os.path.isdir(os.path.join(self.videos_dir, item)) and item.count('_') == 2:
                    # Convert YYYY_MM_DD to YYYY-MM-DD
                    try:
                        date_str = item.replace('_', '-')
                        datetime.strptime(date_str, '%Y-%m-%d')  # Validate format
                        dates.add(date_str)
                    except ValueError:
                        continue

        # Check image directories
        if os.path.exists(self.images_dir):
            for item in os.listdir(self.images_dir):
                if os.path.isdir(os.path.join(self.images_dir, item)) and item.count('_') == 2:
                    try:
                        date_str = item.replace('_', '-')
                        datetime.strptime(date_str, '%Y-%m-%d')  # Validate format
                        dates.add(date_str)
                    except ValueError:
                        continue

        return sorted(list(dates), reverse=True)  # Most recent first

    def switch_date(self, date: str):
        """Switch to a different date."""
        self.current_date = date

    def get_motion_events(self, date_filter=None, limit=100, offset=0):
        """Get motion events with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """SELECT id, start_time, end_time, video_file, duration_seconds,
                              processed, face_count, object_count, track_count, created_at
                       FROM motion_events WHERE 1=1"""
            params = []

            if date_filter:
                query += " AND DATE(start_time) = ?"
                params.append(date_filter)

            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_motion_stats(self):
        """Get statistics about motion events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT DATE(start_time) as date, COUNT(*) as count
                FROM motion_events
                GROUP BY DATE(start_time)
                ORDER BY date DESC
                LIMIT 30
            ''')
            date_counts = cursor.fetchall()

            cursor.execute('SELECT COUNT(*) FROM motion_events')
            total_motion_events = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM face_detections')
            total_face_detections = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM object_detections')
            total_object_detections = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(face_count) FROM motion_events')
            total_faces_in_events = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(object_count) FROM motion_events')
            total_objects_in_events = cursor.fetchone()[0] or 0

            cursor.execute('SELECT SUM(track_count) FROM motion_events')
            total_tracks_in_events = cursor.fetchone()[0] or 0

            cursor.execute('SELECT processed, COUNT(*) FROM motion_events GROUP BY processed')
            processing_stats = dict(cursor.fetchall())

            return {
                'date_counts': date_counts,
                'total_motion_events': total_motion_events,
                'total_face_detections': total_face_detections,
                'total_object_detections': total_object_detections,
                'total_faces_in_events': total_faces_in_events,
                'total_objects_in_events': total_objects_in_events,
                'total_tracks_in_events': total_tracks_in_events,
                'processed_events': processing_stats.get(1, 0),
                'unprocessed_events': processing_stats.get(0, 0)
            }
    
    def get_face_detections(self, motion_event_id=None, date_filter=None, limit=100, offset=0):
        """Get face detections with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """SELECT fd.id, fd.motion_event_id, fd.frame_timestamp,
                              fd.confidence, fd.bbox_x, fd.bbox_y, fd.bbox_width, fd.bbox_height,
                              fd.created_at, me.video_file, fd.known_person, fd.recognition_confidence
                       FROM face_detections fd
                       JOIN motion_events me ON fd.motion_event_id = me.id
                       WHERE 1=1"""
            params = []

            if motion_event_id:
                query += " AND fd.motion_event_id = ?"
                params.append(motion_event_id)

            if date_filter:
                query += " AND DATE(fd.frame_timestamp) = ?"
                params.append(date_filter)

            query += " ORDER BY fd.frame_timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return cursor.fetchall()

    def get_face_detection_stats(self):
        """Get statistics about face detections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT DATE(frame_timestamp) as date, COUNT(*) as count
                FROM face_detections
                GROUP BY DATE(frame_timestamp)
                ORDER BY date DESC
                LIMIT 30
            ''')
            face_date_counts = cursor.fetchall()

            cursor.execute('SELECT COUNT(*) FROM face_detections')
            total_face_detections = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(confidence) FROM face_detections')
            avg_confidence = cursor.fetchone()[0] or 0

            return {
                'face_date_counts': face_date_counts,
                'total_face_detections': total_face_detections,
                'avg_confidence': round(avg_confidence, 3)
            }

    def get_object_detections(self, motion_event_id=None, date_filter=None, limit=100, offset=0):
        """Get object detections with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """SELECT od.id, od.motion_event_id, od.frame_timestamp, od.class_name,
                              od.confidence, od.bbox_x, od.bbox_y, od.bbox_width, od.bbox_height,
                              od.created_at, me.video_file
                       FROM object_detections od
                       JOIN motion_events me ON od.motion_event_id = me.id
                       WHERE 1=1"""
            params = []

            if motion_event_id:
                query += " AND od.motion_event_id = ?"
                params.append(motion_event_id)

            if date_filter:
                query += " AND DATE(od.frame_timestamp) = ?"
                params.append(date_filter)

            query += " ORDER BY od.frame_timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return cursor.fetchall()

    def get_object_detection_stats(self):
        """Get statistics about object detections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT DATE(frame_timestamp) as date, COUNT(*) as count
                FROM object_detections
                GROUP BY DATE(frame_timestamp)
                ORDER BY date DESC
                LIMIT 30
            ''')
            object_date_counts = cursor.fetchall()

            cursor.execute('SELECT COUNT(*) FROM object_detections')
            total_object_detections = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(confidence) FROM object_detections')
            avg_confidence = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT class_name, COUNT(*) as count
                FROM object_detections
                GROUP BY class_name
                ORDER BY count DESC
                LIMIT 10
            ''')
            class_counts = cursor.fetchall()

            return {
                'object_date_counts': object_date_counts,
                'total_object_detections': total_object_detections,
                'avg_confidence': round(avg_confidence, 3),
                'class_counts': class_counts
            }

    def get_object_tracks(self, motion_event_id=None, date_filter=None, limit=100, offset=0):
        """Get object tracks with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """SELECT ot.id, ot.motion_event_id, ot.class_name, ot.track_start_time, ot.track_end_time,
                              ot.detection_count, ot.avg_confidence, ot.first_bbox_x, ot.first_bbox_y,
                              ot.first_bbox_width, ot.first_bbox_height, ot.last_bbox_x, ot.last_bbox_y,
                              ot.last_bbox_width, ot.last_bbox_height, ot.created_at, me.video_file
                       FROM object_tracks ot
                       JOIN motion_events me ON ot.motion_event_id = me.id
                       WHERE 1=1"""
            params = []

            if motion_event_id:
                query += " AND ot.motion_event_id = ?"
                params.append(motion_event_id)

            if date_filter:
                query += " AND DATE(ot.track_start_time) = ?"
                params.append(date_filter)

            query += " ORDER BY ot.track_start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            return cursor.fetchall()

    def get_object_track_stats(self):
        """Get statistics about object tracks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT DATE(track_start_time) as date, COUNT(*) as count
                FROM object_tracks
                GROUP BY DATE(track_start_time)
                ORDER BY date DESC
                LIMIT 30
            ''')
            track_date_counts = cursor.fetchall()

            cursor.execute('SELECT COUNT(*) FROM object_tracks')
            total_tracks = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(avg_confidence) FROM object_tracks')
            avg_track_confidence = cursor.fetchone()[0] or 0

            cursor.execute('SELECT AVG(detection_count) FROM object_tracks')
            avg_detections_per_track = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT class_name, COUNT(*) as count, AVG(detection_count) as avg_detections
                FROM object_tracks
                GROUP BY class_name
                ORDER BY count DESC
                LIMIT 10
            ''')
            track_class_stats = cursor.fetchall()

            return {
                'track_date_counts': track_date_counts,
                'total_tracks': total_tracks,
                'avg_track_confidence': round(avg_track_confidence, 3),
                'avg_detections_per_track': round(avg_detections_per_track, 1),
                'track_class_stats': track_class_stats
            }

    def get_images_for_date(self, date):
        """Get list of images for a specific date."""
        date_str = date.replace('-', '_')  # Convert YYYY-MM-DD to YYYY_MM_DD
        date_dir = os.path.join(self.images_dir, date_str)

        if not os.path.exists(date_dir):
            return []

        images = []
        for file in os.listdir(date_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append({
                    'filename': file,
                    'path': f"{date_str}/{file}",
                    'full_path': os.path.join(date_dir, file),
                    'url': f"/api/serve_image/{date_str}/{file}"
                })

        return sorted(images, key=lambda x: x['filename'])

    def get_videos_for_date(self, date):
        """Get list of videos for a specific date."""
        date_str = date.replace('-', '_')  # Convert YYYY-MM-DD to YYYY_MM_DD
        date_dir = os.path.join(self.videos_dir, date_str)

        if not os.path.exists(date_dir):
            return []

        videos = []
        for file in os.listdir(date_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                full_path = os.path.join(date_dir, file)
                # Get file stats
                stat = os.stat(full_path)
                videos.append({
                    'filename': file,
                    'path': f"{date_str}/{file}",
                    'full_path': full_path,
                    'size_bytes': stat.st_size,
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'url': f"/api/serve_video/{date_str}/{file}"
                })

        return sorted(videos, key=lambda x: x['filename'])


# Initialize viewer (will be updated in main function)
viewer = None

@app.route('/')
def index():
    """Main detection viewer page."""
    return render_template('index.html')

@app.route('/api/motion_events')
def api_motion_events():
    """API endpoint to get motion events with filtering."""
    date_filter = request.args.get('date')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    events = viewer.get_motion_events(
        date_filter=date_filter,
        limit=limit,
        offset=offset
    )

    # Convert to JSON-serializable format
    event_list = []
    for event in events:
        event_data = {
            'id': event[0],
            'start_time': event[1],
            'end_time': event[2],
            'video_file': event[3],
            'duration_seconds': event[4],
            'processed': bool(event[5]),
            'face_count': event[6],
            'object_count': event[7] if len(event) > 7 else 0,
            'track_count': event[8] if len(event) > 8 else 0,
            'created_at': event[9] if len(event) > 9 else (event[8] if len(event) > 8 else event[7])
        }
        event_list.append(event_data)

    return jsonify(event_list)

@app.route('/api/stats')
def api_stats():
    """API endpoint to get motion event statistics."""
    stats = viewer.get_motion_stats()
    return jsonify(stats)

@app.route('/api/face_detections')
def api_face_detections():
    """API endpoint to get face detections with filtering."""
    motion_event_id = request.args.get('motion_event_id', type=int)
    date_filter = request.args.get('date')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    detections = viewer.get_face_detections(
        motion_event_id=motion_event_id,
        date_filter=date_filter,
        limit=limit,
        offset=offset
    )

    # Convert to JSON-serializable format
    detection_list = []
    for detection in detections:
        detection_data = {
            'id': detection[0],
            'motion_event_id': detection[1],
            'frame_timestamp': detection[2],
            'confidence': detection[3],
            'bbox_x': detection[4],
            'bbox_y': detection[5],
            'bbox_width': detection[6],
            'bbox_height': detection[7],
            'created_at': detection[8],
            'video_file': detection[9],
            'known_person': detection[10] if len(detection) > 10 else None,
            'recognition_confidence': detection[11] if len(detection) > 11 else None
        }
        detection_list.append(detection_data)

    return jsonify(detection_list)

@app.route('/api/face_stats')
def api_face_stats_route():
    """API endpoint to get face detection statistics."""
    stats = viewer.get_face_detection_stats()
    return jsonify(stats)

@app.route('/api/object_detections')
def api_object_detections():
    """API endpoint to get object detections with filtering."""
    motion_event_id = request.args.get('motion_event_id', type=int)
    date_filter = request.args.get('date')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    detections = viewer.get_object_detections(
        motion_event_id=motion_event_id,
        date_filter=date_filter,
        limit=limit,
        offset=offset
    )

    # Convert to JSON-serializable format
    detection_list = []
    for detection in detections:
        detection_data = {
            'id': detection[0],
            'motion_event_id': detection[1],
            'frame_timestamp': detection[2],
            'class_name': detection[3],
            'confidence': detection[4],
            'bbox_x': detection[5],
            'bbox_y': detection[6],
            'bbox_width': detection[7],
            'bbox_height': detection[8],
            'created_at': detection[9],
            'video_file': detection[10]
        }
        detection_list.append(detection_data)

    return jsonify(detection_list)

@app.route('/api/object_stats')
def api_object_stats_route():
    """API endpoint to get object detection statistics."""
    stats = viewer.get_object_detection_stats()
    return jsonify(stats)

@app.route('/api/object_tracks')
def api_object_tracks():
    """API endpoint to get object tracks with filtering."""
    motion_event_id = request.args.get('motion_event_id', type=int)
    date_filter = request.args.get('date')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    tracks = viewer.get_object_tracks(
        motion_event_id=motion_event_id,
        date_filter=date_filter,
        limit=limit,
        offset=offset
    )

    # Convert to JSON-serializable format
    track_list = []
    for track in tracks:
        track_data = {
            'id': track[0],
            'motion_event_id': track[1],
            'class_name': track[2],
            'track_start_time': track[3],
            'track_end_time': track[4],
            'detection_count': track[5],
            'avg_confidence': track[6],
            'first_bbox_x': track[7],
            'first_bbox_y': track[8],
            'first_bbox_width': track[9],
            'first_bbox_height': track[10],
            'last_bbox_x': track[11],
            'last_bbox_y': track[12],
            'last_bbox_width': track[13],
            'last_bbox_height': track[14],
            'created_at': track[15],
            'video_file': track[16]
        }
        track_list.append(track_data)

    return jsonify(track_list)

@app.route('/api/motion_event/<int:event_id>/tracks')
def api_motion_event_tracks(event_id):
    """API endpoint to get tracks for a motion event (primary view)."""
    include_detections = request.args.get('include_detections', 'false').lower() == 'true'

    try:
        tracks = viewer.get_object_tracks(motion_event_id=event_id)

        track_list = []
        for track in tracks:
            track_data = {
                'id': track[0],
                'motion_event_id': track[1],
                'class_name': track[2],
                'track_start_time': track[3],
                'track_end_time': track[4],
                'detection_count': track[5],
                'avg_confidence': track[6],
                'first_bbox': {
                    'x': track[7],
                    'y': track[8],
                    'width': track[9],
                    'height': track[10]
                },
                'last_bbox': {
                    'x': track[11],
                    'y': track[12],
                    'width': track[13],
                    'height': track[14]
                },
                'created_at': track[15],
                'video_file': track[16],
                'crop_url': f'/api/object_track/{track[0]}/image'
            }

            # Optionally include individual detections for this track
            if include_detections:
                with viewer._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, frame_timestamp, confidence, bbox_x, bbox_y, bbox_width, bbox_height FROM object_detections WHERE track_id = ? ORDER BY frame_timestamp",
                        (track[0],)
                    )
                    detections = cursor.fetchall()

                    track_data['detections'] = []
                    for det in detections:
                        detection_data = {
                            'id': det[0],
                            'frame_timestamp': det[1],
                            'confidence': det[2],
                            'bbox': {
                                'x': det[3],
                                'y': det[4],
                                'width': det[5],
                                'height': det[6]
                            },
                            'crop_url': f'/api/object_detection/{det[0]}/image'
                        }
                        track_data['detections'].append(detection_data)

            track_list.append(track_data)

        return jsonify(track_list)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/track_stats')
def api_track_stats_route():
    """API endpoint to get object track statistics."""
    stats = viewer.get_object_track_stats()
    return jsonify(stats)

@app.route('/api/hourly_activity')
def api_hourly_activity():
    """API endpoint to get motion events grouped by hour."""
    date_filter = request.args.get('date')

    if not date_filter:
        return jsonify({'error': 'Date parameter is required'}), 400

    try:
        with viewer._get_connection() as conn:
            cursor = conn.cursor()

            # Get all events for the specific date and group by hour
            cursor.execute("""
                SELECT id, start_time
                FROM motion_events
                WHERE DATE(start_time) = ?
                ORDER BY start_time
            """, (date_filter,))

            events = cursor.fetchall()

            # Group events by hour
            hourly_data = {}
            for event_id, start_time in events:
                # Extract hour from timestamp
                hour = start_time.split(' ')[1].split(':')[0] if ' ' in start_time else start_time.split(':')[0]

                # Handle different timestamp formats
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    hour = dt.strftime('%H')
                except:
                    # Fallback for different formats
                    hour = start_time.split('T')[1].split(':')[0] if 'T' in start_time else hour

                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(event_id)

            return jsonify(hourly_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images')
def api_images():
    """API endpoint to get images for a date."""
    date = request.args.get('date', viewer.current_date)
    images = viewer.get_images_for_date(date)
    return jsonify(images)

@app.route('/api/videos')
def api_videos():
    """API endpoint to get videos for a date."""
    date = request.args.get('date', viewer.current_date)
    videos = viewer.get_videos_for_date(date)
    return jsonify(videos)

@app.route('/api/motion_event/<int:event_id>')
def api_motion_event_detail(event_id):
    """API endpoint to get detailed motion event information."""
    try:
        with viewer._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, start_time, end_time, video_file, duration_seconds, processed, face_count, object_count, track_count, created_at FROM motion_events WHERE id = ?",
                (event_id,)
            )
            event = cursor.fetchone()

            if not event:
                return jsonify({'error': 'Motion event not found'}), 404

            event_data = {
                'id': event[0],
                'start_time': str(event[1]) if event[1] else None,
                'end_time': str(event[2]) if event[2] else None,
                'video_file': str(event[3]) if event[3] else None,
                'duration_seconds': float(event[4]) if event[4] else 0.0,
                'processed': bool(event[5]) if event[5] is not None else False,
                'face_count': int(event[6]) if event[6] else 0,
                'object_count': int(event[7]) if event[7] else 0,
                'track_count': int(event[8]) if event[8] else 0,
                'created_at': str(event[9]) if event[9] else None
            }

            # Get face detections for this event (exclude binary data and handle data type conversion safely)
            cursor.execute(
                "SELECT id, frame_timestamp, confidence, bbox_x, bbox_y, bbox_width, bbox_height, known_person, recognition_confidence FROM face_detections WHERE motion_event_id = ?",
                (event_id,)
            )
            face_detections = cursor.fetchall()

            event_data['face_detections'] = []
            for fd in face_detections:
                try:
                    face_detection = {
                        'id': int(fd[0]),
                        'frame_timestamp': str(fd[1]) if fd[1] else None,
                        'confidence': float(fd[2]) if fd[2] is not None and not isinstance(fd[2], bytes) else 0.0,
                        'bbox_x': int(fd[3]) if fd[3] is not None and not isinstance(fd[3], bytes) else 0,
                        'bbox_y': int(fd[4]) if fd[4] is not None and not isinstance(fd[4], bytes) else 0,
                        'bbox_width': int(fd[5]) if fd[5] is not None and not isinstance(fd[5], bytes) else 0,
                        'bbox_height': int(fd[6]) if fd[6] is not None and not isinstance(fd[6], bytes) else 0,
                        'known_person': str(fd[7]) if fd[7] else None,
                        'recognition_confidence': float(fd[8]) if fd[8] is not None and not isinstance(fd[8], bytes) else None
                    }
                    event_data['face_detections'].append(face_detection)
                except (ValueError, TypeError) as e:
                    # Skip invalid face detection data
                    continue

            # Get object detections for this event
            cursor.execute(
                "SELECT id, frame_timestamp, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height FROM object_detections WHERE motion_event_id = ?",
                (event_id,)
            )
            object_detections = cursor.fetchall()

            event_data['object_detections'] = []
            for od in object_detections:
                try:
                    object_detection = {
                        'id': int(od[0]),
                        'frame_timestamp': str(od[1]) if od[1] else None,
                        'class_name': str(od[2]) if od[2] else None,
                        'confidence': float(od[3]) if od[3] is not None else 0.0,
                        'bbox_x': int(od[4]) if od[4] is not None else 0,
                        'bbox_y': int(od[5]) if od[5] is not None else 0,
                        'bbox_width': int(od[6]) if od[6] is not None else 0,
                        'bbox_height': int(od[7]) if od[7] is not None else 0
                    }
                    event_data['object_detections'].append(object_detection)
                except (ValueError, TypeError) as e:
                    # Skip invalid object detection data
                    continue

            # Get object tracks for this event (primary object detection data)
            cursor.execute(
                "SELECT id, class_name, track_start_time, track_end_time, detection_count, avg_confidence, first_bbox_x, first_bbox_y, first_bbox_width, first_bbox_height, last_bbox_x, last_bbox_y, last_bbox_width, last_bbox_height FROM object_tracks WHERE motion_event_id = ?",
                (event_id,)
            )
            object_tracks = cursor.fetchall()

            event_data['object_tracks'] = []
            for ot in object_tracks:
                try:
                    duration = None
                    if ot[2] and ot[3]:  # start_time and end_time
                        try:
                            from datetime import datetime
                            start = datetime.fromisoformat(str(ot[2]).replace('Z', '+00:00'))
                            end = datetime.fromisoformat(str(ot[3]).replace('Z', '+00:00'))
                            duration = (end - start).total_seconds()
                        except:
                            duration = None

                    object_track = {
                        'id': int(ot[0]),
                        'class_name': str(ot[1]) if ot[1] else None,
                        'track_start_time': str(ot[2]) if ot[2] else None,
                        'track_end_time': str(ot[3]) if ot[3] else None,
                        'duration_seconds': duration,
                        'detection_count': int(ot[4]) if ot[4] is not None else 0,
                        'avg_confidence': float(ot[5]) if ot[5] is not None else 0.0,
                        'first_bbox': {
                            'x': int(ot[6]) if ot[6] is not None else 0,
                            'y': int(ot[7]) if ot[7] is not None else 0,
                            'width': int(ot[8]) if ot[8] is not None else 0,
                            'height': int(ot[9]) if ot[9] is not None else 0
                        },
                        'last_bbox': {
                            'x': int(ot[10]) if ot[10] is not None else 0,
                            'y': int(ot[11]) if ot[11] is not None else 0,
                            'width': int(ot[12]) if ot[12] is not None else 0,
                            'height': int(ot[13]) if ot[13] is not None else 0
                        },
                        'crop_url': f'/api/object_track/{int(ot[0])}/image'
                    }
                    event_data['object_tracks'].append(object_track)
                except (ValueError, TypeError) as e:
                    # Skip invalid track data
                    continue

            # Check if the video file exists, if not find closest match
            if event_data['video_file']:
                video_path = event_data['video_file']
                full_path = os.path.join(viewer.videos_dir, video_path)

                if not os.path.exists(full_path):
                    # File doesn't exist, find closest match by timestamp
                    date_str = video_path.split('/')[0]  # e.g., "2025_09_14"
                    video_dir = os.path.join(viewer.videos_dir, date_str)

                    if os.path.exists(video_dir):
                        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                        if video_files:
                            # Extract time from event
                            event_time_str = event_data['start_time'].split(' ')[1] if ' ' in event_data['start_time'] else event_data['start_time']
                            event_time_clean = event_time_str.replace(':', '')[:6]  # HHMMSS

                            closest_match = None
                            closest_diff = float('inf')

                            for video_file in video_files:
                                try:
                                    # Extract time from filename: "20250914_132048.mp4" -> "132048"
                                    video_time_str = video_file.split('_')[1].replace('.mp4', '')[:6]
                                    if len(video_time_str) == 6 and len(event_time_clean) == 6:
                                        # Convert to seconds for comparison
                                        event_seconds = int(event_time_clean[:2]) * 3600 + int(event_time_clean[2:4]) * 60 + int(event_time_clean[4:6])
                                        video_seconds = int(video_time_str[:2]) * 3600 + int(video_time_str[2:4]) * 60 + int(video_time_str[4:6])

                                        diff = abs(video_seconds - event_seconds)
                                        if diff < closest_diff:
                                            closest_diff = diff
                                            closest_match = video_file
                                except (ValueError, IndexError):
                                    continue

                            if closest_match:
                                event_data['video_file'] = f"{date_str}/{closest_match}"

            return jsonify(event_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/serve_image/<path:image_path>')
def serve_image(image_path):
    """Serve image files from the images directory."""
    try:
        # image_path format: YYYY_MM_DD/filename.jpg
        return send_from_directory(viewer.images_dir, image_path)
    except Exception as e:
        return f"Image not found: {str(e)}", 404

@app.route('/api/serve_video/<path:video_path>')
def serve_video(video_path):
    """Serve video files from the videos directory."""
    try:
        # Handle both full paths and relative paths
        if video_path.startswith('data/videos/'):
            # Strip the data/videos/ prefix to get the relative path
            relative_path = video_path.replace('data/videos/', '')
        else:
            # Already a relative path (YYYY_MM_DD/filename.mp4)
            relative_path = video_path

        return send_from_directory(viewer.videos_dir, relative_path)
    except Exception as e:
        return f"Video not found: {str(e)}", 404

@app.route('/api/face_detection/<int:detection_id>/image')
def api_face_detection_image(detection_id):
    """API endpoint to get face crop image."""
    try:
        with viewer._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT face_crop FROM face_detections WHERE id = ?",
                (detection_id,)
            )
            result = cursor.fetchone()

            if not result or not result[0]:
                return "Face crop not found", 404

            return Response(result[0], mimetype='image/jpeg')
    except Exception as e:
        return f"Error retrieving face crop: {str(e)}", 500

@app.route('/api/object_detection/<int:detection_id>/image')
def api_object_detection_image(detection_id):
    """API endpoint to get object crop image - prioritizes stored crops, falls back to video extraction."""
    try:
        with viewer._get_connection() as conn:
            cursor = conn.cursor()

            # First, try to get stored object crop
            cursor.execute("""
                SELECT od.object_crop, od.bbox_x, od.bbox_y, od.bbox_width, od.bbox_height,
                       od.frame_timestamp, me.video_file
                FROM object_detections od
                JOIN motion_events me ON od.motion_event_id = me.id
                WHERE od.id = ?
            """, (detection_id,))

            result = cursor.fetchone()
            if not result:
                return "Object detection not found", 404

            object_crop, bbox_x, bbox_y, bbox_width, bbox_height, frame_timestamp, video_file = result

            # If we have a stored crop, return it directly
            if object_crop:
                return Response(object_crop, mimetype='image/jpeg')

            # Fallback to video frame extraction
            # Handle video file path (remove data/videos/ prefix if present)
            if video_file.startswith('data/videos/'):
                relative_path = video_file.replace('data/videos/', '')
            else:
                relative_path = video_file

            video_path = os.path.join(viewer.videos_dir, relative_path)

            if not os.path.exists(video_path):
                return "Video file not found", 404

            # Open video and extract frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "Could not open video", 404

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Parse frame timestamp to get approximate frame number
            # This is approximate since we don't have exact frame timing
            try:
                from datetime import datetime
                # Use middle frame as fallback
                frame_number = total_frames // 2
            except:
                frame_number = total_frames // 2

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return "Could not read video frame", 404

            # Crop the object from frame using bbox coordinates
            y1, y2 = max(0, bbox_y), min(frame.shape[0], bbox_y + bbox_height)
            x1, x2 = max(0, bbox_x), min(frame.shape[1], bbox_x + bbox_width)

            if x2 <= x1 or y2 <= y1:
                return "Invalid bounding box", 404

            cropped_object = frame[y1:y2, x1:x2]

            # Resize crop to standard thumbnail size
            thumbnail_size = (80, 80)
            cropped_resized = cv2.resize(cropped_object, thumbnail_size)

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', cropped_resized)

            return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        return f"Error generating object crop: {str(e)}", 500

@app.route('/api/object_track/<int:track_id>/image')
def api_object_track_image(track_id):
    """API endpoint to get object track representative crop image."""
    try:
        with viewer._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT representative_crop FROM object_tracks WHERE id = ?",
                (track_id,)
            )
            result = cursor.fetchone()

            if not result or not result[0]:
                return "Track representative crop not found", 404

            return Response(result[0], mimetype='image/jpeg')
    except Exception as e:
        return f"Error retrieving track crop: {str(e)}", 500


@app.route('/api/video_info/<path:video_path>')
def api_video_info(video_path):
    """Get video file information."""
    try:
        full_path = os.path.join(viewer.videos_dir, video_path)

        if not os.path.exists(full_path):
            return jsonify({'error': 'Video file not found'}), 404

        # Get video info
        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = total_frames / fps if fps > 0 else 0

        return jsonify({
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'stream_url': f'/api/serve_video/{video_path}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/available_dates')
def api_available_dates():
    """Get list of available dates with data."""
    dates = viewer.get_available_dates()
    return jsonify(dates)

@app.route('/api/current_date')
def api_current_date():
    """Get current viewing date."""
    return jsonify({'date': viewer.current_date})

@app.route('/api/switch_date', methods=['POST'])
def api_switch_date():
    """Switch to a different date."""
    data = request.get_json()
    if not data or 'date' not in data:
        return jsonify({'error': 'Date is required'}), 400
    
    date = data['date']
    try:
        viewer.switch_date(date)
        return jsonify({'success': True, 'date': date})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




def main():
    parser = argparse.ArgumentParser(description='Web viewer for motion detection events')
    parser.add_argument('--base-path', default='data',
                       help='Base data path (default: data)')
    parser.add_argument('--date',
                       help='Specific date to view (YYYY-MM-DD format)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=3000,
                       help='Port to bind to (default: 3000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Initialize viewer
    global viewer
    viewer = MotionViewer(base_path=args.base_path, date=args.date)

    print(f"Starting motion viewer on http://{args.host}:{args.port}")
    print(f"Using base path: {args.base_path}")
    print(f"Current date: {viewer.current_date}")
    print(f"Database: {viewer.db_path}")

    # List available dates
    available_dates = viewer.get_available_dates()
    if available_dates:
        print(f"Available dates: {', '.join(available_dates)}")
    else:
        print("No existing data found")

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()