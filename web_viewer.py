#!/usr/bin/env python3

from flask import Flask, render_template, jsonify, request, Response
import os
import sqlite3
import json
from database import DetectionDatabase
import cv2
import argparse
from datetime import datetime

app = Flask(__name__)

class DetectionViewer:
    def __init__(self, db_path: str = "detections.db"):
        self.db = DetectionDatabase(db_path)
        self.db_path = db_path
    
    def _get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_detections_by_date(self, date_filter=None, object_type_filter=None, 
                              limit=100, offset=0):
        """Get detections with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """SELECT id, object_type, time, crop_of_object, original_video_link, 
                              frame_num_original_video, caption, embeddings, confidence, 
                              bbox_x, bbox_y, bbox_width, bbox_height, created_at 
                       FROM detections WHERE 1=1"""
            params = []
            
            if date_filter:
                query += " AND DATE(time) = ?"
                params.append(date_filter)
            
            if object_type_filter:
                query += " AND object_type = ?"
                params.append(object_type_filter)
            
            query += " ORDER BY time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_detection_stats(self):
        """Get statistics about detections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT object_type, COUNT(*) as count 
                FROM detections 
                GROUP BY object_type 
                ORDER BY count DESC
            ''')
            object_counts = cursor.fetchall()
            
            cursor.execute('''
                SELECT DATE(time) as date, COUNT(*) as count 
                FROM detections 
                GROUP BY DATE(time) 
                ORDER BY date DESC 
                LIMIT 30
            ''')
            date_counts = cursor.fetchall()
            
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_count = cursor.fetchone()[0]
            
            return {
                'object_counts': object_counts,
                'date_counts': date_counts,
                'total_count': total_count
            }
    
    def get_tracks_by_date(self, date_filter=None, object_type_filter=None, 
                          limit=100, offset=0):
        """Get tracks with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM tracks WHERE 1=1"
            params = []
            
            if date_filter:
                query += " AND DATE(start_time) = ?"
                params.append(date_filter)
            
            if object_type_filter:
                query += " AND object_type = ?"
                params.append(object_type_filter)
            
            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_track_stats(self):
        """Get statistics about tracks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT object_type, COUNT(*) as count 
                FROM tracks 
                GROUP BY object_type 
                ORDER BY count DESC
            ''')
            track_counts = cursor.fetchall()
            
            cursor.execute('''
                SELECT DATE(start_time) as date, COUNT(*) as count 
                FROM tracks 
                GROUP BY DATE(start_time) 
                ORDER BY date DESC 
                LIMIT 30
            ''')
            track_date_counts = cursor.fetchall()
            
            cursor.execute('SELECT COUNT(*) FROM tracks')
            total_tracks = cursor.fetchone()[0]
            
            return {
                'track_counts': track_counts,
                'track_date_counts': track_date_counts,
                'total_tracks': total_tracks
            }
    

# Initialize viewer
viewer = DetectionViewer()

@app.route('/')
def index():
    """Main detection viewer page."""
    return render_template('index.html')

@app.route('/api/detections')
def api_detections():
    """API endpoint to get detections with filtering."""
    date_filter = request.args.get('date')
    object_type = request.args.get('type')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    detections = viewer.get_detections_by_date(
        date_filter=date_filter,
        object_type_filter=object_type,
        limit=limit,
        offset=offset
    )
    
    # Convert to JSON-serializable format
    detection_list = []
    for det in detections:
        detection_data = {
            'id': det[0],
            'object_type': det[1],
            'time': det[2],
            'original_video_link': det[4],
            'frame_num': det[5],
            'caption': det[6],
            'confidence': det[8],
            'bbox_x': det[9],
            'bbox_y': det[10],
            'bbox_width': det[11],
            'bbox_height': det[12],
            'created_at': det[13]
        }
        detection_list.append(detection_data)
    
    return jsonify(detection_list)

@app.route('/api/stats')
def api_stats():
    """API endpoint to get detection statistics."""
    stats = viewer.get_detection_stats()
    track_stats = viewer.get_track_stats()
    # Combine both stats
    stats.update(track_stats)
    return jsonify(stats)

@app.route('/api/tracks')
def api_tracks():
    """API endpoint to get tracks with filtering."""
    date_filter = request.args.get('date')
    object_type = request.args.get('type')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    tracks = viewer.get_tracks_by_date(
        date_filter=date_filter,
        object_type_filter=object_type,
        limit=limit,
        offset=offset
    )
    
    # Convert to JSON-serializable format
    track_list = []
    for track in tracks:
        track_data = {
            'id': track[0],
            'object_type': track[1],
            'original_video_link': track[2],
            'recorded_video_path': track[3],
            'start_frame': track[4],
            'end_frame': track[5],
            'start_time': track[6],
            'end_time': track[7],
            'track_data': json.loads(track[8]),  # Parse JSON string
            'best_crop_detection_id': track[9],
            'avg_confidence': track[10],
            'detection_count': track[11],
            'created_at': track[12]
        }
        track_list.append(track_data)
    
    return jsonify(track_list)

@app.route('/api/track/<int:track_id>')
def api_track_detail(track_id):
    """API endpoint to get detailed track information."""
    track = viewer.db.get_track_by_id(track_id)
    if not track:
        return jsonify({'error': 'Track not found'}), 404
    
    track_data = {
        'id': track[0],
        'object_type': track[1],
        'original_video_link': track[2],
        'recorded_video_path': track[3],
        'start_frame': track[4],
        'end_frame': track[5],
        'start_time': track[6],
        'end_time': track[7],
        'track_data': json.loads(track[8]),  # Parse JSON string
        'best_crop_detection_id': track[9],
        'avg_confidence': track[10],
        'detection_count': track[11],
        'created_at': track[12]
    }
    
    return jsonify(track_data)

@app.route('/api/track/<int:track_id>/image')
def api_track_image(track_id):
    """Get track's best crop image."""
    track = viewer.db.get_track_by_id(track_id)
    if not track:
        return "Track not found", 404
    
    best_detection_id = track[9]  # best_crop_detection_id
    if not best_detection_id:
        return "No detection found for track", 404
    
    detection = viewer.db.get_detection_by_id(best_detection_id)
    if not detection:
        return "Detection not found", 404
    
    crop_bytes = detection[3]  # crop_of_object
    return Response(crop_bytes, mimetype='image/jpeg')

@app.route('/api/detection/<int:detection_id>/image')
def api_detection_image(detection_id):
    """Get detection crop image."""
    detection = viewer.db.get_detection_by_id(detection_id)
    if not detection:
        return "Detection not found", 404
    
    crop_bytes = detection[3]  # crop_of_object
    return Response(crop_bytes, mimetype='image/jpeg')


@app.route('/api/detection/<int:detection_id>/video')
def api_detection_video(detection_id):
    """Get video segment around detection."""
    # Get detection info first to calculate correct video path based on timestamp
    detection = viewer.db.get_detection_by_id(detection_id)
    if not detection:
        return jsonify({'error': 'Detection not found'}), 404
    
    # Calculate correct video path based on detection timestamp
    detection_timestamp = detection[2]  # timestamp
    detection_datetime = datetime.fromisoformat(detection_timestamp.replace('Z', '+00:00')) if isinstance(detection_timestamp, str) else detection_timestamp
    
    # Generate the expected video file path based on detection time
    date_str = detection_datetime.strftime("%Y-%m-%d")
    hour_str = detection_datetime.strftime("%H")
    min_str = detection_datetime.strftime("%M")
    expected_video_path = f"videos/{date_str}/{hour_str}/{min_str}.mp4"
    
    if os.path.exists(expected_video_path):
        video_path = expected_video_path
    else:
        # Fall back to original video link
        video_path = detection[4]  # original_video_link
    
    frame_num = detection[5] if detection else 0   # frame_num_original_video
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    # Calculate time offsets based on detection timestamp within the video file
    detection_timestamp = detection[2]  # timestamp
    detection_datetime = datetime.fromisoformat(detection_timestamp.replace('Z', '+00:00')) if isinstance(detection_timestamp, str) else detection_timestamp
    
    # Calculate the video file's start time (start of the minute)
    video_start_time = detection_datetime.replace(second=0, microsecond=0)
    
    # Calculate detection time as offset within the video file
    time_diff = detection_datetime - video_start_time
    detection_time = time_diff.total_seconds()
    
    start_time = max(0, detection_time - 5)
    end_time = min(total_frames / fps if fps > 0 else 60, detection_time + 5)
    
    # Detection is already loaded above
    
    bbox = None
    if detection:
        bbox = {
            'x': detection[9],    # bbox_x
            'y': detection[10],   # bbox_y  
            'width': detection[11],   # bbox_width
            'height': detection[12]   # bbox_height
        }
    
    return jsonify({
        'video_path': os.path.basename(video_path),
        'detection_time': detection_time,
        'start_time': start_time,
        'end_time': end_time,
        'fps': fps,
        'total_frames': total_frames,
        'stream_url': f'/api/detection/{detection_id}/video_stream',
        'bbox': bbox
    })

@app.route('/api/detection/<int:detection_id>/video_stream')
def api_detection_video_stream(detection_id):
    """Stream video segment around detection."""
    # Get detection info first to calculate correct video path based on timestamp
    detection = viewer.db.get_detection_by_id(detection_id)
    if not detection:
        return "Detection not found", 404
    
    # Calculate correct video path based on detection timestamp
    detection_timestamp = detection[2]  # timestamp
    detection_datetime = datetime.fromisoformat(detection_timestamp.replace('Z', '+00:00')) if isinstance(detection_timestamp, str) else detection_timestamp
    
    # Generate the expected video file path based on detection time
    date_str = detection_datetime.strftime("%Y-%m-%d")
    hour_str = detection_datetime.strftime("%H")
    min_str = detection_datetime.strftime("%M")
    expected_video_path = f"videos/{date_str}/{hour_str}/{min_str}.mp4"
    
    if os.path.exists(expected_video_path):
        video_path = expected_video_path
    else:
        # Fall back to original video link
        video_path = detection[4]  # original_video_link
    
    if not os.path.exists(video_path):
        return "Video file not found", 404
    
    # Get range header for partial content support
    range_header = request.headers.get('Range')
    
    # For video streaming, return the actual video file with range support
    # This is a simplified approach - for production, consider using dedicated streaming
    try:
        def generate():
            with open(video_path, 'rb') as f:
                data = f.read(1024)
                while data:
                    yield data
                    data = f.read(1024)
        
        file_size = os.path.getsize(video_path)
        
        if range_header:
            # Handle partial content requests
            ranges = range_header.replace('bytes=', '').split('-')
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if ranges[1] else file_size - 1
            
            def generate_range():
                with open(video_path, 'rb') as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk_size = min(1024, remaining)
                        data = f.read(chunk_size)
                        if not data:
                            break
                        yield data
                        remaining -= len(data)
            
            return Response(
                generate_range(),
                206,  # Partial Content
                {
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(end - start + 1),
                    'Content-Type': 'video/mp4',
                }
            )
        else:
            return Response(
                generate(),
                200,
                {
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(file_size),
                    'Accept-Ranges': 'bytes',
                }
            )
            
    except Exception as e:
        return f"Error streaming video: {str(e)}", 500

@app.route('/api/object_types')
def api_object_types():
    """Get list of all object types in database."""
    with viewer._get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM tracks')
        track_count = cursor.fetchone()[0]
        
        if track_count > 0:
            cursor.execute('SELECT DISTINCT object_type FROM tracks ORDER BY object_type')
        else:
            cursor.execute('SELECT DISTINCT object_type FROM detections ORDER BY object_type')
        
        types = [row[0] for row in cursor.fetchall()]
        
        return jsonify(types)




def main():
    parser = argparse.ArgumentParser(description='Web viewer for detections')
    parser.add_argument('--db', default='detections.db',
                       help='Database path (default: detections.db)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=3000,
                       help='Port to bind to (default: 3000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize viewer with custom database
    global viewer
    viewer = DetectionViewer(args.db)
    
    print(f"Starting detection viewer on http://{args.host}:{args.port}")
    print(f"Using database: {args.db}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()