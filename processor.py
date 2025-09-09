#!/usr/bin/env python3

import cv2
import argparse
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import sys
import os
from database import DetectionDatabase
from tqdm import tqdm


class VideoParser:
    def __init__(self, model_path: str = "yolov8n.pt", db_path: str = "detections.db"):
        self.model = YOLO(model_path)
        self.db = DetectionDatabase(db_path)
        
        # Target object classes (COCO dataset class names)
        self.target_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            15: 'cat',
            16: 'dog',
            14: 'bird',
            24: "backpack",
            25: "umbrella",
            26: "handbag",
        }
        
        self.target_class_names = set(self.target_classes.values())
        
        # Track management
        self.active_tracks = {}  # object_type -> list of active tracks
        self.next_track_id = 1
        self.max_track_gap = 30  # frames
        self.min_track_length = 3  # minimum detections per track
        
        # Video recording state
        self.current_recorded_video_path = None
    
    def _create_video_path(self, timestamp: datetime) -> str:
        """Create video file path with format videos/date/hour/min.mp4"""
        date_str = timestamp.strftime("%Y-%m-%d")
        hour_str = timestamp.strftime("%H")
        min_str = timestamp.strftime("%M")
        
        # Create directory structure
        video_dir = Path(f"videos/{date_str}/{hour_str}")
        video_dir.mkdir(parents=True, exist_ok=True)
        
        return str(video_dir / f"{min_str}.mp4")
    
    def _setup_video_writer(self, cap, output_path: str):
        """Setup video writer to record stream without re-encoding"""
        # Get video properties from input
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use H.264 codec for efficient compression
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        # Create video writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print(f"Warning: Could not open video writer for {output_path}")
            return None
            
        print(f"Recording video to: {output_path}")
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        return writer
    
    def process_input(self, input_source: str, confidence_threshold: float = 0.5, 
                     is_stream: bool = False, max_seconds: int = None,
                     continuous: bool = False, show_live: bool = False, save_tracks: bool = True, record_video: bool = False):
        
        # if camera convert to int
        if is_stream and input_source.isdigit():
            input_source = int(input_source)

        """Process video file or stream and detect objects."""
        cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open {'stream' if is_stream else 'video'} {input_source}")
            return
        
        frame_count = 0
        start_time_processing = datetime.now() if is_stream and max_seconds else None
        
        # Video recording setup
        video_writer = None
        current_video_path = None
        recording_start_time = None
        
        if record_video and is_stream:
            recording_start_time = datetime.now()
            current_video_path = self._create_video_path(recording_start_time)
            self.current_recorded_video_path = current_video_path
            video_writer = self._setup_video_writer(cap, current_video_path)
        
        # Get video properties for files
        if not is_stream:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing video: {input_source}")
            print(f"FPS: {fps}, Total frames: {total_frames}")
            pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        else:
            fps = None
            print(f"Processing stream: {input_source}")
            print("Press 'q' to quit")
            pbar = tqdm(desc="Processing stream", unit="frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if is_stream:
                        print("Failed to read frame from stream")
                    break
                
                # Calculate timestamp
                if is_stream:
                    timestamp = datetime.now()
                else:
                    timestamp = datetime.now().replace(
                        microsecond=int((frame_count / fps) * 1000000) % 1000000
                    )
                
                # Handle video recording for streams
                if record_video and is_stream and video_writer is not None:
                    # Check if we need to start a new video file (every minute)
                    if recording_start_time and timestamp.minute != recording_start_time.minute:
                        # Close current video writer
                        video_writer.release()
                        print(f"Finished recording: {current_video_path}")
                        
                        # Start new video file
                        recording_start_time = timestamp
                        current_video_path = self._create_video_path(recording_start_time)
                        self.current_recorded_video_path = current_video_path
                        video_writer = self._setup_video_writer(cap, current_video_path)
                    
                    # Write frame to video file
                    if video_writer and video_writer.isOpened():
                        video_writer.write(frame)
                
                # Run YOLO detection
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                
                # Create a copy for display if showing live
                display_frame = frame.copy() if show_live else None
                
                # Process detections and update tracks
                frame_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class info
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Only process target classes
                            if class_name in self.target_class_names:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                                bbox = (x1, y1, x2 - x1, y2 - y1)
                                
                                # Expand crop bounds by 25% on each side
                                crop_w = x2 - x1
                                crop_h = y2 - y1
                                expand_w = int(crop_w * 0.25)
                                expand_h = int(crop_h * 0.25)
                                
                                # Calculate expanded coordinates with frame bounds checking
                                frame_h, frame_w = frame.shape[:2]
                                crop_x1 = max(0, x1 - expand_w)
                                crop_y1 = max(0, y1 - expand_h)
                                crop_x2 = min(frame_w, x2 + expand_w)
                                crop_y2 = min(frame_h, y2 + expand_h)
                                
                                # Crop the detected object with expanded bounds
                                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                
                                # Convert images to bytes
                                crop_bytes = self.db.image_to_bytes(crop)
                                
                                # Store detection data
                                detection_data = {
                                    'object_type': class_name,
                                    'caption': class_name,
                                    'timestamp': timestamp,
                                    'crop_image': crop_bytes,
                                    'original_video_link': input_source,
                                    'frame_num': frame_count,
                                    'confidence': confidence,
                                    'bbox': bbox
                                }
                                
                                frame_detections.append(detection_data)
                                
                                # Draw bounding box and label on display frame
                                if show_live and display_frame is not None:
                                    # Draw bounding box
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw label with confidence
                                    label = f"{class_name}: {confidence:.2f}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    
                                    # Background rectangle for label
                                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                                    
                                    # Label text
                                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Update tracks with current frame detections
                self._update_tracks(frame_detections, frame_count, save_tracks, show_live)
                
                
                frame_count += 1
                pbar.update(1)
                
                # Show live frame with detections if enabled
                if show_live and display_frame is not None:
                    try:
                        # Add frame info including active tracks
                        total_active_tracks = sum(len(tracks) for tracks in self.active_tracks.values())
                        info_text = f"Frame: {frame_count}, Detections: {len(frame_detections)}, Active Tracks: {total_active_tracks}"
                        cv2.putText(display_frame, info_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Resize frame for display if too large
                        height, width = display_frame.shape[:2]
                        if width > 1280 or height > 720:
                            scale = min(1280/width, 720/height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            display_frame = cv2.resize(display_frame, (new_width, new_height))
                        
                        cv2.imshow('Live Detection', display_frame)
                        
                        # Exit on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\\nStopping processing (q pressed)")
                            break
                    except cv2.error as e:
                        if frame_count == 1:  # Only warn once
                            print(f"Warning: Cannot display video (no GUI available): {e}")
                            print("Continuing processing without live display...")
                        show_live = False  # Disable live display for remaining frames
                
                # Check stream-specific conditions
                if is_stream and max_seconds and start_time_processing:
                    elapsed_seconds = (datetime.now() - start_time_processing).total_seconds()
                    if elapsed_seconds >= max_seconds:
                        print(f"\nStopping processing after {elapsed_seconds:.1f} seconds (max: {max_seconds}s)")
                        break
        
        except KeyboardInterrupt:
            if is_stream:
                print("\\nStream processing interrupted by user")
        finally:
            pbar.close()
            cap.release()
            
            # Clean up video writer
            if video_writer is not None:
                video_writer.release()
                if current_video_path:
                    print(f"Finished recording: {current_video_path}")
            
            if show_live:
                cv2.destroyAllWindows()
    
        # Finalize all remaining tracks
        if save_tracks:
            completed_tracks = self._finalize_all_tracks(input_source)
            print(f"Processing complete. Total tracks saved: {completed_tracks}")

    def process_video(self, video_path: str, confidence_threshold: float = 0.5, continuous: bool = False, show_live: bool = False, save_tracks: bool = True):
        """Process a video file and detect objects."""
        self.process_input(video_path, confidence_threshold, is_stream=False, continuous=continuous, show_live=show_live, save_tracks=save_tracks)
    
    def process_stream(self, stream_url: str, confidence_threshold: float = 0.5, 
                      max_seconds: int = None, continuous: bool = False, show_live: bool = False, save_tracks: bool = True, record_video: bool = False):
        """Process a video stream and detect objects."""
        self.process_input(stream_url, confidence_threshold, is_stream=True, max_seconds=max_seconds, continuous=continuous, show_live=show_live, save_tracks=save_tracks, record_video=record_video)
    
    def _update_tracks(self, frame_detections, frame_num, save_tracks, show_live):
        """Update active tracks with current frame detections."""
        tracks_updated = 0
        tracks_created = 0
        tracks_completed = 0
        
        # Group detections by object type
        detections_by_type = {}
        for detection in frame_detections:
            obj_type = detection['object_type']
            if obj_type not in detections_by_type:
                detections_by_type[obj_type] = []
            detections_by_type[obj_type].append(detection)
        
        # Update tracks for each object type
        for obj_type, obj_detections in detections_by_type.items():
            if obj_type not in self.active_tracks:
                self.active_tracks[obj_type] = []
            
            # Try to match detections to existing tracks
            matched_detections = set()
            
            for track in self.active_tracks[obj_type][:]:
                if track['end_frame'] < frame_num - self.max_track_gap:
                    # Track is too old, finalize it
                    if save_tracks and len(track['detections']) >= self.min_track_length:
                        self._save_track_to_db(track)
                        tracks_completed += 1
                    self.active_tracks[obj_type].remove(track)
                    continue
                
                # Find best matching detection for this track
                best_match = None
                best_distance = float('inf')
                
                for i, detection in enumerate(obj_detections):
                    if i in matched_detections:
                        continue
                    
                    # Calculate distance between track's last detection and current detection
                    last_bbox = track['detections'][-1]['bbox']
                    current_bbox = detection['bbox']
                    
                    last_center = (last_bbox[0] + last_bbox[2]/2, last_bbox[1] + last_bbox[3]/2)
                    current_center = (current_bbox[0] + current_bbox[2]/2, current_bbox[1] + current_bbox[3]/2)
                    
                    distance = ((last_center[0] - current_center[0])**2 + 
                               (last_center[1] - current_center[1])**2)**0.5
                    
                    if distance < best_distance and distance < 100:  # 100 pixel threshold
                        best_match = i
                        best_distance = distance
                
                # Update track with best match
                if best_match is not None:
                    detection = obj_detections[best_match]
                    track['end_frame'] = frame_num
                    track['end_time'] = detection['timestamp']
                    track['detections'].append(detection)
                    track['track_data'][str(frame_num)] = detection['bbox']
                    matched_detections.add(best_match)
                    tracks_updated += 1
            
            # Create new tracks for unmatched detections
            for i, detection in enumerate(obj_detections):
                if i not in matched_detections:
                    new_track = {
                        'id': self.next_track_id,
                        'object_type': obj_type,
                        'start_frame': frame_num,
                        'end_frame': frame_num,
                        'start_time': detection['timestamp'],
                        'end_time': detection['timestamp'],
                        'detections': [detection],
                        'track_data': {str(frame_num): detection['bbox']}
                    }
                    self.active_tracks[obj_type].append(new_track)
                    self.next_track_id += 1
                    tracks_created += 1
        
        # Print track status occasionally
        if show_live and frame_num % 60 == 0:  # Every 60 frames
            total_active = sum(len(tracks) for tracks in self.active_tracks.values())
            if tracks_created > 0 or tracks_updated > 0 or tracks_completed > 0:
                print(f"Frame {frame_num}: Created {tracks_created}, Updated {tracks_updated}, Completed {tracks_completed}, Active {total_active}")
    
    def _save_track_to_db(self, track):
        """Save a track to the database."""
        # Find best detection (highest confidence)
        best_detection = max(track['detections'], key=lambda x: x['confidence'])
        
        # Save best detection first
        detection_id = self.db.save_detection(**best_detection)
        
        # Calculate average confidence
        avg_confidence = sum(d['confidence'] for d in track['detections']) / len(track['detections'])
        
        # Calculate the correct recorded video path based on track start time
        recorded_video_path = None
        if self.current_recorded_video_path:  # Only if we're recording
            track_start_time = track['start_time']
            date_str = track_start_time.strftime("%Y-%m-%d")
            hour_str = track_start_time.strftime("%H")
            min_str = track_start_time.strftime("%M")
            recorded_video_path = f"videos/{date_str}/{hour_str}/{min_str}.mp4"
        
        # Save track
        track_id = self.db.save_track(
            object_type=track['object_type'],
            original_video_link=best_detection['original_video_link'],
            start_frame=track['start_frame'],
            end_frame=track['end_frame'],
            start_time=track['start_time'],
            end_time=track['end_time'],
            track_data=track['track_data'],
            best_crop_detection_id=detection_id,
            avg_confidence=avg_confidence,
            detection_count=len(track['detections']),
            recorded_video_path=recorded_video_path
        )
        
        return track_id
    
    def _finalize_all_tracks(self, input_source):
        """Finalize and save all remaining active tracks."""
        total_saved = 0
        
        for obj_type, tracks in self.active_tracks.items():
            for track in tracks[:]:
                if len(track['detections']) >= self.min_track_length:
                    self._save_track_to_db(track)
                    total_saved += 1
        
        # Clear all active tracks
        self.active_tracks = {}
        
        return total_saved



def main():
    parser = argparse.ArgumentParser(description='Parse video for object detection')
    parser.add_argument('input', help='Video file path, webcam index (e.g., 0), or stream URL (e.g., rtsp://...)')
    parser.add_argument('--model', default='yolov8n.pt', 
                       help='YOLOv8 model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.15,
                       help='Confidence threshold (default: 0.15)')
    parser.add_argument('--db', default='detections.db',
                       help='Database path (default: detections.db)')
    parser.add_argument('--stream', action='store_true',
                       help='Force processing as stream (auto-detected for webcam/URLs)')
    parser.add_argument('--max_seconds', type=int, default=None,
                       help='Maximum seconds to process for streams (default: infinite)')
    parser.add_argument('--continuous', action='store_true',
                       help='Save detections immediately as they are found')
    parser.add_argument('--show-live', action='store_true',
                       help='Display live video with detections during processing')
    parser.add_argument('--record', action='store_true',
                       help='Record video files for streams (saved to videos/date/hour/min.mp4)')

    args = parser.parse_args()
    
    # Initialize video parser
    parser_obj = VideoParser(model_path=args.model, db_path=args.db)
    
    # Auto-detect input type and process accordingly
    input_str = args.input
    
    # Check if input is a webcam index (digit)
    if input_str.isdigit():
        print(f"Detected webcam input: {input_str}")
        parser_obj.process_stream(
            input_str, 
            confidence_threshold=args.confidence,
            max_seconds=args.max_seconds,
            continuous=args.continuous,
            show_live=getattr(args, 'show_live', False),
            save_tracks=True,
            record_video=args.record
        )
    # Check if input is a stream URL (starts with rtsp://, http://, etc.)
    elif input_str.startswith(('rtsp://', 'http://', 'https://', 'tcp://', 'udp://')):
        print(f"Detected stream input: {input_str}")
        parser_obj.process_stream(
            input_str, 
            confidence_threshold=args.confidence,
            max_seconds=args.max_seconds,
            continuous=args.continuous,
            show_live=getattr(args, 'show_live', False),
            save_tracks=True,
            record_video=args.record
        )
    # Check if --stream flag is explicitly set
    elif args.stream:
        print(f"Processing as stream (--stream flag): {input_str}")
        parser_obj.process_stream(
            input_str, 
            confidence_threshold=args.confidence,
            max_seconds=args.max_seconds,
            continuous=args.continuous,
            show_live=getattr(args, 'show_live', False),
            save_tracks=True,
            record_video=args.record
        )
    # Otherwise treat as video file
    else:
        if not Path(input_str).exists():
            print(f"Error: Video file {input_str} does not exist")
            sys.exit(1)
        
        print(f"Detected video file: {input_str}")
        parser_obj.process_video(input_str, confidence_threshold=args.confidence, continuous=args.continuous, show_live=getattr(args, 'show_live', False), save_tracks=True)


if __name__ == "__main__":
    main()