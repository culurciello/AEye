#!/usr/bin/env python3
"""
AI Detection Service - Modular YOLO detection processor
Integrates with existing ai_agent.py for semantic search capabilities
"""

from pathlib import Path
from datetime import datetime
import queue
import threading
import cv2
import numpy as np
from typing import Optional, List, Dict, Callable

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from database import DetectionDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from ai_agent import SemanticSearchAgent
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False


class DetectionService:
    """
    Modular AI detection service that can process frames from any source
    Uses existing ai_agent.py for semantic search capabilities
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 db_path: str = "detections.db",
                 top_n_detections: int = 10,
                 enable_semantic_search: bool = True):
        
        self.confidence_threshold = confidence_threshold
        self.top_n_detections = top_n_detections
        self.db_path = db_path
        self.enable_semantic_search = enable_semantic_search
        
        # Initialize YOLO model
        self.model = None
        if YOLO_AVAILABLE:
            try:
                if not Path(model_path).exists():
                    model_path = Path("models") / model_path
                self.model = YOLO(str(model_path))
                print(f"YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"Warning: Could not load YOLO model {model_path}: {e}")
        else:
            print("Warning: YOLO not available. Detection disabled.")
        
        # Initialize database
        self.db = None
        if DATABASE_AVAILABLE:
            try:
                self.db = DetectionDatabase(db_path)
                print(f"Database initialized: {db_path}")
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")
        
        # Initialize semantic search agent
        self.semantic_agent = None
        if self.enable_semantic_search and SEMANTIC_SEARCH_AVAILABLE and self.db:
            try:
                self.semantic_agent = SemanticSearchAgent(db_path)
                print("Semantic search agent initialized")
            except Exception as e:
                print(f"Warning: Could not initialize semantic search: {e}")
        
        # Target object classes (COCO dataset)
        self.target_classes = {
            0: 'person',
            2: 'car', 
            5: 'bus',
            15: 'cat',
            16: 'dog'
        }
        self.target_class_names = set(self.target_classes.values())
        
        # Detection storage
        self.detections = []
        self.frame_count = 0
        
        # Processing callbacks
        self.detection_callbacks: List[Callable] = []
    
    def add_detection_callback(self, callback: Callable[[Dict], None]):
        """Add callback to be called when detection is found"""
        self.detection_callbacks.append(callback)
    
    def process_frame(self, frame: np.ndarray, source_info: Dict = None) -> List[Dict]:
        """
        Process a single frame for object detection
        
        Args:
            frame: BGR image array
            source_info: Optional metadata about frame source
            
        Returns:
            List of detection dictionaries
        """
        if not self.model:
            return []
        
        frame_detections = []
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Process detections
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
                            detection = self._create_detection(
                                frame, box, class_name, confidence, source_info
                            )
                            if detection:
                                frame_detections.append(detection)
                                self.detections.append(detection)
                                
                                # Call detection callbacks
                                for callback in self.detection_callbacks:
                                    callback(detection)
        
        except Exception as e:
            print(f"Error in detection processing: {e}")
        
        self.frame_count += 1
        return frame_detections
    
    def _create_detection(self, frame: np.ndarray, box, class_name: str, 
                         confidence: float, source_info: Dict = None) -> Optional[Dict]:
        """Create detection dictionary from YOLO box"""
        try:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # Check minimum size requirement (100 pixels)
            width = x2 - x1
            height = y2 - y1
            if width < 100 or height < 100:
                return None
            
            # Expand bounding box by 25% on each side
            padding_x = int(width * 0.25)
            padding_y = int(height * 0.25)
            
            # Get frame dimensions for boundary checking
            frame_height, frame_width = frame.shape[:2]
            
            # Apply padding with boundary checks
            x1_expanded = max(0, x1 - padding_x)
            y1_expanded = max(0, y1 - padding_y)
            x2_expanded = min(frame_width, x2 + padding_x)
            y2_expanded = min(frame_height, y2 + padding_y)
            
            # Crop the detected object with expanded boundaries
            crop = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            # Convert image to bytes for storage
            if self.db:
                crop_bytes = self.db.image_to_bytes(crop)
            else:
                _, buffer = cv2.imencode('.jpg', crop)
                crop_bytes = buffer.tobytes()
            
            # Create detection data
            detection_data = {
                'object_type': class_name,
                'timestamp': datetime.now(),
                'crop_image': crop_bytes,
                'original_video_link': source_info.get('source', 'unknown') if source_info else 'unknown',
                'frame_num': self.frame_count,
                'confidence': confidence,
                'bbox': (x1, y1, width, height),
                'crop_array': crop,  # For real-time display
                'expanded_bbox': (x1_expanded, y1_expanded, x2_expanded, y2_expanded)
            }
            
            return detection_data
            
        except Exception as e:
            print(f"Error creating detection: {e}")
            return None
    
    def save_detections(self) -> int:
        """Save collected detections to database with semantic indexing"""
        if not self.db or not self.detections:
            return 0
        
        print(f"Processing {len(self.detections)} detections...")
        
        # Group detections by object type and keep top N for each type
        detections_by_type = {}
        for detection in self.detections:
            obj_type = detection['object_type']
            if obj_type not in detections_by_type:
                detections_by_type[obj_type] = []
            detections_by_type[obj_type].append(detection)
        
        # Sort each group by confidence and keep top N
        top_detections = []
        for obj_type, obj_detections in detections_by_type.items():
            obj_detections.sort(key=lambda x: x['confidence'], reverse=True)
            top_detections.extend(obj_detections[:self.top_n_detections])
        
        # Save detections to database
        saved_count = 0
        for detection_data in top_detections:
            try:
                detection_id = self.db.save_detection(
                    object_type=detection_data['object_type'],
                    timestamp=detection_data['timestamp'],
                    crop_image=detection_data['crop_image'],
                    original_video_link=detection_data['original_video_link'],
                    frame_num=detection_data['frame_num'],
                    confidence=detection_data['confidence'],
                    bbox=detection_data['bbox']
                )
                detection_data['id'] = detection_id
                saved_count += 1
                print(f"Saved detection {detection_id}: {detection_data['object_type']} "
                      f"(conf: {detection_data['confidence']:.2f}) at frame {detection_data['frame_num']}")
            except Exception as e:
                print(f"Error saving detection: {e}")
        
        print(f"Successfully saved {saved_count} detections to database")
        
        # Index captions for semantic search if agent is available
        if self.semantic_agent:
            try:
                print("Indexing detections for semantic search...")
                indexed_count = self.semantic_agent.index_captions()
                print(f"Indexed {indexed_count} captions for semantic search")
            except Exception as e:
                print(f"Error indexing captions: {e}")
        
        # Create tracks from detections
        self._create_and_save_tracks()
        
        return saved_count
    
    def _create_and_save_tracks(self):
        """Create and save object tracks from detections"""
        if not self.db or not self.detections:
            return
        
        print("Creating tracks from detections...")
        tracks = self._create_tracks_from_detections(self.detections, str(self.detections[0].get('original_video_link', 'unknown')))
        
        print(f"Created {len(tracks)} tracks from {len(self.detections)} detections")
        for track in tracks:
            print(f"Track {track['id']}: {track['object_type']} "
                  f"(frames {track['start_frame']}-{track['end_frame']}, "
                  f"avg conf: {track['avg_confidence']:.2f})")
    
    def _create_tracks_from_detections(self, detections, video_path):
        """Create tracks by grouping detections of same objects across consecutive frames"""
        if not detections:
            return []
        
        # Sort detections by object type and frame number
        detections.sort(key=lambda x: (x['object_type'], x['frame_num']))
        
        tracks = []
        current_tracks = {}  # object_type -> current track data
        
        for detection in detections:
            obj_type = detection['object_type']
            frame_num = detection['frame_num']
            bbox = detection['bbox']
            confidence = detection['confidence']
            timestamp = detection['timestamp']
            
            # Check if this continues an existing track
            track_found = False
            if obj_type in current_tracks:
                last_track = current_tracks[obj_type]
                
                # Check if this detection continues the track (consecutive or nearby frames)
                frame_gap = frame_num - last_track['end_frame']
                
                if frame_gap <= 3:  # Allow up to 3 frame gap
                    # Check spatial overlap (simple bounding box distance)
                    last_bbox = last_track['track_data'][str(last_track['end_frame'])]
                    bbox_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                    last_center = (last_bbox[0] + last_bbox[2]/2, last_bbox[1] + last_bbox[3]/2)
                    distance = ((bbox_center[0] - last_center[0])**2 + 
                               (bbox_center[1] - last_center[1])**2)**0.5
                    
                    # If objects are close enough, continue track
                    if distance < 100:  # pixels threshold
                        last_track['end_frame'] = frame_num
                        last_track['end_time'] = timestamp
                        last_track['track_data'][str(frame_num)] = bbox
                        last_track['confidences'].append(confidence)
                        last_track['detections'].append(detection)
                        track_found = True
            
            # If no existing track found, start new track
            if not track_found:
                # Save previous track if it exists
                if obj_type in current_tracks:
                    tracks.append(self._finalize_track(current_tracks[obj_type], video_path))
                
                # Start new track
                current_tracks[obj_type] = {
                    'object_type': obj_type,
                    'start_frame': frame_num,
                    'end_frame': frame_num,
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'track_data': {str(frame_num): bbox},
                    'confidences': [confidence],
                    'detections': [detection]
                }
        
        # Finalize remaining tracks
        for obj_type, track_data in current_tracks.items():
            tracks.append(self._finalize_track(track_data, video_path))
        
        return tracks
    
    def _finalize_track(self, track_data, video_path):
        """Finalize and save a track to database"""
        # Find best detection (highest confidence)
        best_idx = max(range(len(track_data['confidences'])), 
                      key=lambda i: track_data['confidences'][i])
        best_detection = track_data['detections'][best_idx]
        
        # Calculate average confidence
        avg_confidence = sum(track_data['confidences']) / len(track_data['confidences'])
        
        # Save best detection to database if not already saved
        if 'id' not in best_detection:
            detection_id = self.db.save_detection(
                object_type=best_detection['object_type'],
                timestamp=best_detection['timestamp'],
                crop_image=best_detection['crop_image'],
                original_video_link=best_detection['original_video_link'],
                frame_num=best_detection['frame_num'],
                confidence=best_detection['confidence'],
                bbox=best_detection['bbox']
            )
            best_detection['id'] = detection_id
        
        # Save track to database
        import json
        track_id = self.db.save_track(
            object_type=track_data['object_type'],
            original_video_link=video_path,
            start_frame=track_data['start_frame'],
            end_frame=track_data['end_frame'],
            start_time=track_data['start_time'],
            end_time=track_data['end_time'],
            track_data=track_data['track_data'],
            best_crop_detection_id=best_detection['id'],
            avg_confidence=avg_confidence,
            detection_count=len(track_data['detections'])
        )
        
        return {
            'id': track_id,
            'object_type': track_data['object_type'],
            'start_frame': track_data['start_frame'],
            'end_frame': track_data['end_frame'],
            'avg_confidence': avg_confidence,
            'detection_count': len(track_data['detections'])
        }
    
    def search_detections(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search detections using semantic search"""
        if not self.semantic_agent:
            print("Semantic search not available")
            return []
        
        try:
            results = self.semantic_agent.search(query, top_k=top_k)
            print(f"Found {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'total_frames': self.frame_count,
            'total_detections': len(self.detections),
            'detections_by_type': {}
        }
        
        # Count detections by type
        for detection in self.detections:
            obj_type = detection['object_type']
            stats['detections_by_type'][obj_type] = stats['detections_by_type'].get(obj_type, 0) + 1
        
        return stats
    
    def reset(self):
        """Reset detection counters and storage"""
        self.detections = []
        self.frame_count = 0
    
    def is_available(self) -> bool:
        """Check if detection service is ready"""
        return self.model is not None and self.db is not None


def main():
    """Test the detection service standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Detection Service Test')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--db', default='detections.db', help='Database path')
    parser.add_argument('--test-image', help='Test image path')
    parser.add_argument('--search', help='Search query to test semantic search')
    
    args = parser.parse_args()
    
    # Initialize service
    service = DetectionService(
        model_path=args.model,
        confidence_threshold=args.confidence,
        db_path=args.db
    )
    
    if not service.is_available():
        print("Detection service not available")
        return
    
    if args.test_image:
        # Test detection on image
        print(f"Testing detection on: {args.test_image}")
        
        frame = cv2.imread(args.test_image)
        if frame is None:
            print(f"Could not load image: {args.test_image}")
            return
        
        detections = service.process_frame(frame, {'source': args.test_image})
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  {det['object_type']}: {det['confidence']:.2f}")
        
        # Save detections
        saved = service.save_detections()
        print(f"Saved {saved} detections")
    
    if args.search:
        # Test semantic search
        print(f"Searching for: {args.search}")
        results = service.search_detections(args.search)
        
        for i, (detection, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.3f} - {detection['object_type']} (conf: {detection['confidence']:.2f})")


if __name__ == "__main__":
    main()