#!/usr/bin/env python3
"""
Smart Video System - Modular orchestrator
Combines video recording, AI detection, and semantic search services
"""

import argparse
import signal
import sys
import time
import cv2
from datetime import datetime
from pathlib import Path

try:
    from video_recording_service import VideoRecordingService
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    print("Warning: video_recording_service not available")

try:
    from detection_service import DetectionService
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    print("Warning: detection_service not available")


class SmartVideoSystem:
    """
    Orchestrator that combines video recording and AI detection services
    Uses modular architecture for better maintainability and testability
    """
    
    def __init__(self,
                 input_source: str,
                 resolution: tuple = (1280, 720),
                 fps: int = 30,
                 output_dir: str = "videos",
                 enable_ai: bool = False,
                 ai_config: dict = None):
        
        self.input_source = input_source
        self.enable_ai = enable_ai
        
        # Initialize video recording service
        self.recording_service = None
        if RECORDING_AVAILABLE:
            self.recording_service = VideoRecordingService(
                input_source=input_source,
                resolution=resolution,
                fps=fps,
                output_dir=output_dir
            )
            print(f"Video recording service initialized")
        
        # Initialize AI detection service
        self.detection_service = None
        if self.enable_ai and DETECTION_AVAILABLE:
            ai_config = ai_config or {}
            self.detection_service = DetectionService(
                model_path=ai_config.get('model_path', 'yolov8n.pt'),
                confidence_threshold=ai_config.get('confidence', 0.5),
                db_path=ai_config.get('db_path', 'detections.db'),
                top_n_detections=ai_config.get('top_n', 10),
                enable_semantic_search=ai_config.get('enable_semantic_search', True)
            )
            
            if self.detection_service.is_available():
                print(f"AI detection service initialized")
                
                # Connect detection service to recording service
                if self.recording_service:
                    self.recording_service.add_frame_callback(self._process_frame_callback)
            else:
                print("Warning: AI detection service not available")
                self.detection_service = None
        
        # Display settings
        self.show_frames = False
        self.show_detections = False
        self.display_info = False
        
        # Statistics
        self.start_time = None
        self.running = False
    
    def _process_frame_callback(self, frame, frame_info):
        """Callback to process frames through AI detection"""
        if self.detection_service:
            detections = self.detection_service.process_frame(frame, frame_info)
            
            # Optional: Store detections in frame_info for display
            frame_info['detections'] = detections
    
    def start(self, show_frames: bool = False, show_detections: bool = False):
        """Start the smart video system"""
        self.show_frames = show_frames
        self.show_detections = show_detections
        self.start_time = datetime.now()
        self.running = True
        
        print("\n" + "="*60)
        print("SMART VIDEO SYSTEM STARTING")
        print("="*60)
        print(f"Input Source: {self.input_source}")
        print(f"AI Detection: {'Enabled' if self.detection_service else 'Disabled'}")
        print(f"Semantic Search: {'Available' if self.detection_service and self.detection_service.semantic_agent else 'Not Available'}")
        print(f"Display Mode: {'On' if show_frames else 'Off'}")
        print("="*60)
        
        # Start recording service
        if self.recording_service:
            enable_streaming = self.show_frames or self.detection_service is not None
            if not self.recording_service.start(enable_streaming=enable_streaming):
                print("Failed to start recording service")
                return False
        
        return True
    
    def run_display_loop(self):
        """Run the display loop if frame display is enabled"""
        if not self.show_frames or not self.recording_service:
            return
        
        print("\nDisplay Controls:")
        print("  'q' - quit")
        print("  's' - save current frame")
        print("  'd' - toggle detection info")
        print("  'i' - toggle statistics info")
        if self.detection_service:
            print("  'r' - show recent detections")
        
        while self.running and self.recording_service.is_running():
            frame_data = self.recording_service.get_frame(timeout=1.0)
            if frame_data is None:
                continue
            
            frame, frame_info = frame_data
            display_frame = frame.copy()
            
            # Add detection overlays
            if self.show_detections and 'detections' in frame_info:
                self._draw_detections(display_frame, frame_info['detections'])
            
            # Add info overlays
            if self.display_info:
                self._draw_info_overlay(display_frame, frame_info)
            
            # Display frame
            cv2.imshow('Smart Video System', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_frame(frame, frame_info)
            elif key == ord('d'):
                self.show_detections = not self.show_detections
                print(f"Detection overlay: {'ON' if self.show_detections else 'OFF'}")
            elif key == ord('i'):
                self.display_info = not self.display_info
                print(f"Info overlay: {'ON' if self.display_info else 'OFF'}")
            elif key == ord('r') and self.detection_service:
                self._show_recent_detections()
        
        cv2.destroyAllWindows()
    
    def _draw_detections(self, frame, detections):
        """Draw detection bounding boxes and labels"""
        for detection in detections:
            if 'expanded_bbox' in detection:
                x1, y1, x2, y2 = detection['expanded_bbox']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection['object_type']}: {detection['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_info_overlay(self, frame, frame_info):
        """Draw system information overlay"""
        info_lines = []
        
        # Recording info
        if self.recording_service:
            stats = self.recording_service.get_statistics()
            info_lines.extend([
                f"Frames: {stats['frame_count']}",
                f"Dropped: {stats['dropped_frames']} ({stats['drop_rate']*100:.1f}%)",
            ])
        
        # AI detection info
        if self.detection_service:
            ai_stats = self.detection_service.get_statistics()
            info_lines.extend([
                f"Detections: {ai_stats['total_detections']}",
                f"Types: {', '.join(ai_stats['detections_by_type'].keys())}"
            ])
        
        # Runtime info
        if self.start_time:
            runtime = datetime.now() - self.start_time
            info_lines.append(f"Runtime: {str(runtime).split('.')[0]}")
        
        # Draw info background
        if info_lines:
            bg_height = len(info_lines) * 25 + 10
            cv2.rectangle(frame, (10, 10), (300, bg_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, bg_height), (255, 255, 255), 1)
            
            # Draw text
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (15, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _save_frame(self, frame, frame_info):
        """Save current frame to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved: {filename}")
    
    def _show_recent_detections(self):
        """Show recent detections summary"""
        if not self.detection_service:
            return
        
        stats = self.detection_service.get_statistics()
        print("\n" + "="*40)
        print("RECENT DETECTIONS")
        print("="*40)
        print(f"Total detections: {stats['total_detections']}")
        
        for obj_type, count in stats['detections_by_type'].items():
            print(f"  {obj_type}: {count}")
        
        # Show last few detections
        recent = self.detection_service.detections[-5:] if self.detection_service.detections else []
        if recent:
            print("\nLast 5 detections:")
            for i, det in enumerate(recent, 1):
                print(f"  {i}. {det['object_type']} (conf: {det['confidence']:.2f}) "
                      f"at frame {det['frame_num']}")
        print("="*40)
    
    def search_detections(self, query: str, top_k: int = 10):
        """Search detections using semantic search"""
        if not self.detection_service:
            print("AI detection service not available")
            return []
        
        results = self.detection_service.search_detections(query, top_k)
        
        if results:
            print(f"\nSearch results for: '{query}'")
            print("-" * 50)
            for i, (detection, score) in enumerate(results, 1):
                print(f"{i}. Score: {score:.3f}")
                print(f"   Type: {detection['object_type']}")
                print(f"   Time: {detection['time']}")
                print(f"   Confidence: {detection['confidence']:.2f}")
                if 'caption' in detection and detection['caption']:
                    print(f"   Caption: {detection['caption']}")
                print()
        else:
            print(f"No results found for: '{query}'")
        
        return results
    
    def stop(self):
        """Stop the smart video system"""
        print("\nStopping Smart Video System...")
        self.running = False
        
        # Save AI detections
        if self.detection_service and self.detection_service.detections:
            print("Saving AI detections...")
            saved = self.detection_service.save_detections()
            print(f"Saved {saved} detections to database")
        
        # Stop recording service
        if self.recording_service:
            self.recording_service.stop()
        
        # Print final statistics
        self._print_final_stats()
        
        print("Smart Video System stopped")
    
    def _print_final_stats(self):
        """Print final system statistics"""
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"Total runtime: {str(runtime).split('.')[0]}")
        
        if self.recording_service:
            stats = self.recording_service.get_statistics()
            print(f"Frames recorded: {stats['frame_count']}")
            print(f"Frames dropped: {stats['dropped_frames']} ({stats['drop_rate']*100:.1f}%)")
        
        if self.detection_service:
            stats = self.detection_service.get_statistics()
            print(f"Objects detected: {stats['total_detections']}")
            for obj_type, count in stats['detections_by_type'].items():
                print(f"  {obj_type}: {count}")
        
        print("="*50)
    
    def is_running(self) -> bool:
        """Check if system is running"""
        return self.running and (not self.recording_service or self.recording_service.is_running())


def main():
    parser = argparse.ArgumentParser(description='Smart Video System - Modular AI Video Processing')
    
    # Input source options
    parser.add_argument('-w', '--webcam', metavar='INDEX', default='0',
                        help='Webcam index (default: 0)')
    parser.add_argument('-r', '--rtsp', metavar='URL',
                        help='RTSP stream URL (e.g., rtsp://192.168.6.244:554/11)')
    parser.add_argument('-d', '--device', metavar='PATH',
                        help='Device path (e.g., /dev/video0)')
    
    # Video settings
    parser.add_argument('--resolution', metavar='WxH', default='1280x720',
                        help='Resolution (default: 1280x720)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame rate (default: 30)')
    parser.add_argument('--output-dir', default='videos',
                        help='Output directory (default: videos)')
    
    # AI Detection options
    parser.add_argument('--enable-ai', action='store_true',
                        help='Enable AI object detection')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='AI detection confidence threshold (default: 0.5)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Keep top N detections per object type (default: 10)')
    parser.add_argument('--db-path', default='detections.db',
                        help='Database path (default: detections.db)')
    
    # Display options
    parser.add_argument('--show-frames', action='store_true',
                        help='Display frames in real-time')
    parser.add_argument('--show-detections', action='store_true',
                        help='Show detection bounding boxes')
    
    # Search options
    parser.add_argument('--search', metavar='QUERY',
                        help='Search existing detections with semantic search')
    
    args = parser.parse_args()
    
    # Handle search-only mode
    if args.search:
        if not DETECTION_AVAILABLE:
            print("Detection service not available for search")
            return
        
        # Initialize detection service for search
        ai_config = {
            'model_path': args.model,
            'confidence': args.confidence,
            'db_path': args.db_path,
            'top_n': args.top_n,
            'enable_semantic_search': True
        }
        
        detection_service = DetectionService(**ai_config)
        if detection_service.semantic_agent:
            results = detection_service.search_detections(args.search)
            if not results:
                print("No results found. Make sure detections have been processed and indexed.")
        else:
            print("Semantic search not available")
        return
    
    # Determine input source
    if args.rtsp:
        input_source = args.rtsp
    elif args.device:
        input_source = args.device
    else:
        input_source = args.webcam
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print("Invalid resolution format. Use WxH (e.g., 1280x720)")
        return
    
    # AI configuration
    ai_config = None
    if args.enable_ai:
        ai_config = {
            'model_path': args.model,
            'confidence': args.confidence,
            'db_path': args.db_path,
            'top_n': args.top_n,
            'enable_semantic_search': True
        }
    
    # Create smart video system
    system = SmartVideoSystem(
        input_source=input_source,
        resolution=(width, height),
        fps=args.fps,
        output_dir=args.output_dir,
        enable_ai=args.enable_ai,
        ai_config=ai_config
    )
    
    # Signal handler
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start system
        if not system.start(show_frames=args.show_frames, show_detections=args.show_detections):
            print("Failed to start smart video system")
            return
        
        if args.show_frames:
            # Run display loop
            system.run_display_loop()
        else:
            # Run headless
            print("System running... Press Ctrl+C to stop")
            while system.is_running():
                time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.stop()


if __name__ == "__main__":
    main()