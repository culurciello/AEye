#!/usr/bin/env python3
"""
Video Recording Service - Clean H.265 recording with frame streaming
Separated from AI processing for better modularity
"""

import subprocess
import numpy as np
from threading import Thread, Event
import queue
import time
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Dict, Callable, List


class VideoRecordingService:
    """
    Clean video recording service using FFmpeg H.264 encoding
    Provides frame streaming for external processing (like AI detection)
    """
    
    def __init__(self, 
                 input_source: str,
                 resolution: tuple = (1280, 720),
                 fps: int = 30,
                 output_dir: str = "videos"):
        
        self.input_source = input_source
        self.width, self.height = resolution
        self.fps = fps
        self.output_dir = output_dir
        
        # Frame streaming
        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_callbacks: List[Callable] = []
        self.stop_event = Event()
        
        # FFmpeg processes
        self.read_proc = None
        self.record_proc = None
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamped output pattern
        self.output_pattern = self._get_output_pattern()
    
    def _get_output_pattern(self) -> str:
        """Generate output pattern with timestamp directory structure"""
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        hour_dir = now.strftime("%H")
        
        # Create directory structure: videos/YYYY-MM-DD/HH/
        full_dir = os.path.join(self.output_dir, date_dir, hour_dir)
        os.makedirs(full_dir, exist_ok=True)
        
        # Pattern: videos/YYYY-MM-DD/HH/MM.mp4 (minute-based segments)
        return os.path.join(full_dir, "%M.mp4")
    
    def _get_input_args(self) -> List[str]:
        """Get FFmpeg input arguments based on source type"""
        if self.input_source.startswith('rtsp://'):
            # RTSP stream
            return [
                '-rtsp_transport', 'tcp',
                '-i', self.input_source
            ]
        elif self.input_source.isdigit():
            # Webcam index (macOS/Linux)
            if sys.platform == 'darwin':  # macOS
                return [
                    '-f', 'avfoundation',
                    '-framerate', str(self.fps),
                    '-video_size', f'{self.width}x{self.height}',
                    '-i', f'{self.input_source}:0'  # video:audio
                ]
            else:  # Linux
                return [
                    '-f', 'v4l2',
                    '-framerate', str(self.fps),
                    '-video_size', f'{self.width}x{self.height}',
                    '-i', f'/dev/video{self.input_source}'
                ]
        else:
            # Device path or file
            return ['-i', self.input_source]
    
    def add_frame_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """Add callback to be called for each frame"""
        self.frame_callbacks.append(callback)
    
    def start_recording(self) -> bool:
        """Start H.264 recording process"""
        input_args = self._get_input_args()
        
        self.record_cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',  # Hardware acceleration
            *input_args,
            '-c:v', 'libx264',   # H.264 codec for compatibility
            '-preset', 'ultrafast',  # Fast encoding
            '-crf', '23',        # Quality (lower = better)
            '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
            '-f', 'segment',     # Segmented output
            '-segment_time', '60',  # 1-minute segments
            '-segment_format', 'mp4',
            '-reset_timestamps', '1',
            '-strftime', '1',    # Enable timestamp formatting
            '-y',               # Overwrite output files
            self.output_pattern
        ]
        
        try:
            self.record_proc = subprocess.Popen(
                self.record_cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL
            )
            print(f"Recording started: {self.output_pattern}")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False
    
    def start_frame_streaming(self) -> bool:
        """Start frame streaming for external processing"""
        input_args = self._get_input_args()
        
        self.read_cmd = [
            'ffmpeg',
            '-hwaccel', 'auto',
            *input_args,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-an',  # No audio
            'pipe:'
        ]
        
        try:
            self.read_proc = subprocess.Popen(
                self.read_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            # Start frame reading thread
            Thread(target=self._frame_reader_thread, daemon=True).start()
            print("Frame streaming started")
            return True
        except Exception as e:
            print(f"Failed to start frame streaming: {e}")
            return False
    
    def start(self, enable_streaming: bool = True) -> bool:
        """Start recording and optionally frame streaming"""
        print(f"Starting video recording service")
        print(f"Input: {self.input_source}")
        print(f"Output: {self.output_pattern}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
        
        # Start recording
        if not self.start_recording():
            return False
        
        # Start frame streaming if requested
        if enable_streaming:
            if not self.start_frame_streaming():
                print("Warning: Frame streaming failed, recording will continue")
        
        return True
    
    def _frame_reader_thread(self):
        """Thread to read frames from FFmpeg stdout"""
        frame_size = self.width * self.height * 3  # BGR24
        
        while not self.stop_event.is_set():
            try:
                raw_frame = self.read_proc.stdout.read(frame_size)
                if not raw_frame or len(raw_frame) != frame_size:
                    print("Frame reading ended or incomplete frame")
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
                
                # Frame metadata
                frame_info = {
                    'frame_number': self.frame_count,
                    'timestamp': datetime.now(),
                    'source': self.input_source,
                    'resolution': (self.width, self.height)
                }
                
                # Call frame callbacks (for AI processing, etc.)
                for callback in self.frame_callbacks:
                    try:
                        callback(frame, frame_info)
                    except Exception as e:
                        print(f"Error in frame callback: {e}")
                
                # Put frame in queue for get_frame() calls
                try:
                    self.frame_queue.put((frame, frame_info), timeout=0.01)
                except queue.Full:
                    # Drop frame if queue is full
                    self.dropped_frames += 1
                
                self.frame_count += 1
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Error reading frame: {e}")
                break
    
    def get_frame(self, timeout: float = 1.0) -> Optional[tuple]:
        """
        Get the latest frame for processing
        
        Returns:
            Tuple of (frame_array, frame_info) or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop all recording and streaming processes"""
        print("Stopping video recording service...")
        self.stop_event.set()
        
        # Stop reading process
        if self.read_proc and self.read_proc.poll() is None:
            self.read_proc.terminate()
            try:
                self.read_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.read_proc.kill()
        
        # Stop recording process
        if self.record_proc and self.record_proc.poll() is None:
            self.record_proc.terminate()
            try:
                self.record_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.record_proc.kill()
        
        # Print statistics
        print(f"Recording statistics:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Dropped frames: {self.dropped_frames}")
        if self.frame_count > 0:
            print(f"  Drop rate: {self.dropped_frames/self.frame_count*100:.1f}%")
        
        print("Video recording service stopped")
    
    def is_running(self) -> bool:
        """Check if recording or streaming is still running"""
        record_running = self.record_proc and self.record_proc.poll() is None
        read_running = self.read_proc and self.read_proc.poll() is None
        return record_running or read_running
    
    def get_statistics(self) -> Dict:
        """Get recording statistics"""
        return {
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(1, self.frame_count),
            'is_running': self.is_running(),
            'input_source': self.input_source,
            'output_pattern': self.output_pattern,
            'resolution': (self.width, self.height),
            'fps': self.fps
        }


def main():
    """Test the recording service standalone"""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description='Video Recording Service Test')
    parser.add_argument('-w', '--webcam', default='0', help='Webcam index')
    parser.add_argument('-r', '--rtsp', help='RTSP stream URL')
    parser.add_argument('-d', '--device', help='Device path')
    parser.add_argument('--resolution', default='1280x720', help='Resolution')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--output-dir', default='videos', help='Output directory')
    parser.add_argument('--no-streaming', action='store_true', help='Disable frame streaming')
    parser.add_argument('--show-frames', action='store_true', help='Display frames')
    
    args = parser.parse_args()
    
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
    
    # Create service
    service = VideoRecordingService(
        input_source=input_source,
        resolution=(width, height),
        fps=args.fps,
        output_dir=args.output_dir
    )
    
    # Frame callback for display
    if args.show_frames:
        import cv2
        
        def display_frame(frame, frame_info):
            cv2.imshow('Recording Preview', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                service.stop()
        
        service.add_frame_callback(display_frame)
    
    # Signal handler
    def signal_handler(sig, frame_arg):
        print(f"\nReceived signal {sig}, stopping...")
        service.stop()
        if args.show_frames:
            cv2.destroyAllWindows()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start service
        if not service.start(enable_streaming=not args.no_streaming):
            print("Failed to start recording service")
            return
        
        print("Recording service started. Press Ctrl+C to stop")
        
        # Main loop
        while service.is_running():
            time.sleep(1)
            
            # Print status every 30 seconds
            if service.frame_count > 0 and service.frame_count % (30 * args.fps) == 0:
                stats = service.get_statistics()
                print(f"Status - Frames: {stats['frame_count']}, "
                      f"Dropped: {stats['dropped_frames']} ({stats['drop_rate']*100:.1f}%)")
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()