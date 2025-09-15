import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional, List
import time

class AdaptiveMotionDetector:
    """
    Advanced motion detection system with adaptive thresholding,
    background subtraction, and temporal consistency.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 history_frames: int = 5,
                 min_contour_area: int = 500,
                 noise_reduction_kernel: int = 5):
        """
        Initialize the motion detector.
        
        Args:
            learning_rate: Background model update rate (0.001-0.1)
            history_frames: Number of frames for temporal consistency
            min_contour_area: Minimum area for valid motion detection
            noise_reduction_kernel: Kernel size for morphological operations
        """
        self.learning_rate = learning_rate
        self.history_frames = history_frames
        self.min_contour_area = min_contour_area
        self.kernel_size = noise_reduction_kernel
        
        # Background subtractor - MOG2 is robust and adaptive
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        
        # Morphological kernel for noise reduction
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (noise_reduction_kernel, noise_reduction_kernel)
        )
        
        # Frame history for temporal consistency
        self.motion_history = deque(maxlen=history_frames)
        
        # Statistics tracking for adaptive thresholding
        self.motion_stats = deque(maxlen=100)  # Last 100 frames
        
        # Motion regions tracking
        self.previous_contours = []
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal motion detection."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def compute_adaptive_threshold(self, motion_mask: np.ndarray) -> float:
        """
        Compute adaptive threshold based on frame statistics and history.
        """
        # Calculate current frame motion statistics
        motion_pixels = np.sum(motion_mask > 0)
        total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        # Add to statistics history
        self.motion_stats.append(motion_ratio)
        
        if len(self.motion_stats) < 10:
            # Not enough history, use default
            return 0.02  # 2% of frame
        
        # Calculate adaptive threshold based on recent history
        mean_motion = np.mean(self.motion_stats)
        std_motion = np.std(self.motion_stats)
        
        # Threshold is mean + 2*std (catches significant deviations)
        adaptive_threshold = mean_motion + (2.0 * std_motion)
        
        # Clamp between reasonable bounds
        return np.clip(adaptive_threshold, 0.005, 0.15)  # 0.5% to 15%
    
    def apply_temporal_consistency(self, current_motion: bool) -> bool:
        """
        Apply temporal consistency check to reduce false positives.
        """
        self.motion_history.append(current_motion)
        
        if len(self.motion_history) < self.history_frames:
            return False
        
        # Require motion in at least 60% of recent frames
        motion_count = sum(self.motion_history)
        consistency_threshold = int(self.history_frames * 0.6)
        
        return motion_count >= consistency_threshold
    
    def detect_motion_regions(self, motion_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect and filter motion regions using contour analysis.
        """
        # Find contours
        contours, _ = cv2.findContours(
            motion_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_regions = []
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_contour_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (avoid very thin/wide detections)
            aspect_ratio = w / h
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
                
            motion_regions.append((x, y, w, h))
            valid_contours.append(contour)
        
        self.previous_contours = valid_contours
        return motion_regions
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Process a single frame and detect motion.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Tuple of (motion_detected, processed_mask, motion_regions)
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(processed_frame, learningRate=self.learning_rate)
        
        # Remove shadows (they're marked as 127 in MOG2)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to reduce noise
        # Opening: erosion followed by dilation (removes small noise)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Closing: dilation followed by erosion (fills small holes)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Calculate motion percentage
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        # Get adaptive threshold
        threshold = self.compute_adaptive_threshold(fg_mask)
        
        # Determine if significant motion detected
        significant_motion = motion_ratio > threshold
        
        # Apply temporal consistency
        motion_detected = self.apply_temporal_consistency(significant_motion)
        
        # Detect motion regions if motion is present
        motion_regions = []
        if motion_detected:
            motion_regions = self.detect_motion_regions(fg_mask)
        
        return motion_detected, fg_mask, motion_regions
    
    def visualize_results(self, 
                         frame: np.ndarray, 
                         motion_mask: np.ndarray, 
                         motion_regions: List[Tuple[int, int, int, int]],
                         motion_detected: bool) -> np.ndarray:
        """
        Create visualization of motion detection results.
        """
        # Create output frame
        output = frame.copy()
        
        # Overlay motion mask (semi-transparent)
        if len(frame.shape) == 3:
            motion_overlay = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_overlay[:, :, 1] = 0  # Remove green channel
            motion_overlay[:, :, 2] = 0  # Remove red channel
            output = cv2.addWeighted(output, 0.7, motion_overlay, 0.3, 0)
        
        # Draw bounding boxes around motion regions
        for (x, y, w, h) in motion_regions:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, 'MOTION', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add status text
        status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(output, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add statistics
        if len(self.motion_stats) > 0:
            current_threshold = self.compute_adaptive_threshold(motion_mask)
            threshold_text = f"Threshold: {current_threshold:.3f}"
            cv2.putText(output, threshold_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output

def main():
    """
    Example usage of the AdaptiveMotionDetector.
    """
    # Initialize motion detector
    detector = AdaptiveMotionDetector(
        learning_rate=0.005,      # Slow adaptation for stable backgrounds
        history_frames=3,         # Require motion in 3 frames
        min_contour_area=800,     # Minimum 800 pixels for valid motion
        noise_reduction_kernel=7  # 7x7 kernel for noise reduction
    )
    
    # Initialize video capture (0 for webcam, or path to video file)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Motion Detection System Started")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frame for motion detection
        motion_detected, motion_mask, motion_regions = detector.process_frame(frame)
        
        # Create visualization
        output_frame = detector.visualize_results(
            frame, motion_mask, motion_regions, motion_detected
        )
        
        # Display results
        cv2.imshow('Motion Detection', output_frame)
        cv2.imshow('Motion Mask', motion_mask)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            cv2.imwrite(f'motion_detection_{timestamp}.jpg', output_frame)
            print(f"Frame saved as motion_detection_{timestamp}.jpg")
        
        # Print detection status every 30 frames
        if frame_count % 30 == 0 and motion_detected:
            print(f"Frame {frame_count}: Motion detected - {len(motion_regions)} regions")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Motion detection system stopped")

if __name__ == "__main__":
    main()