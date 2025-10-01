#!/usr/bin/env python3
"""
Demo Object Detector using YOLO
Detects and displays all YOLO classes in real-time from camera or IP camera feeds.
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Real-time object detector using YOLO')
    parser.add_argument('--source', type=str, default='0',
                        help='Camera source: 0 for webcam, or RTSP/HTTP URL for IP camera')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='Path to YOLO model file (default: models/yolov8n.pt)')
    args = parser.parse_args()

    # Load YOLO model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    print("Model loaded successfully!")

    # Open camera
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open camera source: {args.source}")
        return

    print(f"Camera opened successfully. Press 'q' to quit.")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        detections = {}

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    # Process all classes that meet confidence threshold
                    if confidence >= args.confidence:
                        # Count detections per class
                        detections[class_name] = detections.get(class_name, 0) + 1

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Draw bounding box with random color per class
                        color = tuple(int(c) for c in np.random.randint(0, 255, 3, dtype=int))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label with confidence
                        label = f"{class_name} {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw detection counts on frame
        if detections:
            # Calculate info panel height
            panel_height = 20 + (len(detections) * 25) + 20
            cv2.rectangle(frame, (10, 10), (350, panel_height), (0, 0, 0), -1)

            info_y = 35
            cv2.putText(frame, "Detections:", (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            info_y += 30
            for class_name, count in sorted(detections.items()):
                count_text = f"{class_name}: {count}"
                cv2.putText(frame, count_text, (30, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                info_y += 25

        # Display frame
        cv2.imshow('YOLO Object Detector - Press Q to quit', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == '__main__':
    main()
