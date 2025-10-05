#!/usr/bin/env python3
"""
Demo Person Counter using YOLO
Detects and counts people in real-time from camera or IP camera feeds.
Optionally includes face detection and recognition.
"""

import cv2
import numpy as np
import argparse
import sys
from ultralytics import YOLO

# Add lib to path for imports
sys.path.insert(0, 'lib')
from face_detector import FaceDetector


def main():
    parser = argparse.ArgumentParser(description='Real-time person counter using YOLO')
    parser.add_argument('--source', type=str, default='0',
                        help='Camera source: 0 for webcam, or RTSP/HTTP URL for IP camera')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='Path to YOLO model file (default: models/yolov8n.pt)')
    parser.add_argument('--face-recognition', action='store_true',
                        help='Enable face detection and recognition')
    parser.add_argument('--known-faces-dir', type=str, default='data/faces-known',
                        help='Directory containing known faces for recognition (default: data/faces-known)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU for face detection')
    args = parser.parse_args()

    # Load YOLO model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    print("Model loaded successfully!")

    # Initialize face detector if enabled
    face_detector = None
    if args.face_recognition:
        print("Initializing face detector...")
        face_detector = FaceDetector(use_gpu=not args.no_gpu, db_manager=None,
                                     known_faces_dir=args.known_faces_dir)
        face_detector.init_face_detector()
        if face_detector.face_app is None:
            print("Warning: Face detection initialization failed. Continuing without face recognition.")
            face_detector = None
        else:
            print(f"Face detector loaded. Known faces: {list(face_detector.known_face_embeddings.keys())}")

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

        person_count = 0
        person_bboxes = []

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    # Only process 'person' class (class_id 0 in COCO dataset)
                    if class_name == 'person' and confidence >= args.confidence:
                        person_count += 1

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Store person bbox for face detection
                        if face_detector:
                            person_bboxes.append({
                                'bbox': (x1, y1, x2 - x1, y2 - y1),
                                'confidence': confidence
                            })

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label with confidence
                        label = f"Person {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Run face detection if enabled
        recognized_faces = []
        if face_detector and person_bboxes:
            try:
                for person_data in person_bboxes:
                    bbox_x, bbox_y, bbox_w, bbox_h = person_data['bbox']

                    # Add padding to person crop for better face detection
                    padding = 20
                    crop_x1 = max(0, bbox_x - padding)
                    crop_y1 = max(0, bbox_y - padding)
                    crop_x2 = min(frame.shape[1], bbox_x + bbox_w + padding)
                    crop_y2 = min(frame.shape[0], bbox_y + bbox_h + padding)

                    # Extract person crop
                    person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Skip if crop is too small
                    if person_crop.shape[0] < 60 or person_crop.shape[1] < 60:
                        continue

                    # Run face detection on person crop
                    faces = face_detector.face_app.get(person_crop)

                    for face in faces:
                        bbox = face.bbox.astype(int)
                        face_confidence = float(face.det_score)

                        # Skip low confidence detections
                        if face_confidence < 0.7:
                            continue

                        # Adjust face coordinates to full frame
                        face_x1, face_y1, face_x2, face_y2 = bbox
                        full_x1 = face_x1 + crop_x1
                        full_y1 = face_y1 + crop_y1
                        full_x2 = face_x2 + crop_x1
                        full_y2 = face_y2 + crop_y1

                        # Skip very small faces
                        if (full_x2 - full_x1) < 30 or (full_y2 - full_y1) < 30:
                            continue

                        # Get normalized embedding
                        embedding = face.embedding
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                        # Try to recognize the face
                        known_person, recognition_confidence = face_detector.recognize_face(embedding)

                        # Draw face bounding box
                        color = (0, 255, 255) if known_person else (255, 0, 255)
                        cv2.rectangle(frame, (full_x1, full_y1), (full_x2, full_y2), color, 2)

                        # Draw face label
                        if known_person:
                            face_label = f"{known_person} ({recognition_confidence:.2f})"
                            recognized_faces.append(known_person)
                        else:
                            face_label = f"Unknown ({face_confidence:.2f})"

                        face_label_size, _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (full_x1, full_y2),
                                    (full_x1 + face_label_size[0] + 10, full_y2 + face_label_size[1] + 10),
                                    color, -1)
                        cv2.putText(frame, face_label, (full_x1 + 5, full_y2 + face_label_size[1] + 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            except Exception as e:
                print(f"Error in face detection: {e}")

        # Draw person count and recognized faces on frame
        info_y = 30
        count_text = f"People: {person_count}"
        cv2.rectangle(frame, (10, 10), (400, 100 if recognized_faces else 60), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display recognized faces
        if recognized_faces:
            info_y += 30
            faces_text = f"Recognized: {', '.join(set(recognized_faces))}"
            cv2.putText(frame, faces_text, (20, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display frame
        cv2.imshow('Person Counter - Press Q to quit', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


if __name__ == '__main__':
    main()
