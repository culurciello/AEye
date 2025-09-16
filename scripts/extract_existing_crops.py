#!/usr/bin/env python3

import sqlite3
import cv2
import os
import logging
import argparse
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CropExtractor:
    """Extract crops for existing object detections from video files."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def get_detections_without_crops(self):
        """Get all object detections that don't have crops yet."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT od.id, od.motion_event_id, od.frame_timestamp, od.class_name,
                   od.confidence, od.bbox_x, od.bbox_y, od.bbox_width, od.bbox_height,
                   me.video_file
            FROM object_detections od
            JOIN motion_events me ON od.motion_event_id = me.id
            WHERE od.object_crop IS NULL
            ORDER BY od.motion_event_id, od.frame_timestamp
        ''')

        detections = []
        for row in cursor.fetchall():
            detection = {
                'id': row[0],
                'motion_event_id': row[1],
                'frame_timestamp': row[2],
                'class_name': row[3],
                'confidence': row[4],
                'bbox': [row[5], row[6], row[7], row[8]],  # x, y, w, h
                'video_file': row[9]
            }
            detections.append(detection)

        conn.close()
        logger.info(f"Found {len(detections)} detections without crops")
        return detections

    def extract_crop_from_video(self, video_path: str, frame_timestamp: str, bbox: list):
        """Extract crop from video at specific timestamp."""
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            # Parse timestamp
            if isinstance(frame_timestamp, str):
                frame_time = datetime.fromisoformat(frame_timestamp)
            else:
                frame_time = frame_timestamp

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 30  # Default FPS

            # Get video start time from filename or use creation time
            video_basename = os.path.basename(video_path)
            try:
                # Try to extract timestamp from filename (format: YYYYMMDD_HHMMSS.mp4)
                if video_basename.endswith('.mp4') and len(video_basename) >= 17:
                    filename_parts = video_basename[:-4]  # Remove .mp4
                    if '_' in filename_parts and len(filename_parts) == 15:  # YYYYMMDD_HHMMSS
                        video_start_time = datetime.strptime(filename_parts, '%Y%m%d_%H%M%S')
                    else:
                        # Use file modification time as fallback
                        video_start_time = datetime.fromtimestamp(os.path.getmtime(video_path))
                else:
                    # Use file modification time as fallback
                    video_start_time = datetime.fromtimestamp(os.path.getmtime(video_path))
            except:
                logger.warning(f"Could not determine video start time for {video_path}")
                cap.release()
                return None

            # Calculate frame number
            time_offset = (frame_time - video_start_time).total_seconds()
            if time_offset < 0:
                logger.warning(f"Frame timestamp is before video start time")
                cap.release()
                return None

            target_frame = int(time_offset * fps)
            if target_frame >= total_frames:
                logger.warning(f"Target frame {target_frame} exceeds video length {total_frames}")
                cap.release()
                return None

            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                logger.warning(f"Could not read frame at position {target_frame}")
                return None

            # Extract crop
            x, y, w, h = bbox
            frame_h, frame_w = frame.shape[:2]

            # Ensure coordinates are within frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            x2 = min(x + w, frame_w)
            y2 = min(y + h, frame_h)

            if x >= x2 or y >= y2:
                logger.warning(f"Invalid crop coordinates: ({x}, {y}, {x2}, {y2})")
                return None

            # Extract crop
            crop = frame[y:y2, x:x2]

            if crop.size == 0:
                logger.warning(f"Empty crop extracted")
                return None

            # Resize crop to standardized size (max 200x200, maintain aspect ratio)
            h_crop, w_crop = crop.shape[:2]
            if h_crop > 200 or w_crop > 200:
                scale = min(200/w_crop, 200/h_crop)
                new_w = int(w_crop * scale)
                new_h = int(h_crop * scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode as JPEG bytes
            success, crop_encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                return crop_encoded.tobytes()
            else:
                logger.warning(f"Failed to encode crop as JPEG")
                return None

        except Exception as e:
            logger.error(f"Error extracting crop from {video_path}: {e}")
            return None

    def update_detection_crop(self, detection_id: int, crop_bytes: bytes):
        """Update detection with extracted crop."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE object_detections
            SET object_crop = ?
            WHERE id = ?
        ''', (crop_bytes, detection_id))

        conn.commit()
        conn.close()

    def extract_all_crops(self, max_detections: int = None):
        """Extract crops for all detections without crops."""
        detections = self.get_detections_without_crops()

        if max_detections:
            detections = detections[:max_detections]

        if not detections:
            logger.info("No detections without crops found")
            return

        logger.info(f"Processing {len(detections)} detections...")

        processed = 0
        successful = 0
        current_video_path = None
        current_video_start_time = None

        for i, detection in enumerate(detections):
            try:
                video_path = detection['video_file']

                # Skip if video file doesn't exist
                if not os.path.exists(video_path):
                    logger.warning(f"Video file not found: {video_path}")
                    continue

                # Extract crop
                crop_bytes = self.extract_crop_from_video(
                    video_path,
                    detection['frame_timestamp'],
                    detection['bbox']
                )

                if crop_bytes:
                    # Update database
                    self.update_detection_crop(detection['id'], crop_bytes)
                    successful += 1
                    logger.debug(f"Extracted crop for detection {detection['id']} ({detection['class_name']})")

                processed += 1

                # Progress update
                if processed % 100 == 0:
                    logger.info(f"Progress: {processed}/{len(detections)} processed, {successful} successful")

            except Exception as e:
                logger.error(f"Error processing detection {detection['id']}: {e}")
                processed += 1

        logger.info(f"Crop extraction complete: {successful}/{processed} successful")


def main():
    parser = argparse.ArgumentParser(description='Extract crops for existing object detections')
    parser.add_argument('--db-path', default='data/db/detections.db',
                       help='Path to the SQLite database file')
    parser.add_argument('--max-detections', type=int,
                       help='Maximum number of detections to process (for testing)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set the logging level')

    args = parser.parse_args()

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found: {args.db_path}")
        return 1

    # Run crop extraction
    try:
        extractor = CropExtractor(args.db_path)
        extractor.extract_all_crops(args.max_detections)

        logger.info("Crop extraction completed! Run the analysis script to see updated visualizations:")
        logger.info("python3 scripts/analysis.py")
        return 0

    except Exception as e:
        logger.error(f"Crop extraction failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())