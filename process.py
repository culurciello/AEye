#!/usr/bin/env python3

import cv2
import os
import argparse
import logging
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path

from lib.database import DatabaseManager
from lib.object_detector import ObjectDetector
from lib.face_detector import FaceDetector
from lib.video_processor import VideoSegment

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files and saves detections to the database.
    """

    def __init__(self,
                 videos_dir: str = "data/videos",
                 images_dir: str = "data/images",
                 db_path: str = "data/db/detections.db",
                 use_gpu: bool = True,
                 skip_processed: bool = True,
                 file_age_threshold: int = 60,
                 delete_empty: bool = True,
                 image_save_interval: int = 600):
        """
        Initialize the video processor.

        Args:
            videos_dir: Directory containing video files
            db_path: Database path for storing detections
            use_gpu: Whether to use GPU for detection
            skip_processed: Skip videos already marked as processed in database
            file_age_threshold: Minimum age (in seconds) for a file to be considered complete (default: 60)
            delete_empty: Delete videos with no detections (default: True)
            image_save_interval: Save one image every N seconds (default: 600 = 10 minutes)
        """
        self.videos_dir = videos_dir
        self.db_path = db_path
        self.use_gpu = use_gpu
        self.skip_processed = skip_processed
        self.file_age_threshold = file_age_threshold
        self.delete_empty = delete_empty
        self.image_save_interval = image_save_interval
        self.images_dir = images_dir

        # Initialize database manager
        self.db_manager = DatabaseManager(self.db_path)

        # Initialize object detection
        logger.info("Initializing object detector...")
        self.object_detector = ObjectDetector(self.db_manager)
        self.object_detector.init_yolo_detector()

        # Initialize face detection
        logger.info("Initializing face detector...")
        self.face_detector = FaceDetector(self.use_gpu, self.db_manager)
        self.face_detector.init_face_detector()

    def get_all_video_files(self, exclude_recent: bool = False):
        """Get all video files from the videos directory.

        Args:
            exclude_recent: If True, exclude the most recently modified file (likely being written)

        Returns:
            List of video file paths
        """
        video_files = []

        # Search for mp4 files in all subdirectories
        pattern = os.path.join(self.videos_dir, "**", "*.mp4")
        all_files = sorted(glob.glob(pattern, recursive=True))

        current_time = time.time()

        # Filter files based on age threshold and exclude_recent flag
        for video_file in all_files:
            try:
                # Get file modification time
                file_mtime = os.path.getmtime(video_file)
                file_age = current_time - file_mtime

                # Skip files that are too recent (might still be written to)
                if file_age < self.file_age_threshold:
                    logger.debug(f"Skipping recent file (age: {file_age:.1f}s): {os.path.basename(video_file)}")
                    continue

                video_files.append(video_file)

            except OSError as e:
                logger.warning(f"Could not check file {video_file}: {e}")
                continue

        # If exclude_recent is True and we have files, remove the most recent one
        if exclude_recent and video_files:
            # Sort by modification time
            video_files_with_mtime = [(f, os.path.getmtime(f)) for f in video_files]
            video_files_with_mtime.sort(key=lambda x: x[1])

            # Remove the most recent file
            most_recent_file = video_files_with_mtime[-1][0]
            video_files.remove(most_recent_file)
            logger.debug(f"Excluding most recent file: {os.path.basename(most_recent_file)}")

        logger.debug(f"Found {len(video_files)} video files ready for processing")
        return video_files

    def get_unprocessed_video_files(self, exclude_recent: bool = True):
        """Get only unprocessed video files.

        Args:
            exclude_recent: If True, exclude the most recently modified file

        Returns:
            List of unprocessed video file paths
        """
        all_videos = self.get_all_video_files(exclude_recent=exclude_recent)
        unprocessed = []

        for video_path in all_videos:
            if not self.is_video_processed(video_path):
                unprocessed.append(video_path)

        logger.info(f"Found {len(unprocessed)} unprocessed video files")
        return unprocessed

    def is_video_processed(self, video_path: str) -> bool:
        """Check if a video has already been processed."""
        motion_event_data = self.db_manager.get_motion_event_by_video_file(video_path)

        if motion_event_data:
            motion_event_id, event_data = motion_event_data
            return event_data.get('processed', False)

        return False

    def save_periodic_image(self, frame, frame_time: datetime, video_filename: str):
        """Save a frame as an image in the date-organized images directory.

        Args:
            frame: The video frame to save
            frame_time: The timestamp of the frame
            video_filename: The name of the video file being processed
        """
        # Create date-based directory structure (YYYY_MM_DD)
        date_str = frame_time.strftime("%Y_%m_%d")
        daily_images_dir = os.path.join(self.images_dir, date_str)
        os.makedirs(daily_images_dir, exist_ok=True)

        # Create image filename with timestamp
        time_str = frame_time.strftime("%H%M%S")
        image_filename = f"{time_str}_{os.path.splitext(video_filename)[0]}.jpg"
        image_path = os.path.join(daily_images_dir, image_filename)

        # Save the frame as JPEG
        cv2.imwrite(image_path, frame)
        logger.debug(f"Saved periodic image: {image_path}")

    def detect_objects_in_frame(self, frame, frame_time, motion_event_id):
        """Detect objects in a single frame using YOLO and store results."""
        return self.object_detector.detect_objects_in_frame(frame, frame_time, motion_event_id)

    def detect_faces_in_person_crops(self, frame, person_bboxes, frame_time, motion_event_id):
        """Detect faces within person bounding boxes."""
        if self.face_detector:
            return self.face_detector.detect_faces_in_person_crops(frame, person_bboxes, frame_time, motion_event_id)
        return 0

    def process_video_file(self, video_path: str):
        """Process a single video file."""
        logger.info(f"Processing: {video_path}")

        # Check if already processed
        if self.skip_processed and self.is_video_processed(video_path):
            logger.info(f"Skipping already processed video: {video_path}")
            return

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count_total / fps

        # Extract timestamp from filename
        filename = os.path.basename(video_path)
        try:
            # Expected format: YYYYMMDD_HHMMSS.mp4
            timestamp_str = filename.replace('.mp4', '')
            start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            logger.warning(f"Could not parse timestamp from filename: {filename}, using current time")
            start_time = datetime.now()

        end_time = start_time + timedelta(seconds=duration)

        # Create video segment
        segment = VideoSegment(
            start_time=start_time,
            end_time=end_time,
            file_path=video_path,
            motion_detected=True,
            processed=False
        )

        # Store motion event in database or get existing one
        motion_event_data = self.db_manager.get_motion_event_by_video_file(video_path)

        if motion_event_data:
            motion_event_id, _ = motion_event_data
            logger.info(f"Using existing motion event ID: {motion_event_id}")
        else:
            motion_event_id = self.db_manager.store_motion_event(segment)
            logger.info(f"Created new motion event ID: {motion_event_id}")

        # Process video frames
        frame_count = 0
        total_faces = 0
        total_objects = 0
        last_image_save_time = 0  # Track elapsed time for periodic image saving

        # Start tracking session for this video
        if self.object_detector.yolo_model:
            self.object_detector.start_tracking_session(motion_event_id)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                elapsed_time = frame_count / fps  # Time in seconds since video start

                # Save periodic images every image_save_interval seconds
                if self.image_save_interval > 0 and elapsed_time - last_image_save_time >= self.image_save_interval:
                    frame_time = start_time + timedelta(seconds=elapsed_time)
                    self.save_periodic_image(frame, frame_time, filename)
                    last_image_save_time = elapsed_time

                # Process every 15th frame to reduce computational load
                # You can adjust this based on your needs
                if frame_count % 15 == 0:
                    frame_time = start_time + timedelta(seconds=frame_count / fps)

                    # Object detection - returns person bboxes too
                    objects_detected, person_bboxes = self.detect_objects_in_frame(frame, frame_time, motion_event_id)
                    total_objects += objects_detected

                    # Person-triggered face detection - only run if persons detected
                    if person_bboxes and self.face_detector:
                        faces_detected = self.detect_faces_in_person_crops(frame, person_bboxes, frame_time, motion_event_id)
                        total_faces += faces_detected
                        if faces_detected > 0:
                            logger.debug(f"Detected {faces_detected} faces in {len(person_bboxes)} person crops")

                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / frame_count_total) * 100
                    logger.debug(f"Progress: {progress:.1f}% ({frame_count}/{frame_count_total} frames)")

        finally:
            cap.release()

            # End tracking session
            if self.object_detector.yolo_model:
                self.object_detector.end_tracking_session(motion_event_id)

        # Update motion event with counts
        self.db_manager.update_motion_event_counts(motion_event_id, total_faces, total_objects)

        logger.info(f"Completed: {video_path}")
        logger.info(f"  - Frames processed: {frame_count}")
        logger.info(f"  - Objects detected: {total_objects}")
        logger.info(f"  - Faces detected: {total_faces}")

        # Only keep videos that have detections (motion is assumed if video exists)
        # Delete video if NO detections found (no faces AND no objects)
        has_detections = (total_faces > 0 or total_objects > 0)

        if self.delete_empty and not has_detections:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"  - Deleted video (no detections): {os.path.basename(video_path)}")

                    # Also remove from database since file is deleted
                    with self.db_manager._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM motion_events WHERE id = ?', (motion_event_id,))
                        conn.commit()

            except Exception as e:
                logger.warning(f"  - Could not delete video: {e}")
        else:
            # Video kept because it has detections
            if has_detections:
                logger.info(f"  - Video saved (has detections)")

    def process_all_videos(self, exclude_recent: bool = True):
        """Process all video files in the videos directory.

        Args:
            exclude_recent: If True, exclude the most recently modified file
        """
        video_files = self.get_all_video_files(exclude_recent=exclude_recent)

        if not video_files:
            logger.warning("No video files found to process")
            return 0, 0, 0

        total_videos = len(video_files)
        processed_count = 0
        skipped_count = 0
        failed_count = 0

        logger.info(f"Starting to process {total_videos} videos...")

        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n[{i}/{total_videos}] Processing video: {os.path.basename(video_path)}")

            try:
                # Check if already processed before attempting to process
                if self.skip_processed and self.is_video_processed(video_path):
                    logger.info(f"Skipping already processed video")
                    skipped_count += 1
                    continue

                self.process_video_file(video_path)
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                failed_count += 1

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos found: {total_videos}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Skipped (already processed): {skipped_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info("="*60)

        return processed_count, skipped_count, failed_count

    def print_database_stats(self):
        """Print database statistics."""
        motion_stats = self.db_manager.get_motion_event_stats()
        object_stats = self.db_manager.get_object_detection_stats()
        face_stats = self.db_manager.get_face_detection_stats()

        logger.info("\nDATABASE STATISTICS")
        logger.info("="*60)
        logger.info(f"Motion events: {motion_stats['total_motion_events']}")
        logger.info(f"  - Processed: {motion_stats['processed_events']}")
        logger.info(f"  - Total duration: {motion_stats['total_duration_seconds']:.1f}s")
        logger.info(f"Object detections: {object_stats['total_object_detections']}")
        logger.info(f"Face detections: {face_stats['total_face_detections']}")
        logger.info(f"  - Recognized: {face_stats['recognized_faces']}")
        logger.info(f"  - Unknown: {face_stats['unknown_faces']}")
        logger.info("="*60)

    def monitor_and_process(self, check_interval: int = 30):
        """Continuously monitor for new video files and process them.

        Args:
            check_interval: Time in seconds between checks for new files (default: 30)
        """
        logger.info("="*60)
        logger.info("STARTING CONTINUOUS MONITORING MODE")
        logger.info("="*60)
        logger.info(f"Videos directory: {self.videos_dir}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"File age threshold: {self.file_age_threshold} seconds")
        logger.info(f"Excluding most recent file: Yes (to avoid processing incomplete files)")
        logger.info("Press Ctrl+C to stop monitoring")
        logger.info("="*60 + "\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                logger.info(f"\n[Monitor Check #{iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Get unprocessed videos (excluding the most recent one)
                unprocessed_videos = self.get_unprocessed_video_files(exclude_recent=True)

                if unprocessed_videos:
                    logger.info(f"Found {len(unprocessed_videos)} unprocessed video(s)")

                    for i, video_path in enumerate(unprocessed_videos, 1):
                        logger.info(f"\n[{i}/{len(unprocessed_videos)}] Processing: {os.path.basename(video_path)}")

                        try:
                            self.process_video_file(video_path)
                            logger.info(f"✓ Successfully processed: {os.path.basename(video_path)}")

                        except Exception as e:
                            logger.error(f"✗ Failed to process {video_path}: {e}")

                else:
                    logger.info("No new unprocessed videos found")

                # Print database stats periodically (every 10 iterations)
                if iteration % 10 == 0:
                    self.print_database_stats()

                # Wait before next check
                logger.info(f"\nWaiting {check_interval} seconds before next check...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("\n\nMonitoring stopped by user")
            self.print_database_stats()


def main():
    parser = argparse.ArgumentParser(description='Process video files and store detections in database')
    parser.add_argument('--videos-dir', default='data/videos',
                       help='Directory containing video files (default: data/videos)')
    parser.add_argument('--images-dir', default='data/images',
                       help='Directory containing image files (default: data/images)')
    parser.add_argument('--db-path', default='data/db/detections.db',
                       help='Database path (default: data/db/detections.db)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage for detection')
    parser.add_argument('--reprocess', action='store_true',
                       help='Reprocess videos even if already marked as processed')
    parser.add_argument('--video-file',
                       help='Process a single video file instead of all videos')
    parser.add_argument('--monitor', action='store_true',
                       help='Continuously monitor for new video files and process them')
    parser.add_argument('--check-interval', type=int, default=30,
                       help='Time in seconds between checks for new files in monitor mode (default: 30)')
    parser.add_argument('--file-age-threshold', type=int, default=60,
                       help='Minimum age in seconds for a file to be considered complete (default: 60)')
    parser.add_argument('--keep-empty', action='store_true',
                       help='Keep videos with no detections (default: delete them)')
    parser.add_argument('--image-save-interval', type=int, default=300,
                       help='Save one image every N seconds (default: 300 = 5 minutes, 0 to disable)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    os.makedirs('data', exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/process.log')
        ]
    )

    try:
        processor = VideoProcessor(
            videos_dir=args.videos_dir,
            images_dir=args.images_dir,
            db_path=args.db_path,
            use_gpu=not args.no_gpu,
            skip_processed=not args.reprocess,
            file_age_threshold=args.file_age_threshold,
            delete_empty=not args.keep_empty,
            image_save_interval=args.image_save_interval
        )

        if args.video_file:
            # Process single video file
            if not os.path.exists(args.video_file):
                logger.error(f"Video file not found: {args.video_file}")
                return 1
            processor.process_video_file(args.video_file)

        elif args.monitor:
            # Continuous monitoring mode
            processor.monitor_and_process(check_interval=args.check_interval)

        else:
            # Process all videos once
            processor.process_all_videos(exclude_recent=True)
            processor.print_database_stats()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
