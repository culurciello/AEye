#!/usr/bin/env python3
"""
Extract ReID features from existing object detections and store them in the database.
This allows using specialized TorchReID models for better person/vehicle clustering.

Requires: pip install torchreid
"""

import sqlite3
import argparse
import logging
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.reid_extractor import ReIDExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_database(db_path):
    """Add reid_embedding column if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA table_info(object_detections)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'reid_embedding' not in columns:
            logger.info("Adding reid_embedding column to object_detections table...")
            cursor.execute('ALTER TABLE object_detections ADD COLUMN reid_embedding BLOB')
            conn.commit()
            logger.info("Database migration complete")
    except Exception as e:
        logger.warning(f"Database migration warning: {e}")

    conn.close()


def extract_reid_features_for_class(db_path, class_names, model_type, use_gpu=True):
    """Extract ReID features for specific object classes.

    Args:
        db_path: Path to database
        class_names: List of class names to process
        model_type: 'person' or 'vehicle' for ReID model selection
        use_gpu: Whether to use GPU
    """
    # Initialize ReID extractor
    extractor = ReIDExtractor(model_type=model_type, use_gpu=use_gpu)
    if not extractor.init_model():
        logger.error("Failed to initialize ReID model")
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get detections that need ReID features
    placeholders = ','.join('?' * len(class_names))
    cursor.execute(f"""
        SELECT id, object_crop, class_name
        FROM object_detections
        WHERE class_name IN ({placeholders})
        AND object_crop IS NOT NULL
        AND reid_embedding IS NULL
    """, class_names)

    detections = cursor.fetchall()
    logger.info(f"Found {len(detections)} {model_type} detections to process")

    if len(detections) == 0:
        conn.close()
        return 0

    # Process in batches
    batch_size = 128
    processed_count = 0

    for i in range(0, len(detections), batch_size):
        batch = detections[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(detections) + batch_size - 1)//batch_size}...")

        for detection_id, crop_bytes, class_name in batch:
            # Extract ReID features
            features = extractor.extract_features(crop_bytes)

            if features is not None:
                # Serialize and store
                features_bytes = pickle.dumps(features)
                cursor.execute(
                    'UPDATE object_detections SET reid_embedding = ? WHERE id = ?',
                    (features_bytes, detection_id)
                )
                processed_count += 1

        conn.commit()

    conn.close()
    logger.info(f"Extracted ReID features for {processed_count} {model_type} detections")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description='Extract ReID features from existing object detections'
    )
    parser.add_argument(
        '--db',
        default='data/db/detections.db',
        help='Path to detections database (default: data/db/detections.db)'
    )
    parser.add_argument(
        '--persons',
        action='store_true',
        help='Extract ReID features for person detections'
    )
    parser.add_argument(
        '--vehicles',
        action='store_true',
        help='Extract ReID features for vehicle detections'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )

    args = parser.parse_args()

    # If no specific type selected, do both
    if not args.persons and not args.vehicles:
        args.persons = True
        args.vehicles = True

    # Migrate database
    migrate_database(args.db)

    total_processed = 0

    # Process persons
    if args.persons:
        logger.info("=" * 60)
        logger.info("EXTRACTING PERSON REID FEATURES")
        logger.info("=" * 60)
        count = extract_reid_features_for_class(
            args.db,
            ['person'],
            'person',
            use_gpu=not args.no_gpu
        )
        total_processed += count

    # Process vehicles
    if args.vehicles:
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING VEHICLE REID FEATURES")
        logger.info("=" * 60)
        count = extract_reid_features_for_class(
            args.db,
            ['car', 'truck', 'bus', 'motorcycle'],
            'vehicle',
            use_gpu=not args.no_gpu
        )
        total_processed += count

    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL: Processed {total_processed} detections")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
