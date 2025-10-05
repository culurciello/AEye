"""
Data loading functions for detection clustering.
Loads face, person, and vehicle detections from database.
"""

import sqlite3
import numpy as np
import pickle
import struct
from datetime import datetime


def migrate_database(db_path):
    """Ensure database has object_embedding and reid_embedding columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if object_embedding column exists
    try:
        cursor.execute("PRAGMA table_info(object_detections)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'object_embedding' not in columns:
            print("Adding object_embedding column to database...")
            cursor.execute('ALTER TABLE object_detections ADD COLUMN object_embedding BLOB')
            conn.commit()
            print("object_embedding column added")

        if 'reid_embedding' not in columns:
            print("Adding reid_embedding column to database...")
            cursor.execute('ALTER TABLE object_detections ADD COLUMN reid_embedding BLOB')
            conn.commit()
            print("reid_embedding column added")

    except Exception as e:
        print(f"Database migration warning: {e}")

    conn.close()


def load_face_detections(db_path):
    """Load face detections with embeddings and crops from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, frame_timestamp, face_embedding, face_crop, confidence,
               bbox_x, bbox_y, bbox_width, bbox_height, motion_event_id
        FROM face_detections
        WHERE face_embedding IS NOT NULL
        ORDER BY frame_timestamp
    """)

    detections = []
    embeddings = []
    invalid_count = 0

    for row in cursor.fetchall():
        detection_id, timestamp, embedding_blob, face_crop, confidence, x, y, w, h, event_id = row

        try:
            embedding = pickle.loads(embedding_blob)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten()
            else:
                embedding = np.array(embedding).flatten()

            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                invalid_count += 1
                continue
        except Exception as e:
            print(f"Error deserializing face embedding {detection_id}: {e}")
            invalid_count += 1
            continue

        if isinstance(confidence, bytes) and len(confidence) == 4:
            confidence = struct.unpack('f', confidence)[0]
        elif confidence is not None:
            confidence = float(confidence)
        else:
            confidence = 0.0

        detections.append({
            'id': detection_id,
            'timestamp': datetime.fromisoformat(timestamp),
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'motion_event_id': event_id,
            'crop': face_crop,
            'type': 'face'
        })
        embeddings.append(embedding)

    conn.close()

    if invalid_count > 0:
        print(f"Skipped {invalid_count} face detections with invalid embeddings")

    return detections, np.array(embeddings) if embeddings else np.array([])


def load_object_detections(db_path, class_names, use_reid=False):
    """Load object detections with embeddings from database.

    Args:
        db_path: Path to database
        class_names: List of class names to load (e.g., ['person'] or ['car', 'truck', 'bus'])
        use_reid: If True, use reid_embedding instead of object_embedding (CLIP)

    Returns:
        Tuple of (detections, embeddings)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    embedding_col = 'reid_embedding' if use_reid else 'object_embedding'

    placeholders = ','.join('?' * len(class_names))
    cursor.execute(f"""
        SELECT id, frame_timestamp, class_name, {embedding_col}, object_crop, confidence,
               bbox_x, bbox_y, bbox_width, bbox_height, motion_event_id
        FROM object_detections
        WHERE class_name IN ({placeholders}) AND {embedding_col} IS NOT NULL
        ORDER BY frame_timestamp
    """, class_names)

    detections = []
    embeddings = []
    invalid_count = 0

    for row in cursor.fetchall():
        detection_id, timestamp, class_name, embedding_blob, crop, confidence, x, y, w, h, event_id = row

        try:
            embedding = pickle.loads(embedding_blob)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten()
            else:
                embedding = np.array(embedding).flatten()

            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                invalid_count += 1
                continue
        except Exception as e:
            print(f"Error deserializing object embedding {detection_id}: {e}")
            invalid_count += 1
            continue

        if isinstance(confidence, bytes) and len(confidence) == 4:
            confidence = struct.unpack('f', confidence)[0]
        elif confidence is not None:
            confidence = float(confidence)
        else:
            confidence = 0.0

        detections.append({
            'id': detection_id,
            'timestamp': datetime.fromisoformat(timestamp),
            'class_name': class_name,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'motion_event_id': event_id,
            'crop': crop,
            'type': class_name
        })
        embeddings.append(embedding)

    conn.close()

    if invalid_count > 0:
        print(f"Skipped {invalid_count} {class_names} detections with invalid embeddings")

    return detections, np.array(embeddings) if embeddings else np.array([])
