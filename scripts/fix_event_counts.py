#!/usr/bin/env python3
"""
Script to fix face_count, object_count, and processed status in motion_events table.
This script recalculates counts from the actual detections in the database.
"""

import sqlite3
import os
import sys

# Add parent directory to path to import lib modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fix_event_counts(db_path='data/db/detections.db'):
    """Fix face_count, object_count, and processed status for all motion events."""

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all motion events
    cursor.execute("SELECT id FROM motion_events ORDER BY id")
    event_ids = [row[0] for row in cursor.fetchall()]

    print(f"Found {len(event_ids)} motion events to check")
    print("-" * 60)

    updated_count = 0

    for event_id in event_ids:
        # Count face detections for this event
        cursor.execute(
            "SELECT COUNT(*) FROM face_detections WHERE motion_event_id = ?",
            (event_id,)
        )
        face_count = cursor.fetchone()[0]

        # Count object detections for this event
        cursor.execute(
            "SELECT COUNT(*) FROM object_detections WHERE motion_event_id = ?",
            (event_id,)
        )
        object_count = cursor.fetchone()[0]

        # Count object tracks for this event
        cursor.execute(
            "SELECT COUNT(*) FROM object_tracks WHERE motion_event_id = ?",
            (event_id,)
        )
        track_count = cursor.fetchone()[0]

        # Get current values
        cursor.execute(
            "SELECT face_count, object_count, track_count, processed, video_file FROM motion_events WHERE id = ?",
            (event_id,)
        )
        current = cursor.fetchone()
        current_face_count = current[0] if current[0] is not None else 0
        current_object_count = current[1] if current[1] is not None else 0
        current_track_count = current[2] if current[2] is not None else 0
        current_processed = current[3] if current[3] is not None else 0
        video_file = current[4]

        # Determine if event should be marked as processed
        # An event is considered processed if it has any detections OR if it has been scanned
        # For now, we'll mark as processed if there are any face or object detections
        should_be_processed = (face_count > 0 or object_count > 0 or track_count > 0)

        # Check if update is needed
        needs_update = (
            face_count != current_face_count or
            object_count != current_object_count or
            track_count != current_track_count or
            (should_be_processed and not current_processed)
        )

        if needs_update:
            # Update the motion event
            cursor.execute("""
                UPDATE motion_events
                SET face_count = ?, object_count = ?, track_count = ?, processed = ?
                WHERE id = ?
            """, (face_count, object_count, track_count, 1 if should_be_processed else 0, event_id))

            updated_count += 1
            print(f"Event {event_id:3d} ({video_file})")
            print(f"  Face count:   {current_face_count:3d} -> {face_count:3d}")
            print(f"  Object count: {current_object_count:3d} -> {object_count:3d}")
            print(f"  Track count:  {current_track_count:3d} -> {track_count:3d}")
            print(f"  Processed:    {current_processed:3d} -> {1 if should_be_processed else 0:3d}")
            print()

    # Commit changes
    conn.commit()
    conn.close()

    print("-" * 60)
    print(f"Updated {updated_count} motion events")
    print(f"No changes needed for {len(event_ids) - updated_count} events")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fix face_count and object_count in motion_events')
    parser.add_argument('--db-path', default='data/db/detections.db',
                       help='Path to database file (default: data/db/detections.db)')

    args = parser.parse_args()

    print("Fixing event counts in database...")
    print(f"Database: {args.db_path}")
    print()

    fix_event_counts(args.db_path)