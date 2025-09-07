#!/usr/bin/env python3
"""
Test script to load all crop_of_object from the database and save them to test_crop_of_object/ folder.
"""

import os
from database import DetectionDatabase

def export_all_crops():
    """Export all crop images from the database to test_crop_of_object/ folder."""
    # Create output directory
    output_dir = "test_crop_of_object"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database
    db = DetectionDatabase()
    
    # Get all detections
    detections = db.get_all_detections()
    
    if not detections:
        print("No detections found in database.")
        return
    
    print(f"Found {len(detections)} detections. Exporting crops...")
    
    exported_count = 0
    
    for detection in detections:
        # Unpack detection data
        (detection_id, object_type, time, crop_bytes, video_link, 
         frame_num, caption, embeddings, confidence, 
         bbox_x, bbox_y, bbox_width, bbox_height, created_at) = detection
        
        if crop_bytes:
            # Convert bytes to image
            try:
                image = db.bytes_to_image(crop_bytes)
                
                # Create filename with detection info
                filename = f"crop_{detection_id}_{object_type}_{frame_num}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save image using cv2
                import cv2
                success = cv2.imwrite(filepath, image)
                
                if success:
                    exported_count += 1
                    print(f"Exported: {filename}")
                else:
                    print(f"Failed to save: {filename}")
                    
            except Exception as e:
                print(f"Error processing detection {detection_id}: {e}")
        else:
            print(f"No crop data for detection {detection_id}")
    
    print(f"\nExport completed. {exported_count} crops saved to {output_dir}/")

if __name__ == "__main__":
    export_all_crops()