#!/usr/bin/env python3

import sqlite3
import numpy as np
import cv2
import os
import pickle
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import argparse
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObjectAnalyzer:
    """Analyzes detected objects from the database and groups similar objects together."""

    def __init__(self, db_path: str, output_dir: str = "data/reports"):
        """Initialize the object analyzer.

        Args:
            db_path: Path to the SQLite database file
            output_dir: Base directory to save analysis reports (subdirs created per object type)
        """
        self.db_path = db_path
        self.base_output_dir = output_dir

        # Object classes we're interested in
        self.target_classes = ['car', 'bicycle', 'motorbike', 'motorcycle', 'person', 'face']

    def _get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def get_all_detections(self) -> Dict[str, List[Dict]]:
        """Get all object and face detections from database.

        Returns:
            Dictionary with object class as key and list of detection data as values
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        detections = defaultdict(list)

        # Get object detections
        cursor.execute('''
            SELECT od.id, od.motion_event_id, od.frame_timestamp, od.class_name,
                   od.confidence, od.bbox_x, od.bbox_y, od.bbox_width, od.bbox_height,
                   me.video_file, od.object_crop
            FROM object_detections od
            JOIN motion_events me ON od.motion_event_id = me.id
            WHERE od.class_name IN ({})
            ORDER BY od.frame_timestamp
        '''.format(','.join('?' * len(self.target_classes))), self.target_classes)

        for row in cursor.fetchall():
            detection = {
                'id': row[0],
                'motion_event_id': row[1],
                'timestamp': row[2],
                'class_name': row[3],
                'confidence': row[4],
                'bbox': [row[5], row[6], row[7], row[8]],  # x, y, w, h
                'video_file': row[9],
                'object_crop': row[10],
                'type': 'object'
            }
            detections[row[3]].append(detection)

        # Get face detections (treat as 'face' class)
        cursor.execute('''
            SELECT fd.id, fd.motion_event_id, fd.frame_timestamp, fd.confidence,
                   fd.bbox_x, fd.bbox_y, fd.bbox_width, fd.bbox_height,
                   fd.face_embedding, fd.known_person, me.video_file, fd.face_crop
            FROM face_detections fd
            JOIN motion_events me ON fd.motion_event_id = me.id
            ORDER BY fd.frame_timestamp
        ''')

        for row in cursor.fetchall():
            # Deserialize face embedding if available
            embedding = None
            if row[8]:
                try:
                    embedding = pickle.loads(row[8])
                except:
                    embedding = None

            detection = {
                'id': row[0],
                'motion_event_id': row[1],
                'timestamp': row[2],
                'class_name': 'face',
                'confidence': row[3],
                'bbox': [row[4], row[5], row[6], row[7]],  # x, y, w, h
                'embedding': embedding,
                'known_person': row[9],
                'video_file': row[10],
                'face_crop': row[11],
                'type': 'face'
            }
            detections['face'].append(detection)

        conn.close()
        logger.info(f"Retrieved {sum(len(v) for v in detections.values())} total detections")
        return dict(detections)

    def group_similar_faces(self, face_detections: List[Dict]) -> List[List[Dict]]:
        """Group similar faces using embeddings and clustering.

        Args:
            face_detections: List of face detection dictionaries

        Returns:
            List of groups, where each group is a list of similar face detections
        """
        if not face_detections:
            return []

        # Extract valid embeddings
        valid_faces = []
        embeddings = []

        for face in face_detections:
            if face['embedding'] is not None and isinstance(face['embedding'], np.ndarray):
                valid_faces.append(face)
                embeddings.append(face['embedding'].flatten())

        if len(embeddings) < 2:
            return [valid_faces] if valid_faces else []

        # Normalize embeddings
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Use DBSCAN clustering on cosine distance
        # Convert cosine similarity to distance: distance = 1 - similarity
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix

        # Ensure no negative values (clamp to 0)
        distance_matrix = np.maximum(distance_matrix, 0)

        # Make matrix symmetric and set diagonal to 0
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # DBSCAN parameters - adjust based on your data
        eps = 0.3  # Max distance between points in same cluster
        min_samples = 2  # Minimum samples per cluster

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group faces by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(valid_faces[idx])

        # Convert to list and sort by cluster size
        groups = list(clusters.values())
        groups.sort(key=len, reverse=True)

        logger.info(f"Grouped {len(valid_faces)} faces into {len(groups)} clusters")
        return groups

    def group_similar_objects(self, object_detections: List[Dict]) -> List[List[Dict]]:
        """Group similar objects based on spatial and temporal proximity.

        Args:
            object_detections: List of object detection dictionaries

        Returns:
            List of groups, where each group is a list of similar object detections
        """
        if not object_detections:
            return []

        # Sort by timestamp
        object_detections.sort(key=lambda x: x['timestamp'])

        groups = []
        used_detections = set()

        for i, detection in enumerate(object_detections):
            if i in used_detections:
                continue

            # Start a new group with current detection
            current_group = [detection]
            used_detections.add(i)

            # Find similar detections within time and space windows
            for j, other_detection in enumerate(object_detections[i+1:], i+1):
                if j in used_detections:
                    continue

                # Time proximity check (within 10 minutes)
                try:
                    if isinstance(other_detection['timestamp'], str):
                        other_time = datetime.fromisoformat(other_detection['timestamp'])
                    else:
                        other_time = other_detection['timestamp']

                    if isinstance(detection['timestamp'], str):
                        current_time = datetime.fromisoformat(detection['timestamp'])
                    else:
                        current_time = detection['timestamp']

                    time_diff = abs((other_time - current_time).total_seconds())
                except (ValueError, TypeError):
                    continue
                if time_diff > 600:  # 10 minutes
                    continue

                # Spatial proximity check (similar bounding box center)
                bbox1 = detection['bbox']
                bbox2 = other_detection['bbox']

                center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
                center2 = (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)

                # Calculate normalized distance (relative to image size, assume 640x480)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                normalized_distance = distance / np.sqrt(640**2 + 480**2)

                # Size similarity check
                size1 = bbox1[2] * bbox1[3]  # width * height
                size2 = bbox2[2] * bbox2[3]
                size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

                # Group if spatially close and similar size
                if normalized_distance < 0.1 and size_ratio > 0.5:  # Adjust thresholds as needed
                    current_group.append(other_detection)
                    used_detections.add(j)

            groups.append(current_group)

        # Sort groups by size
        groups.sort(key=len, reverse=True)

        logger.info(f"Grouped {len(object_detections)} objects into {len(groups)} spatial-temporal clusters")
        return groups

    def create_group_visualizations(self, groups: List[List[Dict]], class_name: str, max_groups: int = 10):
        """Create visualization images for the top groups.

        Args:
            groups: List of detection groups
            class_name: Object class name
            max_groups: Maximum number of groups to visualize
        """
        # Create class-specific directory structure
        class_dir = os.path.join(self.base_output_dir, class_name)
        images_dir = os.path.join(class_dir, "images")
        os.makedirs(class_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        for group_idx, group in enumerate(groups[:max_groups]):
            if not group:
                continue

            # Create a grid of detections for this group
            group_size = len(group)
            cols = min(5, group_size)  # Max 5 columns
            rows = (group_size + cols - 1) // cols  # Ceiling division

            # Image dimensions for each cell
            cell_width, cell_height = 150, 150
            grid_width = cols * cell_width
            grid_height = rows * cell_height

            # Create output image
            output_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            output_img.fill(50)  # Dark gray background

            for idx, detection in enumerate(group):
                row = idx // cols
                col = idx % cols

                cell_x = col * cell_width
                cell_y = row * cell_height

                # Create a placeholder image for this detection
                cell_img = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

                # Try to load crop image (face or object)
                crop_loaded = False
                if detection['type'] == 'face' and detection.get('face_crop'):
                    # Use face crop if available
                    try:
                        face_crop = detection['face_crop']
                        if isinstance(face_crop, bytes):
                            # Convert bytes to numpy array
                            nparr = np.frombuffer(face_crop, np.uint8)
                            crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if crop_img is not None:
                                # Resize to fit cell
                                crop_img = cv2.resize(crop_img, (cell_width-10, cell_height-40))
                                cell_img[5:cell_height-35, 5:cell_width-5] = crop_img
                                crop_loaded = True
                    except Exception as e:
                        logger.warning(f"Could not load face crop: {e}")

                elif detection['type'] == 'object' and detection.get('object_crop'):
                    # Use object crop if available
                    try:
                        object_crop = detection['object_crop']
                        if isinstance(object_crop, bytes):
                            # Convert bytes to numpy array
                            nparr = np.frombuffer(object_crop, np.uint8)
                            crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if crop_img is not None:
                                # Resize to fit cell
                                crop_img = cv2.resize(crop_img, (cell_width-10, cell_height-40))
                                cell_img[5:cell_height-35, 5:cell_width-5] = crop_img
                                crop_loaded = True
                    except Exception as e:
                        logger.warning(f"Could not load object crop: {e}")

                # If no crop was loaded, show placeholder with class name
                if not crop_loaded:
                    placeholder_text = detection.get('class_name', 'Unknown')
                    cv2.putText(cell_img, placeholder_text, (10, cell_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

                # Add detection info text
                try:
                    info_text = f"ID:{detection['id']}"
                    cv2.putText(cell_img, info_text, (5, cell_height-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    conf_text = f"Conf:{detection['confidence']:.2f}"
                    cv2.putText(cell_img, conf_text, (5, cell_height-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except Exception as e:
                    logger.warning(f"Error adding text to visualization: {e}")
                    # Add minimal text
                    cv2.putText(cell_img, f"ID:{detection.get('id', 'N/A')}", (5, cell_height-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Add timestamp
                try:
                    if detection['timestamp']:
                        timestamp = str(detection['timestamp'])[:16]
                    else:
                        timestamp = "N/A"
                except:
                    timestamp = "N/A"
                cv2.putText(cell_img, timestamp, (5, cell_height-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

                # Draw border
                cv2.rectangle(cell_img, (0, 0), (cell_width-1, cell_height-1), (100, 100, 100), 1)

                # Place cell in output image
                output_img[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width] = cell_img

            # Save group image
            filename = f"{class_name}_group_{group_idx+1:02d}_{len(group):03d}_detections.jpg"
            output_path = os.path.join(images_dir, filename)
            cv2.imwrite(output_path, output_img)
            logger.info(f"Created visualization: {filename}")

    def generate_timeline_data(self, all_detections: Dict[str, List[Dict]]) -> Dict:
        """Generate timeline data for all detections.

        Args:
            all_detections: Dictionary of all detections by class

        Returns:
            Timeline data dictionary
        """
        timeline = defaultdict(lambda: defaultdict(int))

        for class_name, detections in all_detections.items():
            for detection in detections:
                # Group by hour
                try:
                    if isinstance(detection['timestamp'], str):
                        timestamp = datetime.fromisoformat(detection['timestamp'])
                    else:
                        timestamp = detection['timestamp']
                    hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                    timeline[hour_key][class_name] += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse timestamp {detection['timestamp']}: {e}")
                    continue

        # Convert to regular dict and sort by time
        timeline_data = {}
        for time_key in sorted(timeline.keys()):
            timeline_data[time_key] = dict(timeline[time_key])

        return timeline_data

    def generate_summary_report(self, all_detections: Dict[str, List[Dict]],
                              all_groups: Dict[str, List[List[Dict]]],
                              timeline_data: Dict) -> str:
        """Generate a comprehensive summary report.

        Args:
            all_detections: All detections by class
            all_groups: All grouped detections by class
            timeline_data: Timeline data

        Returns:
            Summary report as string
        """
        report = []
        report.append("=== AEye Object Analysis Summary Report ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall statistics
        total_detections = sum(len(detections) for detections in all_detections.values())
        report.append(f"Total Detections: {total_detections}")
        report.append("")

        # Per-class statistics
        report.append("=== Detection Statistics by Class ===")
        for class_name in sorted(all_detections.keys()):
            detections = all_detections[class_name]
            groups = all_groups.get(class_name, [])

            report.append(f"\n{class_name}:")
            report.append(f"  Total Detections: {len(detections)}")
            report.append(f"  Groups Found: {len(groups)}")

            if groups:
                group_sizes = [len(group) for group in groups]
                report.append(f"  Largest Group: {max(group_sizes)} detections")
                report.append(f"  Average Group Size: {np.mean(group_sizes):.1f}")

                # Top 5 groups
                report.append(f"  Top 5 Groups by Size:")
                for i, group in enumerate(groups[:5]):
                    report.append(f"    Group {i+1}: {len(group)} detections")

            # Confidence statistics
            if detections:
                confidences = []
                for d in detections:
                    conf = d.get('confidence')
                    if conf is not None:
                        try:
                            if isinstance(conf, bytes):
                                # Skip binary data that can't be converted
                                continue
                            confidences.append(float(conf))
                        except (ValueError, TypeError):
                            # Skip invalid confidence values
                            continue

                if confidences:
                    report.append(f"  Average Confidence: {np.mean(confidences):.3f}")
                    report.append(f"  Min/Max Confidence: {min(confidences):.3f}/{max(confidences):.3f}")
                else:
                    report.append(f"  No valid confidence data available")

        # Timeline summary
        report.append("\n=== Detection Timeline Summary ===")
        if timeline_data:
            # Find peak activity hours
            hourly_totals = {}
            for time_key, detections in timeline_data.items():
                hourly_totals[time_key] = sum(detections.values())

            # Sort by activity
            sorted_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)

            report.append(f"Most Active Hours:")
            for i, (hour, total) in enumerate(sorted_hours[:5]):
                report.append(f"  {i+1}. {hour}: {total} detections")

            # Daily summary
            daily_totals = defaultdict(int)
            for time_key, detections in timeline_data.items():
                day = time_key.split(' ')[0]  # Extract date part
                daily_totals[day] += sum(detections.values())

            report.append(f"\nDaily Activity Summary:")
            for day in sorted(daily_totals.keys()):
                report.append(f"  {day}: {daily_totals[day]} detections")

        # Face recognition summary if faces detected
        if 'face' in all_detections:
            face_detections = all_detections['face']
            known_faces = [d for d in face_detections if d.get('known_person')]

            report.append(f"\n=== Face Recognition Summary ===")
            report.append(f"Total Face Detections: {len(face_detections)}")
            report.append(f"Known Faces: {len(known_faces)}")
            report.append(f"Unknown Faces: {len(face_detections) - len(known_faces)}")

            if known_faces:
                person_counts = Counter(d['known_person'] for d in known_faces)
                report.append(f"Recognized People:")
                for person, count in person_counts.most_common():
                    report.append(f"  {person}: {count} detections")

        return "\n".join(report)

    def run_analysis(self):
        """Run the complete object analysis."""
        logger.info("Starting object analysis...")

        # Get all detections
        all_detections = self.get_all_detections()

        if not all_detections:
            logger.warning("No detections found in database")
            return

        # Group similar objects for each class
        all_groups = {}

        for class_name, detections in all_detections.items():
            logger.info(f"Analyzing {len(detections)} {class_name} detections...")

            if class_name == 'face':
                groups = self.group_similar_faces(detections)
            else:
                groups = self.group_similar_objects(detections)

            all_groups[class_name] = groups

            # Create visualizations
            if groups:
                self.create_group_visualizations(groups, class_name)

        # Generate timeline
        timeline_data = self.generate_timeline_data(all_detections)

        # Save class-specific reports
        for class_name, detections in all_detections.items():
            if not detections:
                continue

            class_dir = os.path.join(self.base_output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Save class-specific timeline
            class_timeline = {}
            for time_key, time_detections in timeline_data.items():
                if class_name in time_detections:
                    class_timeline[time_key] = {class_name: time_detections[class_name]}

            timeline_path = os.path.join(class_dir, "detection_timeline.json")
            with open(timeline_path, 'w') as f:
                json.dump(class_timeline, f, indent=2)

            # Generate class-specific summary report
            class_groups = {class_name: all_groups.get(class_name, [])}
            class_detections = {class_name: detections}
            summary_report = self.generate_summary_report(class_detections, class_groups, class_timeline)

            # Save class-specific summary report
            summary_path = os.path.join(class_dir, "summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary_report)

            logger.info(f"Analysis complete for {class_name}! Check {class_dir} for results")

        # Save overall summary
        overall_summary = self.generate_summary_report(all_detections, all_groups, timeline_data)
        overall_summary_path = os.path.join(self.base_output_dir, "overall_summary.txt")
        with open(overall_summary_path, 'w') as f:
            f.write(overall_summary)

        # Save overall timeline
        overall_timeline_path = os.path.join(self.base_output_dir, "overall_timeline.json")
        with open(overall_timeline_path, 'w') as f:
            json.dump(timeline_data, f, indent=2)

        logger.info(f"Overall analysis complete! Check {self.base_output_dir} for results")

        print(f"\nAnalysis Summary:")
        print(f"- Total detections processed: {sum(len(d) for d in all_detections.values())}")
        print(f"- Classes analyzed: {list(all_detections.keys())}")
        print(f"- Groups created: {sum(len(g) for g in all_groups.values())}")
        print(f"- Output directory: {self.base_output_dir}")
        print(f"- Class-specific reports created in: data/reports/OBJECT_TYPE/")
        for class_name in all_detections.keys():
            print(f"  - data/reports/{class_name}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze object detections and group similar objects')
    parser.add_argument('--db-path', default='data/db/detections.db',
                       help='Path to the SQLite database file')
    parser.add_argument('--output-dir', default='data/reports',
                       help='Base output directory for analysis reports (subdirs created per object type)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set the logging level')

    args = parser.parse_args()

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check if database exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found: {args.db_path}")
        return 1

    # Run analysis
    try:
        analyzer = ObjectAnalyzer(args.db_path, args.output_dir)
        analyzer.run_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())