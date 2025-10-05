#!/usr/bin/env python3
"""
Generate HTML report with clustered face, person, and vehicle detections.
Supports both CLIP and ReID embeddings for clustering.
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.data_loader import migrate_database, load_face_detections, load_object_detections
from lib.clustering_engine import (
    cluster_embeddings,
    group_detections_by_cluster,
    extract_vehicle_color_features,
    combine_features
)
from lib.report_generator import generate_html_report


def save_crops(items, labels, output_dir, prefix='item', base_dir=None):
    """Save crop images to disk and return image paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_dir is None:
        base_dir = output_dir

    image_paths = {}

    for item, label in zip(items, labels):
        if item['crop'] is None:
            continue

        # Create cluster/track directory
        cluster_dir = output_dir / f"{prefix}_{label}"
        cluster_dir.mkdir(exist_ok=True)

        # Save image
        img_filename = f"{prefix}_{item['id']}.jpg"
        img_path = cluster_dir / img_filename

        with open(img_path, 'wb') as f:
            f.write(item['crop'])

        # Return path relative to base_dir for HTML links
        image_paths[item['id']] = str(img_path.relative_to(base_dir))

    return image_paths


def cluster_and_save(detections, embeddings, labels, output_dir, prefix, base_dir):
    """Cluster detections and save images.

    Args:
        detections: List of detection dictionaries
        embeddings: Feature embeddings
        labels: Cluster labels
        output_dir: Directory to save images
        prefix: Prefix for image files
        base_dir: Base directory for relative paths

    Returns:
        Tuple of (clusters, image_paths)
    """
    clusters = group_detections_by_cluster(detections, labels)
    image_paths = save_crops(detections, labels, output_dir, prefix=prefix, base_dir=base_dir)
    return clusters, image_paths


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML report with clustered faces, persons, and vehicles'
    )
    parser.add_argument(
        '--db',
        default='data/db/detections.db',
        help='Path to detections database (default: data/db/detections.db)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.4,
        help='DBSCAN eps parameter for face clustering (default: 0.4)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=1,
        help='DBSCAN min_samples parameter (default: 1)'
    )
    parser.add_argument(
        '--person-eps',
        type=float,
        default=0.15,
        help='DBSCAN eps parameter for person clustering (default: 0.15)'
    )
    parser.add_argument(
        '--vehicle-eps',
        type=float,
        default=0.08,
        help='DBSCAN eps parameter for vehicle clustering (default: 0.08, lower = more clusters)'
    )
    parser.add_argument(
        '--color-weight',
        type=float,
        default=0.5,
        help='Weight for color features in vehicle clustering, 0-1 (default: 0.5)'
    )
    parser.add_argument(
        '--vehicle-min-samples',
        type=int,
        default=1,
        help='Min samples for vehicle clustering (default: 1, higher = stricter clusters)'
    )
    parser.add_argument(
        '--use-reid',
        action='store_true',
        help='Use ReID embeddings instead of CLIP for person/vehicle clustering (requires running extract_reid_features.py first)'
    )
    parser.add_argument(
        '--reid-person-eps',
        type=float,
        default=0.5,
        help='DBSCAN eps for person ReID clustering (default: 0.5)'
    )
    parser.add_argument(
        '--reid-vehicle-eps',
        type=float,
        default=0.6,
        help='DBSCAN eps for vehicle ReID clustering (default: 0.6)'
    )
    parser.add_argument(
        '--output',
        default='report/detection_report.html',
        help='Output HTML file path (default: report/detection_report.html)'
    )
    parser.add_argument(
        '--images-dir',
        default='report/timeline_images',
        help='Directory to save images (default: report/timeline_images)'
    )

    args = parser.parse_args()

    # Ensure database has required columns
    migrate_database(args.db)

    # Get the directory where HTML will be saved (for relative image paths)
    html_dir = Path(args.output).parent
    html_dir.mkdir(parents=True, exist_ok=True)

    # Determine embedding type
    embedding_type = "ReID" if args.use_reid else "CLIP"
    print(f"Using {embedding_type} embeddings for person/vehicle clustering")

    # ========================================
    # LOAD DATA
    # ========================================
    print(f"\nLoading face detections from {args.db}...")
    face_detections, face_embeddings = load_face_detections(args.db)
    print(f"Loaded {len(face_detections)} face detections")

    print(f"Loading person detections from {args.db}...")
    person_detections, person_embeddings = load_object_detections(args.db, ['person'], use_reid=args.use_reid)
    print(f"Loaded {len(person_detections)} person detections with {embedding_type} embeddings")

    print(f"Loading vehicle detections from {args.db}...")
    vehicle_detections, vehicle_embeddings = load_object_detections(
        args.db, ['car', 'truck', 'bus', 'motorcycle'], use_reid=args.use_reid
    )
    print(f"Loaded {len(vehicle_detections)} vehicle detections with {embedding_type} embeddings")

    # ========================================
    # CLUSTER FACES
    # ========================================
    face_clusters = []
    face_image_paths = {}

    if len(face_detections) > 0:
        print(f"\nClustering faces with eps={args.eps}, min_samples={args.min_samples}...")
        face_labels = cluster_embeddings(face_embeddings, eps=args.eps, min_samples=args.min_samples)

        print(f"Saving face images to {args.images_dir}/faces/...")
        face_clusters, face_image_paths = cluster_and_save(
            face_detections, face_embeddings, face_labels,
            os.path.join(args.images_dir, 'faces'), 'face', html_dir
        )
        print(f"Saved {len(face_image_paths)} face images in {len(face_clusters)} clusters")

    # ========================================
    # CLUSTER PERSONS
    # ========================================
    person_clusters = []
    person_image_paths = {}

    if len(person_detections) > 0:
        person_eps = args.reid_person_eps if args.use_reid else args.person_eps
        print(f"\nClustering persons with eps={person_eps}, min_samples={args.min_samples}...")
        person_labels = cluster_embeddings(person_embeddings, eps=person_eps, min_samples=args.min_samples)

        print(f"Saving person images to {args.images_dir}/persons/...")
        person_clusters, person_image_paths = cluster_and_save(
            person_detections, person_embeddings, person_labels,
            os.path.join(args.images_dir, 'persons'), 'person', html_dir
        )
        print(f"Saved {len(person_image_paths)} person images in {len(person_clusters)} clusters")

    # ========================================
    # CLUSTER VEHICLES
    # ========================================
    vehicle_clusters = []
    vehicle_image_paths = {}

    if len(vehicle_detections) > 0:
        vehicle_eps = args.reid_vehicle_eps if args.use_reid else args.vehicle_eps

        # Use color features only with CLIP embeddings, not with ReID
        # (ReID models already encode color and appearance information)
        if args.use_reid:
            print(f"\nClustering vehicles with ReID embeddings (eps={vehicle_eps}, min_samples={args.vehicle_min_samples})...")
            vehicle_combined = vehicle_embeddings
        else:
            vehicle_color_features = extract_vehicle_color_features(vehicle_detections)
            print(f"Combining CLIP embeddings with color features (color_weight={args.color_weight})...")
            vehicle_combined = combine_features(vehicle_embeddings, vehicle_color_features, color_weight=args.color_weight)
            print(f"Clustering vehicles with eps={vehicle_eps}, min_samples={args.vehicle_min_samples}...")

        vehicle_labels = cluster_embeddings(vehicle_combined, eps=vehicle_eps, min_samples=args.vehicle_min_samples)

        print(f"Saving vehicle images to {args.images_dir}/vehicles/...")
        vehicle_clusters, vehicle_image_paths = cluster_and_save(
            vehicle_detections, vehicle_combined, vehicle_labels,
            os.path.join(args.images_dir, 'vehicles'), 'vehicle', html_dir
        )
        print(f"Saved {len(vehicle_image_paths)} vehicle images in {len(vehicle_clusters)} clusters")

    # ========================================
    # GENERATE HTML REPORT
    # ========================================
    print(f"\nGenerating HTML report...")
    generate_html_report(
        face_clusters, person_clusters, vehicle_clusters,
        face_image_paths, person_image_paths, vehicle_image_paths,
        args.output
    )

    print(f"\n✓ Report generated: {args.output}")
    print(f"✓ Images saved to: {args.images_dir}/")

    # ========================================
    # PRINT SUMMARY
    # ========================================
    print(f"\nSummary:")
    print(f"  Embedding Type: {embedding_type}")
    print(f"  Faces (eps={args.eps}):")
    print(f"    - Total detections: {len(face_detections)}")
    print(f"    - Unique clusters: {len([c for c in face_clusters if c['cluster_id'] != -1])}")
    print(f"    - Outliers: {sum(c['count'] for c in face_clusters if c['cluster_id'] == -1)}")

    person_eps_display = args.reid_person_eps if args.use_reid else args.person_eps
    print(f"  Persons (eps={person_eps_display}):")
    print(f"    - Total detections: {len(person_detections)}")
    print(f"    - Unique clusters: {len([c for c in person_clusters if c['cluster_id'] != -1])}")
    print(f"    - Outliers: {sum(c['count'] for c in person_clusters if c['cluster_id'] == -1)}")

    vehicle_eps_display = args.reid_vehicle_eps if args.use_reid else args.vehicle_eps
    color_info = "" if args.use_reid else f", color_weight={args.color_weight}"
    print(f"  Vehicles (eps={vehicle_eps_display}{color_info}, min_samples={args.vehicle_min_samples}):")
    print(f"    - Total detections: {len(vehicle_detections)}")
    print(f"    - Unique clusters: {len([c for c in vehicle_clusters if c['cluster_id'] != -1])}")
    print(f"    - Outliers: {sum(c['count'] for c in vehicle_clusters if c['cluster_id'] == -1)}")
    if len(vehicle_clusters) > 0:
        avg_cluster_size = np.mean([c['count'] for c in vehicle_clusters if c['cluster_id'] != -1]) if len([c for c in vehicle_clusters if c['cluster_id'] != -1]) > 0 else 0
        print(f"    - Avg cluster size: {avg_cluster_size:.1f} detections")


if __name__ == '__main__':
    main()
