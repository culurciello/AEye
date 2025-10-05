"""
Clustering engine for person and vehicle re-identification.
Handles feature extraction, combination, and clustering.
"""

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict


def extract_color_histogram(image_bytes, bins=16):
    """Extract enhanced color histogram with spatial information.

    Args:
        image_bytes: Image data as bytes
        bins: Number of bins per channel (default 16 for finer color discrimination)

    Returns:
        Normalized color histogram vector with spatial features
    """
    if image_bytes is None:
        return None

    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        # Convert BGR to HSV (better for color-based clustering)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Global histogram
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        global_hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        global_hist = global_hist / (global_hist.sum() + 1e-7)

        # Spatial histograms (top/bottom halves for better discrimination)
        # This helps separate cars with different color patterns (e.g., different roof color)
        h, w = hsv.shape[:2]
        mid_h = h // 2

        # Top half histogram (often captures roof/hood color)
        hsv_top = hsv[:mid_h, :]
        hist_h_top = cv2.calcHist([hsv_top], [0], None, [bins//2], [0, 180])
        hist_s_top = cv2.calcHist([hsv_top], [1], None, [bins//2], [0, 256])
        top_hist = np.concatenate([hist_h_top, hist_s_top]).flatten()
        top_hist = top_hist / (top_hist.sum() + 1e-7)

        # Bottom half histogram (captures body color)
        hsv_bottom = hsv[mid_h:, :]
        hist_h_bottom = cv2.calcHist([hsv_bottom], [0], None, [bins//2], [0, 180])
        hist_s_bottom = cv2.calcHist([hsv_bottom], [1], None, [bins//2], [0, 256])
        bottom_hist = np.concatenate([hist_h_bottom, hist_s_bottom]).flatten()
        bottom_hist = bottom_hist / (bottom_hist.sum() + 1e-7)

        # Combine global + spatial features
        combined_hist = np.concatenate([
            global_hist * 0.6,      # 60% weight to global color
            top_hist * 0.2,         # 20% weight to top region
            bottom_hist * 0.2       # 20% weight to bottom region
        ])

        return combined_hist
    except Exception as e:
        print(f"Error extracting color histogram: {e}")
        return None


def combine_features(embeddings, color_features, color_weight=0.3):
    """Combine CLIP embeddings with color histograms.

    Args:
        embeddings: CLIP embeddings (N x D1)
        color_features: Color histograms (N x D2)
        color_weight: Weight for color features (0-1), default 0.3

    Returns:
        Combined feature vectors
    """
    if color_features is None or len(color_features) == 0:
        return embeddings

    # Normalize both features
    embeddings_norm = normalize(embeddings, norm='l2')
    color_norm = normalize(color_features, norm='l2')

    # Weight and combine
    embedding_weight = 1.0 - color_weight
    combined = np.concatenate([
        embeddings_norm * embedding_weight,
        color_norm * color_weight
    ], axis=1)

    return combined


def cluster_embeddings(embeddings, eps=0.4, min_samples=1):
    """Cluster embeddings using DBSCAN based on cosine similarity."""
    if len(embeddings) == 0:
        return np.array([])

    embeddings_normalized = normalize(embeddings, norm='l2')
    similarity_matrix = cosine_similarity(embeddings_normalized)
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    return labels


def group_detections_by_cluster(detections, labels):
    """Group detections by cluster label.

    Args:
        detections: List of detection dictionaries
        labels: Cluster labels from DBSCAN

    Returns:
        List of cluster dictionaries with statistics
    """
    clusters = []
    groups = defaultdict(list)

    for detection, label in zip(detections, labels):
        groups[label].append(detection)

    for cluster_id, detections_list in sorted(groups.items()):
        detections_list.sort(key=lambda x: x['timestamp'])
        clusters.append({
            'cluster_id': int(cluster_id),
            'count': len(detections_list),
            'first_seen': detections_list[0]['timestamp'],
            'last_seen': detections_list[-1]['timestamp'],
            'avg_confidence': np.mean([d['confidence'] for d in detections_list]),
            'detections': detections_list
        })

    clusters.sort(key=lambda x: x['first_seen'])
    return clusters


def extract_vehicle_color_features(vehicle_detections):
    """Extract color features from vehicle detections.

    Args:
        vehicle_detections: List of vehicle detection dictionaries

    Returns:
        numpy array of color feature vectors
    """
    print(f"Extracting enhanced color features from {len(vehicle_detections)} vehicles...")
    vehicle_color_features = []

    for detection in vehicle_detections:
        color_hist = extract_color_histogram(detection['crop'])
        # Feature dims: 16*3 (global) + 8*2 (top) + 8*2 (bottom) = 48 + 16 + 16 = 80
        vehicle_color_features.append(color_hist if color_hist is not None else np.zeros(80))

    return np.array(vehicle_color_features)
