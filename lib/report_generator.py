"""
HTML report generator for detection clustering results.
Generates dark mode reports with face, person, and vehicle clusters.
"""

import numpy as np


def generate_html_report(face_clusters, person_clusters, vehicle_clusters,
                         face_image_paths, person_image_paths, vehicle_image_paths, output_file):
    """Generate unified HTML report with faces, persons, and vehicles.

    Args:
        face_clusters: List of face cluster dictionaries
        person_clusters: List of person cluster dictionaries
        vehicle_clusters: List of vehicle cluster dictionaries
        face_image_paths: Dict mapping face detection IDs to image paths
        person_image_paths: Dict mapping person detection IDs to image paths
        vehicle_image_paths: Dict mapping vehicle detection IDs to image paths
        output_file: Path to save HTML report
    """
    # Calculate statistics
    total_faces = sum(c['count'] for c in face_clusters)
    total_persons = sum(c['count'] for c in person_clusters)
    total_vehicles = sum(c['count'] for c in vehicle_clusters)
    unique_face_clusters = len([c for c in face_clusters if c['cluster_id'] != -1])
    unique_person_clusters = len([c for c in person_clusters if c['cluster_id'] != -1])
    unique_vehicle_clusters = len([c for c in vehicle_clusters if c['cluster_id'] != -1])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Report - Faces & Cars</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background: #1a1a1a;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            background: #252525;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}

        .tab {{
            padding: 15px 30px;
            background: #333;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            color: #e0e0e0;
            transition: all 0.3s;
        }}

        .tab:hover {{
            background: #444;
        }}

        .tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .summary {{
            background: #252525;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}

        .summary-item {{
            text-align: center;
            padding: 15px;
            background: #333;
            border-radius: 8px;
        }}

        .summary-value {{
            font-size: 2em;
            font-weight: bold;
            color: #8b9dff;
        }}

        .summary-label {{
            color: #aaa;
            margin-top: 5px;
        }}

        .timeline {{
            background: #252525;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
            max-height: 600px;
            overflow-y: auto;
        }}

        .timeline h2 {{
            margin-bottom: 20px;
            color: #8b9dff;
            position: sticky;
            top: 0;
            background: #252525;
            padding: 10px 0;
        }}

        .timeline-item {{
            display: flex;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: #333;
            border-radius: 8px;
            border-left: 4px solid #8b9dff;
            transition: transform 0.2s;
        }}

        .timeline-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.7);
        }}

        .timeline-marker {{
            width: 12px;
            height: 12px;
            background: #8b9dff;
            border-radius: 50%;
            margin-right: 15px;
            flex-shrink: 0;
        }}

        .timeline-content {{
            flex: 1;
        }}

        .cluster {{
            background: #252525;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}

        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #8b9dff;
        }}

        .cluster-title {{
            font-size: 1.5em;
            color: #8b9dff;
        }}

        .cluster-badge {{
            background: #8b9dff;
            color: #1a1a1a;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}

        .cluster-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: #333;
            border-radius: 8px;
        }}

        .cluster-info-item {{
            display: flex;
            flex-direction: column;
        }}

        .cluster-info-label {{
            color: #aaa;
            font-size: 0.9em;
        }}

        .cluster-info-value {{
            font-weight: bold;
            color: #e0e0e0;
            margin-top: 3px;
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
        }}

        .image-card {{
            background: #333;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .image-card:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.8);
        }}

        .image-card img {{
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
        }}

        .image-info {{
            padding: 10px;
        }}

        .image-id {{
            font-size: 0.9em;
            color: #8b9dff;
            font-weight: bold;
        }}

        .image-time {{
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }}

        .image-confidence {{
            font-size: 0.8em;
            color: #4ade80;
            margin-top: 3px;
        }}

        .car-marker {{
            background: #4ade80 !important;
        }}

        .car-border {{
            border-left-color: #4ade80 !important;
        }}

        .car-cluster .cluster-title {{
            color: #4ade80;
        }}

        .car-cluster .cluster-header {{
            border-bottom-color: #4ade80;
        }}

        .car-cluster .cluster-badge {{
            background: #4ade80;
            color: #1a1a1a;
        }}

        .outliers {{
            border-left: 4px solid #f87171;
        }}

        .outliers .cluster-title {{
            color: #f87171;
        }}

        .outliers .cluster-header {{
            border-bottom-color: #f87171;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Detection Analysis Report</h1>
            <p>Clustered detections: Faces, Persons, and Vehicles (color-enhanced)</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="showTab('faces')">👤 Faces</button>
            <button class="tab" onclick="showTab('persons')">🚶 Persons</button>
            <button class="tab" onclick="showTab('vehicles')">🚗 Vehicles</button>
            <button class="tab" onclick="showTab('combined')">📊 Combined Timeline</button>
        </div>

        <!-- FACES TAB -->
        <div id="faces-tab" class="tab-content active">
            <div class="summary">
                <h2>Face Detection Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value">{total_faces}</div>
                        <div class="summary-label">Total Face Detections</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{unique_face_clusters}</div>
                        <div class="summary-label">Unique Face Clusters</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">{sum(c['count'] for c in face_clusters if c['cluster_id'] == -1)}</div>
                        <div class="summary-label">Outliers</div>
                    </div>
                </div>
            </div>

            <div class="timeline">
                <h2>Face Detection Timeline</h2>
"""

    # Generate faces timeline
    html += _generate_timeline_section(face_clusters, is_vehicle=False)

    html += """
            </div>

            <h2 style="margin-bottom: 20px; color: #8b9dff;">Face Clusters</h2>
"""

    # Generate face clusters
    html += _generate_cluster_sections(face_clusters, face_image_paths, cluster_type='face')

    html += """
        </div>

        <!-- PERSONS TAB -->
        <div id="persons-tab" class="tab-content">
            <div class="summary">
                <h2>Person Detection Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_persons) + """</div>
                        <div class="summary-label">Total Person Detections</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(unique_person_clusters) + """</div>
                        <div class="summary-label">Unique Person Clusters</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(sum(c['count'] for c in person_clusters if c['cluster_id'] == -1)) + """</div>
                        <div class="summary-label">Outliers</div>
                    </div>
                </div>
            </div>

            <div class="timeline">
                <h2>Person Detection Timeline</h2>
"""

    # Generate persons timeline
    html += _generate_timeline_section(person_clusters, is_vehicle=True)

    html += """
            </div>

            <h2 style="margin-bottom: 20px; color: #4ade80;">Person Clusters</h2>
"""

    # Generate person clusters
    html += _generate_cluster_sections(person_clusters, person_image_paths, cluster_type='person', is_vehicle_style=True)

    html += """
        </div>

        <!-- VEHICLES TAB -->
        <div id="vehicles-tab" class="tab-content">
            <div class="summary">
                <h2>Vehicle Detection Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_vehicles) + """</div>
                        <div class="summary-label">Total Vehicle Detections</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(unique_vehicle_clusters) + """</div>
                        <div class="summary-label">Unique Vehicle Clusters</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(sum(c['count'] for c in vehicle_clusters if c['cluster_id'] == -1)) + """</div>
                        <div class="summary-label">Outliers</div>
                    </div>
                </div>
            </div>

            <div class="timeline">
                <h2>Vehicle Detection Timeline</h2>
"""

    # Generate vehicles timeline
    html += _generate_timeline_section(vehicle_clusters[:100], is_vehicle=True)

    if len(vehicle_clusters) > 100:
        html += f"""
                <div style="text-align: center; padding: 20px; color: #aaa;">
                    <em>Showing first 100 of {len(vehicle_clusters)} clusters</em>
                </div>
"""

    html += """
            </div>

            <h2 style="margin-bottom: 20px; color: #4ade80;">Vehicle Clusters</h2>
"""

    # Generate vehicle clusters
    html += _generate_cluster_sections(vehicle_clusters, vehicle_image_paths, cluster_type='vehicle', is_vehicle_style=True)

    html += """
        </div>

        <!-- COMBINED TIMELINE TAB -->
        <div id="combined-tab" class="tab-content">
            <div class="summary">
                <h2>Combined Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_faces + total_persons + total_vehicles) + """</div>
                        <div class="summary-label">Total Events</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_faces) + """</div>
                        <div class="summary-label">Face Detections</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_persons) + """</div>
                        <div class="summary-label">Person Detections</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value">""" + str(total_vehicles) + """</div>
                        <div class="summary-label">Vehicle Detections</div>
                    </div>
                </div>
            </div>

            <div class="timeline">
                <h2>Combined Timeline (All Events)</h2>
"""

    # Generate combined timeline
    html += _generate_combined_timeline(face_clusters, person_clusters, vehicle_clusters)

    html += """
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');

            // Add active class to clicked tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html)


def _generate_timeline_section(clusters, is_vehicle=False):
    """Generate HTML for timeline section."""
    html = ""
    marker_class = "car-marker" if is_vehicle else ""
    border_class = "car-border" if is_vehicle else ""

    for cluster in clusters:
        cluster_name = f"Cluster {cluster['cluster_id']}" if cluster['cluster_id'] != -1 else "Outliers"
        html += f"""
                <div class="timeline-item {border_class}">
                    <div class="timeline-marker {marker_class}"></div>
                    <div class="timeline-content">
                        <strong>{cluster_name}</strong> - {cluster['count']} detection(s)<br>
                        <small>{cluster['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                    </div>
                </div>
"""
    return html


def _generate_cluster_sections(clusters, image_paths, cluster_type='face', is_vehicle_style=False):
    """Generate HTML for cluster detail sections."""
    html = ""

    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        cluster_class = "outliers" if cluster_id == -1 else ("car-cluster" if is_vehicle_style else "")
        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Outliers"

        html += f"""
            <div class="cluster {cluster_class}">
                <div class="cluster-header">
                    <h3 class="cluster-title">{cluster_name}</h3>
                    <span class="cluster-badge">{cluster['count']} detections</span>
                </div>

                <div class="cluster-info">
                    <div class="cluster-info-item">
                        <span class="cluster-info-label">First Seen</span>
                        <span class="cluster-info-value">{cluster['first_seen'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div class="cluster-info-item">
                        <span class="cluster-info-label">Last Seen</span>
                        <span class="cluster-info-value">{cluster['last_seen'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div class="cluster-info-item">
                        <span class="cluster-info-label">Avg Confidence</span>
                        <span class="cluster-info-value">{cluster['avg_confidence']:.3f}</span>
                    </div>
                    <div class="cluster-info-item">
                        <span class="cluster-info-label">Duration</span>
                        <span class="cluster-info-value">{(cluster['last_seen'] - cluster['first_seen']).total_seconds() / 3600:.1f} hours</span>
                    </div>
                </div>

                <div class="image-grid">
"""

        for detection in cluster['detections']:
            img_path = image_paths.get(detection['id'])
            if img_path:
                detection_label = f"{detection.get('class_name', cluster_type).title()} ID: {detection['id']}" if cluster_type == 'vehicle' else f"ID: {detection['id']}"
                html += f"""
                    <div class="image-card">
                        <img src="{img_path}" alt="Detection {detection['id']}" loading="lazy">
                        <div class="image-info">
                            <div class="image-id">{detection_label}</div>
                            <div class="image-time">{detection['timestamp'].strftime('%m/%d %H:%M:%S')}</div>
                            <div class="image-confidence">Conf: {detection['confidence']:.3f}</div>
                        </div>
                    </div>
"""

        html += """
                </div>
            </div>
"""
    return html


def _generate_combined_timeline(face_clusters, person_clusters, vehicle_clusters):
    """Generate combined timeline from all cluster types."""
    combined_events = []

    # Add face clusters
    for cluster in face_clusters:
        combined_events.append({
            'timestamp': cluster['first_seen'],
            'type': 'face',
            'label': f"Face Cluster {cluster['cluster_id']}" if cluster['cluster_id'] != -1 else "Face Outliers",
            'count': cluster['count']
        })

    # Add person clusters
    for cluster in person_clusters:
        combined_events.append({
            'timestamp': cluster['first_seen'],
            'type': 'person',
            'label': f"Person Cluster {cluster['cluster_id']}" if cluster['cluster_id'] != -1 else "Person Outliers",
            'count': cluster['count']
        })

    # Add vehicle clusters (limit to first 100)
    for cluster in vehicle_clusters[:100]:
        combined_events.append({
            'timestamp': cluster['first_seen'],
            'type': 'vehicle',
            'label': f"Vehicle Cluster {cluster['cluster_id']}" if cluster['cluster_id'] != -1 else "Vehicle Outliers",
            'count': cluster['count']
        })

    # Sort by timestamp
    combined_events.sort(key=lambda x: x['timestamp'])

    html = ""
    for event in combined_events:
        marker_class = "car-marker" if event['type'] in ['person', 'vehicle'] else ""
        border_class = "car-border" if event['type'] in ['person', 'vehicle'] else ""

        html += f"""
                <div class="timeline-item {border_class}">
                    <div class="timeline-marker {marker_class}"></div>
                    <div class="timeline-content">
                        <strong>{event['label']}</strong> - {event['count']} detection(s)<br>
                        <small>{event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
                    </div>
                </div>
"""

    if len(vehicle_clusters) > 100:
        html += f"""
                <div style="text-align: center; padding: 20px; color: #aaa;">
                    <em>Vehicle clusters limited to first 100 for timeline display</em>
                </div>
"""

    return html
