#!/usr/bin/env python3

from flask import Flask, render_template_string, jsonify, request, Response
import sqlite3
import os
import argparse
from datetime import datetime

app = Flask(__name__)

class FaceViewer:
    def __init__(self, face_db_path: str = "faces_2025-09-10.db"):
        """
        Initialize face viewer.
        
        Args:
            face_db_path: Path to faces database
        """
        self.face_db_path = face_db_path
        if not os.path.exists(face_db_path):
            raise FileNotFoundError(f"Face database not found: {face_db_path}")
    
    def _get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.face_db_path)
    
    def _convert_bbox_fields(self, rows):
        """Convert bbox fields from bytes to integers."""
        result = []
        for row in rows:
            converted_row = list(row)
            # Convert bbox fields (indices 3-6) from bytes to int if needed
            for i in range(3, 7):
                if isinstance(converted_row[i], bytes):
                    converted_row[i] = int.from_bytes(converted_row[i], byteorder='little')
                elif converted_row[i] is None:
                    converted_row[i] = 0
            result.append(tuple(converted_row))
        return result
    
    def get_face_groups(self, limit: int = 100, offset: int = 0):
        """Get face groups with statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT fg.id, fg.group_name, fg.representative_face_id, fg.face_count, 
                       fg.avg_confidence, fg.created_at
                FROM face_groups fg
                ORDER BY fg.face_count DESC, fg.avg_confidence DESC
                LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (limit, offset))
            return cursor.fetchall()
    
    def get_faces_in_group(self, group_id: int, limit: int = 50):
        """Get all faces in a specific group."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT f.id, f.detection_id, f.confidence, f.bbox_x, f.bbox_y, 
                       f.bbox_width, f.bbox_height, f.created_at
                FROM faces f
                WHERE f.similarity_group = ?
                ORDER BY f.confidence DESC
                LIMIT ?
            """
            
            cursor.execute(query, (group_id, limit))
            rows = cursor.fetchall()
            
            return self._convert_bbox_fields(rows)
    
    def get_ungrouped_faces(self, limit: int = 50, offset: int = 0):
        """Get faces that are not in any group."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT f.id, f.detection_id, f.confidence, f.bbox_x, f.bbox_y, 
                       f.bbox_width, f.bbox_height, f.created_at
                FROM faces f
                WHERE f.similarity_group IS NULL
                ORDER BY f.confidence DESC
                LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            
            return self._convert_bbox_fields(rows)
    
    def get_face_by_id(self, face_id: int):
        """Get a specific face by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, detection_id, face_crop, face_embeddings, similarity_group, 
                       confidence, bbox_x, bbox_y, bbox_width, bbox_height, created_at
                FROM faces WHERE id = ?
            """, (face_id,))
            
            return cursor.fetchone()
    
    def get_all_faces(self, limit: int = 200, offset: int = 0):
        """Get all faces ordered by group status (grouped first), then by confidence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT f.id, f.detection_id, f.confidence, f.bbox_x, f.bbox_y, 
                       f.bbox_width, f.bbox_height, f.created_at, f.similarity_group,
                       fg.group_name, fg.face_count
                FROM faces f
                LEFT JOIN face_groups fg ON f.similarity_group = fg.id
                ORDER BY 
                    CASE WHEN f.similarity_group IS NOT NULL THEN 0 ELSE 1 END,
                    f.similarity_group,
                    f.confidence DESC
                LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            
            return self._convert_bbox_fields(rows)
    
    def get_grouped_faces_with_details(self, limit: int = 100, offset: int = 0):
        """Get all grouped faces with group information."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT f.id, f.detection_id, f.confidence, f.bbox_x, f.bbox_y, 
                       f.bbox_width, f.bbox_height, f.created_at, f.similarity_group,
                       fg.group_name, fg.face_count, fg.representative_face_id
                FROM faces f
                JOIN face_groups fg ON f.similarity_group = fg.id
                ORDER BY fg.face_count DESC, f.confidence DESC
                LIMIT ? OFFSET ?
            """
            
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            
            return self._convert_bbox_fields(rows)

    def get_face_statistics(self):
        """Get statistics about faces and groups."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total faces
            cursor.execute('SELECT COUNT(*) FROM faces')
            total_faces = cursor.fetchone()[0]
            
            # Grouped faces
            cursor.execute('SELECT COUNT(*) FROM faces WHERE similarity_group IS NOT NULL')
            grouped_faces = cursor.fetchone()[0]
            
            # Face groups
            cursor.execute('SELECT COUNT(*) FROM face_groups')
            total_groups = cursor.fetchone()[0]
            
            # Average group size
            cursor.execute('SELECT AVG(face_count) FROM face_groups')
            avg_group_size = cursor.fetchone()[0] or 0
            
            # Largest group
            cursor.execute('SELECT MAX(face_count) FROM face_groups')
            max_group_size = cursor.fetchone()[0] or 0
            
            return {
                'total_faces': total_faces,
                'grouped_faces': grouped_faces,
                'ungrouped_faces': total_faces - grouped_faces,
                'total_groups': total_groups,
                'avg_group_size': round(avg_group_size, 1),
                'max_group_size': max_group_size
            }


# Global viewer instance (initialized in main function)
viewer = None

# HTML template for the face viewer
FACE_VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEye - Face Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            line-height: 1.6;
            transition: all 0.3s ease;
        }

        body.dark-mode {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: relative;
        }

        .dark-mode .header {
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .dark-mode-toggle {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }

        .dark-mode-toggle:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .header h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .header p {
            text-align: center;
            opacity: 0.9;
        }

        .stats-panel {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            transition: all 0.3s ease;
        }

        .dark-mode .stats-panel {
            background: #2d3748;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }

        .stat-label {
            color: #666;
            margin-top: 5px;
        }

        .nav-tabs {
            display: flex;
            background: white;
            border-radius: 10px 10px 0 0;
            overflow: hidden;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .dark-mode .nav-tabs {
            background: #2d3748;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .nav-tab {
            flex: 1;
            padding: 15px;
            background: #f8f9fa;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            color: #333;
        }

        .dark-mode .nav-tab {
            background: #4a5568;
            color: #e0e0e0;
        }

        .nav-tab.active {
            background: white;
            color: #667eea;
            font-weight: bold;
        }

        .dark-mode .nav-tab.active {
            background: #2d3748;
            color: #9f7aea;
        }

        .nav-tab:hover {
            background: #e9ecef;
        }

        .dark-mode .nav-tab:hover {
            background: #2d3748;
        }

        .content-panel {
            background: white;
            padding: 20px;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-height: 400px;
            transition: all 0.3s ease;
            width: 100%;
            overflow: hidden;
        }

        .dark-mode .content-panel {
            background: #2d3748;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .face-groups {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .face-group {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .dark-mode .face-group {
            background: #4a5568;
        }

        .face-group:hover {
            border-color: #667eea;
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .dark-mode .face-group:hover {
            border-color: #9f7aea;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }

        .group-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .group-name {
            font-weight: bold;
            color: #333;
        }

        .group-count {
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 15px;
            font-size: 0.8rem;
        }

        .group-faces {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
            gap: 5px;
            max-height: 200px;
            overflow: hidden;
        }

        .face-thumbnail {
            aspect-ratio: 1;
            border-radius: 5px;
            object-fit: cover;
            border: 2px solid #ddd;
        }

        .face-thumbnail:hover {
            border-color: #667eea;
        }


        .all-faces {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }


        .face-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
        }

        .dark-mode .face-item {
            background: #4a5568;
        }

        .face-item.grouped {
            border: 2px solid #667eea;
        }

        .dark-mode .face-item.grouped {
            border-color: #9f7aea;
        }

        .face-item.grouped:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            border-color: #4c63d2;
        }

        .dark-mode .face-item.grouped:hover {
            box-shadow: 0 5px 20px rgba(159, 122, 234, 0.3);
            border-color: #b794f6;
        }

        .group-badge {
            position: absolute;
            top: 5px;
            right: 5px;
            background: #667eea;
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: bold;
        }

        .dark-mode .group-badge {
            background: #9f7aea;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        .error {
            color: #dc3545;
            text-align: center;
            padding: 20px;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }

        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 10px;
            max-width: 80%;
            max-height: 80%;
            overflow-y: auto;
            transition: all 0.3s ease;
        }

        .dark-mode .modal-content {
            background: #2d3748;
            color: #e0e0e0;
        }

        .close {
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: #aaa;
        }

        .close:hover {
            color: #000;
        }

        .group-detail-faces {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .detailed-face {
            text-align: center;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .dark-mode .detailed-face {
            background: #4a5568;
        }

        .detailed-face img {
            width: 100%;
            max-width: 150px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .face-info {
            font-size: 0.9rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>AEye Face Viewer</h1>
            <p>Face Detection and Similarity Grouping</p>
        </div>
        <button class="dark-mode-toggle" onclick="toggleDarkMode()">
            🌙 Dark Mode
        </button>
    </div>

    <div class="container">
        <div class="stats-panel" id="statsPanel">
            <div class="stat-item">
                <div class="stat-value" id="totalFaces">-</div>
                <div class="stat-label">Total Faces</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="totalGroups">-</div>
                <div class="stat-label">Face Groups</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="groupedFaces">-</div>
                <div class="stat-label">Grouped Faces</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="ungroupedFaces">-</div>
                <div class="stat-label">Ungrouped Faces</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="avgGroupSize">-</div>
                <div class="stat-label">Avg Group Size</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="maxGroupSize">-</div>
                <div class="stat-label">Largest Group</div>
            </div>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('groups')">Face Groups</button>
            <button class="nav-tab" onclick="showTab('all')">All Faces</button>
        </div>

        <div class="content-panel">
            <div id="groupsTab" class="tab-content">
                <div class="loading" id="groupsLoading">Loading face groups...</div>
                <div class="face-groups" id="faceGroups" style="display: none;"></div>
            </div>

            <div id="allTab" class="tab-content" style="display: none;">
                <div class="loading" id="allLoading">Loading all faces...</div>
                <div class="all-faces" id="allFaces" style="display: none;"></div>
            </div>
        </div>
    </div>

    <!-- Group Detail Modal -->
    <div id="groupModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeGroupModal()">&times;</span>
            <h2 id="modalGroupName">Group Details</h2>
            <div class="group-detail-faces" id="modalGroupFaces"></div>
        </div>
    </div>

    <script>
        let currentTab = 'groups';
        let stats = {};

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            // Load dark mode preference
            if (localStorage.getItem('darkMode') === 'true') {
                document.body.classList.add('dark-mode');
                updateDarkModeToggle();
            }
            
            loadStats();
            loadFaceGroups();
        });

        function loadStats() {
            fetch('/api/face_stats')
                .then(response => response.json())
                .then(data => {
                    stats = data;
                    document.getElementById('totalFaces').textContent = data.total_faces;
                    document.getElementById('totalGroups').textContent = data.total_groups;
                    document.getElementById('groupedFaces').textContent = data.grouped_faces;
                    document.getElementById('ungroupedFaces').textContent = data.ungrouped_faces;
                    document.getElementById('avgGroupSize').textContent = data.avg_group_size;
                    document.getElementById('maxGroupSize').textContent = data.max_group_size;
                })
                .catch(error => {
                    console.error('Error loading stats:', error);
                });
        }

        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDark);
            updateDarkModeToggle();
        }

        function updateDarkModeToggle() {
            const toggle = document.querySelector('.dark-mode-toggle');
            const isDark = document.body.classList.contains('dark-mode');
            toggle.innerHTML = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
        }

        function loadFaceGroups() {
            const loading = document.getElementById('groupsLoading');
            const container = document.getElementById('faceGroups');
            
            loading.style.display = 'block';
            container.style.display = 'none';

            fetch('/api/face_groups')
                .then(response => response.json())
                .then(groups => {
                    loading.style.display = 'none';
                    container.style.display = 'grid';
                    container.innerHTML = '';

                    groups.forEach(group => {
                        const groupElement = createGroupElement(group);
                        container.appendChild(groupElement);
                    });
                })
                .catch(error => {
                    loading.innerHTML = `<div class="error">Error loading face groups: ${error.message}</div>`;
                    console.error('Error loading face groups:', error);
                });
        }

        function loadAllFaces() {
            const loading = document.getElementById('allLoading');
            const container = document.getElementById('allFaces');
            
            loading.style.display = 'block';
            container.style.display = 'none';

            fetch('/api/all_faces')
                .then(response => response.json())
                .then(faces => {
                    loading.style.display = 'none';
                    container.style.display = 'grid';
                    container.innerHTML = '';

                    faces.forEach(face => {
                        const faceElement = createAllFaceElement(face);
                        container.appendChild(faceElement);
                    });
                })
                .catch(error => {
                    loading.innerHTML = `<div class="error">Error loading all faces: ${error.message}</div>`;
                    console.error('Error loading all faces:', error);
                });
        }


        function createGroupElement(group) {
            const div = document.createElement('div');
            div.className = 'face-group';
            div.onclick = () => showGroupDetails(group[0]); // group[0] is the group ID from face_groups table

            const header = document.createElement('div');
            header.className = 'group-header';

            const name = document.createElement('div');
            name.className = 'group-name';
            name.textContent = group[1]; // group_name

            const count = document.createElement('div');
            count.className = 'group-count';
            count.textContent = `${group[3]} faces`; // face_count

            header.appendChild(name);
            header.appendChild(count);

            const facesDiv = document.createElement('div');
            facesDiv.className = 'group-faces';

            // Load representative face
            const repFaceId = group[2]; // representative_face_id
            if (repFaceId) {
                const img = document.createElement('img');
                img.className = 'face-thumbnail';
                img.src = `/api/face/${repFaceId}/image`;
                img.onerror = () => img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjZjhmOWZhIi8+Cjx0ZXh0IHg9IjUwIiB5PSI1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OTkiIGZvbnQtc2l6ZT0iMTIiPkZhY2U8L3RleHQ+Cjwvc3ZnPg==';
                facesDiv.appendChild(img);
            }

            div.appendChild(header);
            div.appendChild(facesDiv);

            return div;
        }


        function createAllFaceElement(face) {
            const div = document.createElement('div');
            div.className = face[8] !== null ? 'face-item grouped' : 'face-item'; // similarity_group

            // Add click handler for grouped faces
            if (face[8] !== null) {
                div.style.cursor = 'pointer';
                div.onclick = () => showGroupDetailsBySimilarityGroup(face[8]); // similarity_group id
                div.title = `Click to view all faces in ${face[9] || `Group ${face[8]}`}`;
            }

            const img = document.createElement('img');
            img.className = 'face-thumbnail';
            img.src = `/api/face/${face[0]}/image`; // face id
            img.style.width = '100px';
            img.style.height = '100px';
            img.onerror = () => img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjZjhmOWZhIi8+Cjx0ZXh0IHg9IjUwIiB5PSI1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OTkiIGZvbnQtc2l6ZT0iMTIiPkZhY2U8L3RleHQ+Cjwvc3ZnPg==';

            // Add group badge if grouped
            if (face[8] !== null) {
                const badge = document.createElement('div');
                badge.className = 'group-badge';
                badge.textContent = face[9] || `G${face[8]}`; // group_name or fallback
                div.appendChild(badge);
            }

            const info = document.createElement('div');
            info.className = 'face-info';
            const groupInfo = face[8] !== null ? `<div>Group: ${face[9] || face[8]} (${face[10]} faces)</div>` : '<div>Ungrouped</div>';
            info.innerHTML = `
                <div>Confidence: ${(face[2] * 100).toFixed(1)}%</div>
                <div>Detection: ${face[1]}</div>
                ${groupInfo}
            `;

            div.appendChild(img);
            div.appendChild(info);

            return div;
        }

        function showTab(tab) {
            // Update tab buttons
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tab}')"]`).classList.add('active');

            // Show/hide tab content
            document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
            document.getElementById(`${tab}Tab`).style.display = 'block';

            currentTab = tab;

            // Load content if needed
            if (tab === 'all' && document.getElementById('allFaces').children.length === 0) {
                loadAllFaces();
            }
        }

        function showGroupDetailsBySimilarityGroup(similarityGroupId) {
            const modal = document.getElementById('groupModal');
            const modalGroupName = document.getElementById('modalGroupName');
            const modalGroupFaces = document.getElementById('modalGroupFaces');

            // Set title based on similarity group
            modalGroupName.textContent = `Group ${similarityGroupId} Details`;

            modalGroupFaces.innerHTML = '<div class="loading">Loading group faces...</div>';
            modal.style.display = 'block';

            // Directly call the faces API using similarity group
            fetch(`/api/similarity_group/${similarityGroupId}/faces`)
                .then(response => response.json())
                .then(faces => {
                    modalGroupFaces.innerHTML = '';
                    
                    if (faces.length === 0) {
                        modalGroupFaces.innerHTML = '<div class="error">No faces found in this group.</div>';
                        return;
                    }

                    modalGroupName.textContent = `Group ${similarityGroupId} (${faces.length} faces)`;

                    faces.forEach(face => {
                        const faceDiv = document.createElement('div');
                        faceDiv.className = 'detailed-face';

                        const img = document.createElement('img');
                        img.src = `/api/face/${face[0]}/image`;
                        img.onerror = () => img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjZjhmOWZhIi8+Cjx0ZXh0IHg9IjUwIiB5PSI1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OTkiIGZvbnQtc2l6ZT0iMTIiPkZhY2U8L3RleHQ+Cjwvc3ZnPg==';

                        const info = document.createElement('div');
                        info.className = 'face-info';
                        info.innerHTML = `
                            <div><strong>Face ID:</strong> ${face[0]}</div>
                            <div><strong>Detection:</strong> ${face[1]}</div>
                            <div><strong>Confidence:</strong> ${(face[2] * 100).toFixed(1)}%</div>
                            <div><strong>BBox:</strong> ${face[3]}, ${face[4]}, ${face[5]}×${face[6]}</div>
                            <div><strong>Created:</strong> ${new Date(face[7]).toLocaleString()}</div>
                        `;

                        faceDiv.appendChild(img);
                        faceDiv.appendChild(info);
                        modalGroupFaces.appendChild(faceDiv);
                    });
                })
                .catch(error => {
                    modalGroupFaces.innerHTML = `<div class="error">Error loading group faces: ${error.message}</div>`;
                    console.error('Error loading group faces:', error);
                });
        }

        function showGroupDetails(groupId) {
            const modal = document.getElementById('groupModal');
            const modalGroupName = document.getElementById('modalGroupName');
            const modalGroupFaces = document.getElementById('modalGroupFaces');

            // First get the group information to display proper name
            fetch(`/api/face_groups`)
                .then(response => response.json())
                .then(groups => {
                    const group = groups.find(g => g[0] === groupId);
                    const groupName = group ? group[1] : `Group ${groupId}`;
                    const faceCount = group ? group[3] : 'Unknown';
                    modalGroupName.textContent = `${groupName} (${faceCount} faces)`;
                })
                .catch(error => {
                    modalGroupName.textContent = `Group ${groupId} Details`;
                    console.error('Error loading group info:', error);
                });

            modalGroupFaces.innerHTML = '<div class="loading">Loading group faces...</div>';
            modal.style.display = 'block';

            fetch(`/api/face_group/${groupId}/faces`)
                .then(response => response.json())
                .then(faces => {
                    modalGroupFaces.innerHTML = '';
                    
                    if (faces.length === 0) {
                        modalGroupFaces.innerHTML = '<div class="error">No faces found in this group.</div>';
                        return;
                    }

                    faces.forEach(face => {
                        const faceDiv = document.createElement('div');
                        faceDiv.className = 'detailed-face';

                        const img = document.createElement('img');
                        img.src = `/api/face/${face[0]}/image`;
                        img.onerror = () => img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiBmaWxsPSIjZjhmOWZhIi8+Cjx0ZXh0IHg9IjUwIiB5PSI1NSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZmlsbD0iIzk5OTkiIGZvbnQtc2l6ZT0iMTIiPkZhY2U8L3RleHQ+Cjwvc3ZnPg==';

                        const info = document.createElement('div');
                        info.className = 'face-info';
                        info.innerHTML = `
                            <div><strong>Face ID:</strong> ${face[0]}</div>
                            <div><strong>Detection:</strong> ${face[1]}</div>
                            <div><strong>Confidence:</strong> ${(face[2] * 100).toFixed(1)}%</div>
                            <div><strong>BBox:</strong> ${face[3]}, ${face[4]}, ${face[5]}×${face[6]}</div>
                            <div><strong>Created:</strong> ${new Date(face[7]).toLocaleString()}</div>
                        `;

                        faceDiv.appendChild(img);
                        faceDiv.appendChild(info);
                        modalGroupFaces.appendChild(faceDiv);
                    });
                })
                .catch(error => {
                    modalGroupFaces.innerHTML = `<div class="error">Error loading group faces: ${error.message}</div>`;
                    console.error('Error loading group faces:', error);
                });
        }

        function closeGroupModal() {
            document.getElementById('groupModal').style.display = 'none';
        }

        // Close modal when clicking outside of it
        window.onclick = function(event) {
            const modal = document.getElementById('groupModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main face viewer page."""
    return render_template_string(FACE_VIEWER_TEMPLATE)

@app.route('/api/face_stats')
def api_face_stats():
    """API endpoint to get face statistics."""
    stats = viewer.get_face_statistics()
    return jsonify(stats)

@app.route('/api/face_groups')
def api_face_groups():
    """API endpoint to get face groups."""
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    groups = viewer.get_face_groups(limit=limit, offset=offset)
    return jsonify(groups)

@app.route('/api/face_group/<int:group_id>/faces')
def api_group_faces(group_id):
    """API endpoint to get faces in a specific group."""
    limit = int(request.args.get('limit', 50))
    
    # First get the similarity_group value for this face_groups table ID
    with viewer._get_connection() as conn:
        cursor = conn.cursor()
        # Find the similarity_group value by looking at a face that belongs to this group
        cursor.execute("""
            SELECT f.similarity_group 
            FROM faces f 
            JOIN face_groups fg ON f.id = fg.representative_face_id 
            WHERE fg.id = ?
            LIMIT 1
        """, (group_id,))
        
        result = cursor.fetchone()
        if not result:
            return jsonify([])
        
        similarity_group = result[0]
    
    # Now get all faces with this similarity_group
    faces = viewer.get_faces_in_group(similarity_group, limit=limit)
    return jsonify(faces)

@app.route('/api/similarity_group/<int:similarity_group_id>/faces')
def api_similarity_group_faces(similarity_group_id):
    """API endpoint to get faces in a specific similarity group."""
    limit = int(request.args.get('limit', 50))
    
    faces = viewer.get_faces_in_group(similarity_group_id, limit=limit)
    return jsonify(faces)

@app.route('/api/all_faces')
def api_all_faces():
    """API endpoint to get all faces ordered by group status."""
    limit = int(request.args.get('limit', 200))
    offset = int(request.args.get('offset', 0))
    
    faces = viewer.get_all_faces(limit=limit, offset=offset)
    return jsonify(faces)

@app.route('/api/ungrouped_faces')
def api_ungrouped_faces():
    """API endpoint to get ungrouped faces."""
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    faces = viewer.get_ungrouped_faces(limit=limit, offset=offset)
    return jsonify(faces)

@app.route('/api/face/<int:face_id>/image')
def api_face_image(face_id):
    """Get face crop image."""
    face = viewer.get_face_by_id(face_id)
    if not face:
        return "Face not found", 404
    
    face_crop_bytes = face[2]  # face_crop
    return Response(face_crop_bytes, mimetype='image/jpeg')


def main():
    parser = argparse.ArgumentParser(description='Web viewer for grouped faces')
    parser.add_argument('--face-db', 
                       default='faces_2025-09-10.db',
                       help='Path to faces database (default: faces_2025-09-10.db)')
    parser.add_argument('--host', default='localhost',
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=3002,
                       help='Port to bind to (default: 3001)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.face_db):
        print(f"Error: Face database not found: {args.face_db}")
        print("Please run facedetect.py first to create the face database.")
        return 1
    
    # Initialize viewer
    global viewer
    try:
        viewer = FaceViewer(face_db_path=args.face_db)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Starting face viewer on http://{args.host}:{args.port}")
    print(f"Using face database: {args.face_db}")
    
    # Show statistics
    stats = viewer.get_face_statistics()
    print(f"\n=== Face Database Statistics ===")
    print(f"Total faces: {stats['total_faces']}")
    print(f"Face groups: {stats['total_groups']}")
    print(f"Grouped faces: {stats['grouped_faces']}")
    print(f"Ungrouped faces: {stats['ungrouped_faces']}")
    print(f"Average group size: {stats['avg_group_size']}")
    print(f"Largest group: {stats['max_group_size']}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    exit(main())