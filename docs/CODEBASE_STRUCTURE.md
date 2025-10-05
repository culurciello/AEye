# AEye Codebase Structure

This document explains the organization of the AEye codebase.

## Project Structure

```
AEye/
├── scripts/                    # User-facing scripts
│   ├── timeline.py            # Main clustering report generator (refactored)
│   ├── timeline_old.py        # Old monolithic version (backup)
│   └── extract_reid_features.py  # Extract ReID embeddings
│
├── lib/                        # Core library modules
│   ├── data_loader.py         # Database loading functions
│   ├── clustering_engine.py   # Clustering algorithms and features
│   ├── report_generator.py    # HTML report generation
│   ├── reid_extractor.py      # ReID feature extraction (TorchReID)
│   ├── database.py            # Database management
│   ├── face_detector.py       # Face detection
│   ├── object_detector.py     # Object detection and tracking
│   └── video_processor.py     # Video processing
│
├── docs/                       # Documentation
│   ├── REID_CLUSTERING.md     # ReID clustering guide
│   ├── REID_QUICKSTART.md     # Quick start for ReID
│   ├── REID_MODELS.md         # Model selection guide
│   └── CODEBASE_STRUCTURE.md  # This file
│
├── data/                       # Data directory
│   ├── db/                     # Databases
│   └── models/                 # Model weights
│
└── report/                     # Generated reports
    ├── detection_report.html   # HTML report
    └── timeline_images/        # Detection image crops
```

## Module Organization

### Scripts (`scripts/`)

**User-facing entry points:**

- **`timeline.py`** - Main script for generating clustering reports
  - Orchestrates data loading, clustering, and report generation
  - Handles command-line arguments
  - Coordinates between modules
  - Clean, readable ~300 lines (was 1300+ lines)

- **`extract_reid_features.py`** - Extract ReID embeddings from detections
  - Batch processing of existing detections
  - Stores ReID embeddings in database

### Core Library (`lib/`)

**Modular components with single responsibilities:**

#### Data Loading (`data_loader.py`)
- `migrate_database()` - Database schema migrations
- `load_face_detections()` - Load face data with embeddings
- `load_object_detections()` - Load person/vehicle data
- Handles both CLIP and ReID embeddings

#### Clustering Engine (`clustering_engine.py`)
- `extract_color_histogram()` - Extract color features from images
- `combine_features()` - Combine embeddings with color features
- `cluster_embeddings()` - DBSCAN clustering with cosine similarity
- `group_detections_by_cluster()` - Organize detections into clusters
- `extract_vehicle_color_features()` - Batch color extraction

#### Report Generator (`report_generator.py`)
- `generate_html_report()` - Main HTML generation function
- `_generate_timeline_section()` - Timeline HTML
- `_generate_cluster_sections()` - Cluster details HTML
- `_generate_combined_timeline()` - Combined timeline HTML
- Dark mode, responsive design
- ~600 lines of focused HTML generation

#### ReID Extractor (`reid_extractor.py`)
- `ReIDExtractor` class - TorchReID wrapper
- `init_model()` - Load pretrained models
- `extract_features()` - Extract embeddings from images
- Fallback to ResNet50 if TorchReID unavailable

### Other Core Modules

- **`database.py`** - Database operations (store detections, tracks)
- **`face_detector.py`** - Face detection using InsightFace
- **`object_detector.py`** - Object detection using YOLO + CLIP
- **`video_processor.py`** - Video processing and motion detection

## Data Flow

### Report Generation Flow

```
┌─────────────────┐
│  timeline.py    │ ← Entry point
└────────┬────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌──────────────────┐
│ data_loader.py  │   │ clustering_      │
│                 │   │ engine.py        │
│ - Load faces    │   │ - Extract colors │
│ - Load persons  │   │ - Cluster        │
│ - Load vehicles │   │ - Group          │
└────────┬────────┘   └─────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ report_         │
           │ generator.py    │
           │                 │
           │ - Generate HTML │
           │ - Dark mode CSS │
           │ - Tabs & charts │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ detection_      │
           │ report.html     │
           └─────────────────┘
```

### ReID Feature Extraction Flow

```
┌─────────────────────────┐
│ extract_reid_           │
│ features.py             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ reid_extractor.py       │
│ - Load TorchReID model  │
│ - Extract embeddings    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Database                │
│ (reid_embedding column) │
└─────────────────────────┘
```

## Benefits of New Structure

### Before (Monolithic)

**`timeline.py`** - 1300+ lines
- Data loading
- Color feature extraction
- Clustering algorithms
- HTML generation
- CSS styles
- JavaScript
- Argument parsing
- All mixed together

**Problems:**
- ❌ Hard to navigate
- ❌ Difficult to test individual components
- ❌ Can't reuse code easily
- ❌ Changes affect everything

### After (Modular)

**`timeline.py`** - ~300 lines (orchestration only)
- Clean, readable main flow
- Imports from focused modules

**`data_loader.py`** - ~150 lines
- Just database operations
- Reusable for other scripts

**`clustering_engine.py`** - ~150 lines
- Pure clustering logic
- Easy to test and improve

**`report_generator.py`** - ~600 lines
- Just HTML generation
- Easy to customize styling

**Benefits:**
- ✅ Each file has single responsibility
- ✅ Easy to find and modify code
- ✅ Reusable components
- ✅ Better for testing
- ✅ Easier to onboard new developers

## Usage Examples

### Using Individual Modules

```python
# Load data
from lib.data_loader import load_face_detections
faces, embeddings = load_face_detections('data/db/detections.db')

# Cluster
from lib.clustering_engine import cluster_embeddings
labels = cluster_embeddings(embeddings, eps=0.4)

# Generate report
from lib.report_generator import generate_html_report
generate_html_report(clusters, paths, 'report.html')
```

### Extending Functionality

Want to add a new clustering algorithm?
→ Modify `clustering_engine.py` only

Want to change HTML style?
→ Modify `report_generator.py` only

Want to support new database format?
→ Modify `data_loader.py` only

## Testing Strategy

Each module can be tested independently:

```python
# Test clustering
from lib.clustering_engine import cluster_embeddings
import numpy as np

embeddings = np.random.rand(100, 512)
labels = cluster_embeddings(embeddings, eps=0.4)
assert len(labels) == 100

# Test data loading
from lib.data_loader import load_face_detections
faces, embs = load_face_detections('test.db')
assert len(faces) == len(embs)
```

## Future Improvements

With this modular structure, we can easily:

- [ ] Add unit tests for each module
- [ ] Support multiple report formats (PDF, JSON, etc.)
- [ ] Add different clustering algorithms
- [ ] Create a web API using these modules
- [ ] Build a GUI that uses the same core logic
- [ ] Parallel processing for large datasets

## Migration Guide

If you have custom code using old `timeline.py`:

**Old:**
```python
# Everything was in timeline.py
from scripts.timeline import load_face_detections, cluster_embeddings
```

**New:**
```python
# Import from specific modules
from lib.data_loader import load_face_detections
from lib.clustering_engine import cluster_embeddings
```

The old `timeline_old.py` is kept as a backup for reference.

## Questions?

- **Which file handles X?** See module descriptions above
- **How to add feature Y?** Find the relevant module, modify it
- **Can I use modules separately?** Yes! They're independent
- **What if I break something?** `timeline_old.py` has the original code

## Contributing

When adding new features:

1. **Identify the right module** - Data loading? Clustering? HTML?
2. **Keep functions focused** - One responsibility per function
3. **Update this doc** - Explain new functionality
4. **Test independently** - Each module should work standalone

