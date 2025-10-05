# AEye - Motion Detection & Object Recognition System

AEye: An AI-powered motion detection system with real-time object recognition, face detection, and interactive web interface.

![](docs/pics/demo.png)

## Features

- **Motion-Triggered Recording**: Smart video recording with pre/post motion buffers
- **Real-time Object Detection**: YOLO-powered object recognition with live thumbnails
- **Face Detection & Recognition**: InsightFace integration for face detection and embeddings
- **Interactive Web Dashboard**: Timeline-based interface with object thumbnails and classes
- **Date-based Organization**: Automatic file and database organization by date
- **Multi-Source Support**: Webcams, IP cameras, RTSP streams, and video files


## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for face detection)
- FFmpeg (for RTSP stream handling)

### Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install YOLO model (will auto-download on first run)
# YOLOv8n model will be downloaded to models/yolov8n.pt

# For face detection, InsightFace models will auto-download on first run
```

#### For insightface:

```
sudo apt update
sudo apt install python3.13-dev
```

### Create Required Directories

```bash
mkdir -p data/videos
mkdir -p data/images
mkdir -p data/db
mkdir -p data/faces-known
mkdir -p models
```

## Usage


### Quick start:

In 3 terminal windows, run:

```
# terminal 1:
./ingest_ffmpeg rtsp://admin:passwd@192.168.6.254:554/Preview_01_sub data/videos

# terminal 2:
python process.py --monitor

# terminal 3:
python web_viewer.py 
```

### Details:

#### ingest video segments - run 1st

Will record 30s videos to be processed.

For maximum stability with RTSP streams, use the C-based FFmpeg ingest:

```bash
# Compile the C binary (one-time setup)
make

# Or compile automatically on first run
python ingest.py --compile --video-source rtsp://192.168.1.100:554/stream

# Run stable ingest
python ingest.py --video-source rtsp://192.168.1.100:554/stream

# Or run the C binary directly
./ingest_ffmpeg rtsp://192.168.1.100:554/stream data/videos
```

#### process video segments - run 2nd

Best for: Batch or continuous processing of recorded videos

```bash
# Continuous monitoring mode (recommended for production)
python process.py --monitor

# Custom check interval (60 seconds between checks)
python process.py --monitor --check-interval 60

# One-time batch processing
python process.py

# Process a single video file
python process.py --video-file data/videos/2024_03_15/20240315_143022.mp4

# Reprocess all videos (ignore processed flag in database)
python process.py --reprocess

# Custom file age threshold (only process files older than 45 seconds)
python process.py --monitor --file-age-threshold 45
```

**Key Parameters:**
- `--monitor`: Continuously monitor for new video files
- `--check-interval`: Seconds between checks in monitor mode (default: 30)
- `--file-age-threshold`: Minimum file age in seconds before processing (default: 60)
- `--keep-empty`: Keep videos with no detections (default: delete them to save space)
- `--videos-dir`: Directory containing video files (default: data/videos)
- `--db-path`: SQLite database path (default: data/db/detections.db)
- `--reprocess`: Reprocess already-processed videos
- `--no-gpu`: Disable GPU usage

**Auto-Delete Empty Videos:**
By default, videos with no face or object detections are automatically deleted to save disk space. The motion event is also removed from the database. To keep all videos regardless of detections, use `--keep-empty`.


#### launch Web Dashboard - run 3rd

```bash
# Local access
python3 web_viewer.py --port 3000

# Network access (from other devices)
python3 web_viewer.py --host 0.0.0.0 --port 3000

# View specific date
python3 web_viewer.py --date 2025-09-15 --port 3000
```

#### Data Reports

At any point in time, you can run a timeline of events that also clusters similar events together.

To generate a html report clustering detections of people, faces and vehicles, run:


```bash

# Default (enhanced settings)
python scripts/timeline.py

# Even stricter clustering (more clusters)
python scripts/timeline.py --vehicle-eps 0.06 --color-weight 0.6

# Group more aggressively (fewer clusters, must be VERY similar)
python scripts/timeline.py --vehicle-eps 0.05 --vehicle-min-samples 2

# Focus mostly on color (70% color, 30% appearance)
python scripts/timeline.py --color-weight 0.7 --vehicle-eps 0.08

# Focus mostly on appearance (30% color, 70% appearance)
python scripts/timeline.py --color-weight 0.3 --vehicle-eps 0.10

# Extract ReID features
python scripts/extract_reid_features.py

# Generate report with ReID
python scripts/timeline.py --use-reid
```

How to Tune:

If cars in same cluster are still too different:
- ⬇️ Decrease --vehicle-eps (try 0.06 or 0.05)
- ⬆️ Increase --color-weight (try 0.6 or 0.7)
- ⬆️ Increase --vehicle-min-samples (try 2 or 3)

If you have too many tiny clusters:
- ⬆️ Increase --vehicle-eps (try 0.10 or 0.12)
- ⬇️ Decrease --color-weight (try 0.3 or 0.4)


## Face Recognition Setup

To enable face recognition, add known faces to the system:

1. Create person directories in `data/faces-known/`:
```bash
mkdir -p data/faces-known/John
mkdir -p data/faces-known/Jane
```

2. Add face images (JPG, PNG) for each person:
```bash
# Add multiple photos of the same person for better recognition
data/faces-known/John/photo1.jpg
data/faces-known/John/photo2.jpg
data/faces-known/Jane/photo1.jpg
```

3. The system will automatically load and use these faces on startup

## Database Schema

AEyeMon uses SQLite with the following main tables:

### motion_events
Stores video recording events
- `id`: Primary key
- `start_time`, `end_time`: Event timestamps
- `video_file`: Path to recorded video
- `duration_seconds`: Event duration
- `processed`: Whether video has been processed
- `face_count`, `object_count`: Detection counts

### face_detections
Stores detected faces with embeddings
- `id`: Primary key
- `motion_event_id`: Foreign key to motion_events
- `frame_timestamp`: When face was detected
- `face_crop`: Face image (BLOB)
- `face_embedding`: Face embedding vector (BLOB)
- `confidence`: Detection confidence
- `known_person`: Recognized person name (if matched)
- `recognition_confidence`: Recognition confidence
- `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`: Bounding box

### object_detections
Stores detected objects
- `id`: Primary key
- `motion_event_id`: Foreign key to motion_events
- `frame_timestamp`: When object was detected
- `class_name`: Object class (person, car, dog, etc.)
- `confidence`: Detection confidence
- `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`: Bounding box
- `object_crop`: Object image (BLOB)
- `track_id`: Foreign key to object_tracks

### object_tracks
Groups related object detections across frames
- `id`: Primary key
- `motion_event_id`: Foreign key to motion_events
- `class_name`: Object class
- `track_start_time`, `track_end_time`: Track duration
- `detection_count`: Number of detections in track
- `avg_confidence`: Average detection confidence
- `first_bbox_*`, `last_bbox_*`: First and last bounding boxes
- `representative_crop`: Best crop image (BLOB)


## Configuration

### Object Detection Categories

Edit `lib/object_detector.py` allowed_categories:

```python
self.allowed_categories = {
    'person', 'bicycle', 'car', 'truck', 'bus', 'motorcycle',
    'bird', 'cat', 'dog', 'backpack', 'handbag', 'suitcase'
}
```

### Face Recognition Threshold

Edit `lib/face_detector.py`:

```python
self.recognition_threshold = 0.4  # Lower = stricter matching
```

## Performance Optimization

### GPU Acceleration
- Ensure CUDA is installed for GPU support
- Face detection benefits significantly from GPU
- YOLO automatically uses GPU if available

### Frame Processing
- Process every Nth frame to reduce CPU load
- Default: Every 15th frame in process.py
- Adjust in `process_video_file()`: `if frame_count % 15 == 0:`

### Video Resolution
- Lower resolution reduces processing time
- Configure in video capture settings

### Detection Confidence
- Higher confidence thresholds reduce false positives
- Adjust in detector classes (default: 0.5 for objects, 0.7 for faces)

## Troubleshooting

### RTSP Stream Timeout Issues

If you're experiencing constant stream timeouts with Python-based ingest:

```
[ WARN:0@3015.378] global cap_ffmpeg_impl.hpp:453 _opencv_ffmpeg_interrupt_callback Stream timeout triggered
```

**Solution: Use C-based stable ingest:**

```bash
# Compile the stable ingest
make

# Run stable ingest
python ingest_stable.py --video-source rtsp://192.168.1.100:554/stream
```

The C-based ingest solves these issues with:
- Better timeout configuration (10s connection, 10s read)
- Automatic reconnection with backoff
- Proper interrupt handling
- Direct FFmpeg library usage (no OpenCV wrapper overhead)

### RTSP Connection Issues
```bash
# Test RTSP stream with FFmpeg
ffmpeg -i rtsp://192.168.1.100:554/stream -frames:v 1 test.jpg

# Test with stable ingest (compile first)
make
./ingest_ffmpeg rtsp://192.168.1.100:554/stream data/videos

# Common issues:
# - Firewall blocking RTSP port (554)
# - Incorrect credentials in RTSP URL
# - Camera not supporting TCP transport
# - Network instability (use C-based ingest for auto-reconnect)
```

### C Binary Compilation Issues

If `make` fails:

```bash
# Check if FFmpeg libraries are installed
pkg-config --exists libavformat libavcodec libavutil libswscale && echo "FFmpeg libs found" || echo "FFmpeg libs missing"

# Install FFmpeg development libraries
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

# Fedora/CentOS
sudo dnf install ffmpeg-devel

# Manual compilation (if make fails)
gcc -o ingest_ffmpeg ingest_ffmpeg.c \
    $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale) -O3
```

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Face Detection Not Working
```bash
# Ensure InsightFace is installed correctly
pip install insightface onnxruntime-gpu

# Check model download
# Models should be in ~/.insightface/models/
```

### Database Locked Errors
- Only one process should write to database at a time
- Use `--monitor` mode in process.py (handles this automatically)
- If error persists, check for zombie processes

## Monitoring and Logs

Logs are written to:
- `data/aeye.log` - Main application (main.py)
- `data/ingest.log` - Video ingestion (ingest.py)
- `data/process.log` - Video processing (process.py)

View logs in real-time:
```bash
tail -f data/aeye.log
tail -f data/ingest.log
tail -f data/process.log
```

## Web Viewer

AEyeMon includes a web-based viewer to browse and analyze recorded events.

### Starting the Web Viewer

```bash
# Start web viewer on default port (3000)
python web_viewer.py

# Custom port and host
python web_viewer.py --host 0.0.0.0 --port 5000

# Debug mode
python web_viewer.py --debug
```

Then open your browser to: `http://localhost:3000`

### Web Viewer Features

**Dashboard Overview:**
- Total motion events, face detections, object detections, and object tracks
- Processed vs unprocessed events count
- Peak activity hour

**Interactive Timeline:**
- 24-hour activity timeline showing events per hour
- Click any hour to see detailed events
- Color-coded activity levels (green = activity, orange = high activity)

**Event Details:**
- Video playback
- Object tracks with thumbnails and statistics
- Face detections with recognition results
- Detection confidence scores
- Duration and timestamp information

**Dark/Light Mode:**
- Toggle between dark and light themes
- Preference saved in browser

**Date Navigation:**
- Browse events by date
- Automatic detection of available dates

## API and Database Queries

Query the database directly:

```bash
sqlite3 data/db/detections.db
```

Example queries:

```sql
-- Count motion events by date
SELECT DATE(start_time) as date, COUNT(*) as events
FROM motion_events
GROUP BY DATE(start_time)
ORDER BY date DESC;

-- Find all videos with persons detected
SELECT video_file, object_count, face_count
FROM motion_events
WHERE object_count > 0
ORDER BY start_time DESC;

-- Get all recognized faces
SELECT known_person, COUNT(*) as detections
FROM face_detections
WHERE known_person IS NOT NULL
GROUP BY known_person;

-- Find recent object detections
SELECT m.video_file, o.class_name, o.confidence, o.frame_timestamp
FROM object_detections o
JOIN motion_events m ON o.motion_event_id = m.id
ORDER BY o.frame_timestamp DESC
LIMIT 20;
```

## License

TBD

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues and questions, please open an issue on GitHub.
