# AEye - Video Object Detection

AEye: an AI-powered object detection system with day-based database sharding and web interface.

## Features

- **Day-based Database Sharding**: Scalable storage with separate database files per day
- **Multiple YOLO Models**: Support for different YOLO model sizes from nano to extra-large
- **Web Interface**: Browse detections by date with dynamic date switching
- **Video Recording**: Organized video storage by date/hour/minute
- **Real-time Processing**: Support for video files, webcams, and RTSP streams

## Quick Start

### 1. Download YOLO Models

```bash
# Download recommended model (YOLOv8 Small - good balance of speed/accuracy)
./download_models.sh

# Or download specific model sizes
./download_models.sh small    # YOLOv8 Small (21.5MB)
./download_models.sh medium   # YOLOv8 Medium (49.7MB)
./download_models.sh large    # YOLOv8 Large (83.7MB)

# See all available models
python3 processor.py --list-models
```

### 2. Process Video/Stream

```bash
# Video file with default model
python3 processor.py video.mp4 --confidence 0.5

# Video file with specific model
python3 processor.py video.mp4 --model yolov8s.pt --confidence 0.5

# Webcam (auto-detected)
python3 processor.py 0 --confidence 0.5

# RTSP stream with recording
python3 processor.py rtsp://192.168.1.100:554/stream --confidence 0.5 --record
```

### 3. View Results

```bash
# View current date
python3 web_viewer.py --port 3000

# View specific date
python3 web_viewer.py --date 2025-09-09 --port 3000

# From other PC on network
python3 web_viewer.py --host 0.0.0.0 --port 3000
```

Visit http://localhost:3000 to browse detections.


## Docker 

### Build and Setup

```bash
# Build the image
./docker/docker-run.sh build
```

### Run Both Services (Recommended)

```bash
# Run processor + web viewer together in background
./docker/docker-run.sh both 0 --record --confidence 0.5

# Run with RTSP stream
./docker/docker-run.sh both rtsp://camera --confidence 0.3

# Check status and manage
./docker/docker-run.sh status
./docker/docker-run.sh logs processor
./docker/docker-run.sh stop all
```

### Individual Services

```bash
# Web viewer only
./docker/docker-run.sh viewer

# Processor only
./docker/docker-run.sh processor video.mp4 --confidence 0.5

# Debug shell
./docker/docker-run.sh shell
```

## Raspberry Pi Optimizations

### Setup and Status

```bash
# Check Pi system status and get recommendations
./docker/docker-pi.sh status

# Build Pi-optimized image
./docker/docker-pi.sh build
```

### Run Both Services (Recommended for Pi)

```bash
# Run processor + viewer with Pi optimizations
./docker/docker-pi.sh both 0 --record --confidence 0.4

# Run with RTSP and performance tuning
./docker/docker-pi.sh both rtsp://camera --frame-skip 3 --confidence 0.4 --resize-factor 0.5

# Monitor Pi performance
./docker/docker-pi.sh monitor
./docker/docker-pi.sh logs processor
./docker/docker-pi.sh stop all
```

### Individual Pi Services

```bash
# Processor only with Pi optimizations
./docker/docker-pi.sh processor rtsp://camera --frame-skip 3 --confidence 0.4 --resize-factor 0.5

# Web viewer only
./docker/docker-pi.sh viewer
```

### Pi Performance Tips

- Use `--frame-skip 2-4` for better performance
- Set `--confidence 0.4+` to reduce false positives  
- Use `--resize-factor 0.5-0.75` for faster processing
- Limit `--max-detections` to 5-10
- Monitor temperature with `./docker/docker-pi.sh status`



## YOLO Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6.2MB | Fastest | Basic | Real-time, low power |
| yolov8s.pt | 21.5MB | Fast | Good | **Recommended balance** |
| yolov8m.pt | 49.7MB | Medium | Better | Higher accuracy needs |
| yolov8l.pt | 83.7MB | Slow | High | Quality over speed |
| yolov8x.pt | 131.4MB | Slowest | Best | Maximum accuracy |

## Data Organization

```
data/
├── db/                    # Daily database files
│   ├── detections_2025-09-09.db
│   └── detections_2025-09-10.db
└── videos/               # Video recordings
    └── 2025-09-10/
        └── 08/
            ├── 30.mp4    # 08:30 recording
            └── 31.mp4    # 08:31 recording
```

## Command Line Options

**Processor (`processor.py`):**
- `--model`: YOLO model to use (default: yolov8n.pt)
- `--confidence`: Detection threshold 0.0-1.0 (default: 0.15)
- `--base-path`: Data storage path (default: data)
- `--record`: Record video files for streams
- `--show-live`: Display video during processing
- `--list-models`: Show available YOLO models

**Web Viewer (`web_viewer.py`):**
- `--base-path`: Data storage path (default: data)
- `--date`: Specific date to view (YYYY-MM-DD)
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to bind to (default: 3000)


#### ctronics camera:
rtsp://192.168.6.244:554/11