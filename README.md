# My2ndEye - Video Surveillance & AI Detection System

A comprehensive cross-platform security camera system that combines high-performance video capture with advanced AI object detection and analysis. The system captures video streams with H.264 compression, processes them with YOLO models, and provides a complete web interface for viewing and analyzing detections.

## 🎯 System Overview

My2ndEye consists of two main components:
1. **Video Capture System** - Cross-platform C program for H.264 webcam capture
2. **AI Detection Pipeline** - Python-based object detection, captioning, and web viewer

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Video Capture  │───▶│  Object Detection │───▶│    Database     │
│   (C/ffmpeg)    │    │    (YOLOv8)      │    │   (SQLite)      │
│     H.264       │    │     Python       │    │   + Web UI      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ✨ Key Features

### Video Capture System
- **Cross-platform support** - Works on Linux (V4L2) and macOS (AVFoundation)
- **H.264 compression** - Efficient encoding with universal compatibility
- **Automatic organization** - Videos saved in `videos/YYYY-MM-DD/HH/MM.mp4` structure
- **1-minute segments** - Automatic video segmentation for easy management
- **Real-time frame sharing** - Shared memory interface for live AI processing
- **Reliable encoding** - Simplified pipeline for stable MP4 output

### AI Detection & Analysis
- **Advanced object detection** - Cars, persons, bicycles, motorcycles, buses, cats, dogs, birds
- **Smart captioning** - Detailed AI-generated descriptions using BLIP model
- **Semantic search** - Embeddings-based similarity searches
- **Real-time processing** - Live stream analysis capabilities
- **Web interface** - Modern responsive UI for viewing detections
- **Database storage** - Efficient SQLite storage with full metadata

## 🚀 Quick Start

### 1. System Requirements

**Linux:**
- Ubuntu 18.04+ or equivalent
- V4L2 compatible webcam
- ffmpeg with libx264: `sudo apt install build-essential ffmpeg`

**macOS:**
- macOS 10.12+ (Sierra or later)
- ffmpeg: `brew install ffmpeg`
- Xcode command line tools: `xcode-select --install`

**Python Environment:**
- Python 3.8+
- OpenCV and AI libraries: `pip install -r requirements.txt`

### 2. Installation & Build

```bash
# Check dependencies
make check-deps

# Build video capture system
make

# Install Python dependencies
pip install -r requirements.txt

# Test the system
make test
```

### 3. Basic Usage

**Start video capture:**
```bash
# Run with platform defaults
make run

# Or run directly:
# Linux: ./video_capture /dev/video0
# macOS: ./video_capture
```

**Process captured video with AI:**
```bash
# Process recent video files
python parse_video.py videos/2024-01-15/09/00.mp4 --confidence 0.6

# Or process live frames (while capture is running)
python parse_video.py 0 --stream --max-frames 1000
```

**View results in web interface:**
```bash
python web_viewer.py
# Visit http://localhost:3000
```

## 📁 File Structure & Organization

### Video Files (Automatic)
```
videos/
├── 2024-01-15/          # Date
│   ├── 09/              # Hour
│   │   ├── 00.mp4       # Minute 00 (H.264 compressed)
│   │   ├── 01.mp4       # Minute 01
│   │   └── ...
│   └── 10/
│       ├── 00.mp4
│       └── ...
└── 2024-01-16/
    └── ...
```

### Project Structure
```
my2ndeye/
├── README.md                  # This guide
├── requirements.txt           # Python dependencies
├── Makefile                  # Build system
│
├── video_capture.c           # H.264 video capture (main program)
├── frame_reader.py          # Python interface for live frames
│
├── parse_video.py           # AI object detection
├── caption_crops.py         # AI captioning system
├── web_viewer.py           # Web interface
├── database.py             # Database utilities
│
├── templates/              # Web UI templates
│   └── index.html          # Detection viewer
└── detections.db           # SQLite database (auto-created)
```

## 🛠 Build System

### Available Commands

```bash
make                    # Build H.264 video capture (default)
make clean              # Remove build files
make install            # Install to /usr/local/bin
make check-deps         # Check system dependencies
make test               # Show test command
make run                # Run with platform defaults
make debug              # Build with debug symbols
make help               # Show all available commands
```

## 🎥 Video Capture System

### Technical Specifications
- **Resolution:** 640x480 @ 30fps (configurable in source)
- **Codec:** H.264/AVC (libx264)
- **Quality:** CRF 23 (balanced quality/size)
- **Container:** MP4 with fast-start
- **Storage:** ~1.5GB/hour
- **Encoding:** Simplified pipeline for reliability

### Platform-Specific Implementation

**Linux (V4L2):**
- Direct hardware access via Video4Linux2
- Memory-mapped buffers for efficiency
- RGB24 capture → H.264 encoding

**macOS (ffmpeg):**
- AVFoundation backend via ffmpeg
- Threaded capture pipeline
- Enhanced error reporting

### ffmpeg Encoding Command
```bash
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 640x480 -r 30 -i - \
       -c:v libx264 -preset fast -crf 23 \
       -pix_fmt yuv420p \
       -movflags +faststart output.mp4
```

## 🤖 AI Detection Pipeline

### 1. Object Detection (`parse_video.py`)

**Supported Objects:**
- Vehicles: Cars, bicycles, motorcycles, buses
- People: Persons
- Animals: Cats, dogs, birds

**Usage Examples:**
```bash
# Process video file
python parse_video.py video.mp4 --confidence 0.6

# Process live stream
python parse_video.py rtsp://camera.ip/stream --stream

# Process webcam
python parse_video.py 0 --stream --max-frames 1000

# Custom model and database
python parse_video.py video.mp4 --model yolov8s.pt --db custom.db
```

**Command Line Options:**
- `input`: Video file path or stream URL (required)
- `--model`: YOLOv8 model path (default: yolov8n.pt)  
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--db`: Database file path (default: detections.db)
- `--stream`: Process as stream instead of video file
- `--max-frames`: Maximum frames to process (streams only)
- `--top-n`: Top detections per object type to save (default: 10)

### 2. AI Captioning (`caption_crops.py`)

Generates detailed descriptions using BLIP model:

**Usage Examples:**
```bash
# Caption all uncaptioned detections
python caption_crops.py

# Process in smaller batches
python caption_crops.py --batch-size 5

# Caption specific detection
python caption_crops.py --detection-id 123
```

### 3. Web Viewer (`web_viewer.py`)

Modern responsive web interface:

**Features:**
- Grid view with thumbnails
- Date/time organization
- Search and filtering by object type
- Video playback (5 seconds before/after detection)
- Statistics dashboard
- Mobile-friendly responsive design

**Usage:**
```bash
# Start web server
python web_viewer.py

# Custom host/port
python web_viewer.py --host 0.0.0.0 --port 8080

# Debug mode
python web_viewer.py --debug
```

## 🔄 Real-Time Processing

### Live Stream Analysis

The system supports real-time processing with shared memory:

```python
# frame_reader.py - Access live frames from C capture
from frame_reader import FrameReader

def process_frame(frame_data):
    frame = frame_data['frame']  # OpenCV BGR format
    timestamp = frame_data['timestamp']
    
    # Your AI processing here
    print(f"Processing {frame.shape} frame at {timestamp}")

reader = FrameReader()
if reader.connect():
    reader.start_continuous_reading(process_frame)
```

### Integration with YOLO
```python
from frame_reader import FrameReader
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
reader = FrameReader()

def detect_objects(frame_data):
    frame = frame_data['frame']
    results = model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                print(f"Detected: {class_name} ({confidence:.2f})")

if reader.connect():
    reader.start_continuous_reading(detect_objects)
```

## 💾 Database Schema

Each detection record contains:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type TEXT NOT NULL,
    time TEXT NOT NULL,
    crop_of_object BLOB NOT NULL,
    full_frame BLOB,
    original_video_link TEXT,
    frame_num_original_video INTEGER,
    caption TEXT,
    embeddings BLOB,
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Additional tables:**
- `tracks` - Object tracking across frames
- `track_detections` - Links tracks to detections

## 📊 Performance & Optimization

### Hardware Requirements
- **Minimum:** Dual-core CPU, 4GB RAM
- **Recommended:** Quad-core CPU, 8GB RAM, SSD storage
- **Optional:** NVIDIA GPU with CUDA for faster AI processing

### Model Selection & Performance
| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| yolov8n.pt | Fastest | Basic | 3GB | Real-time, low-power |
| yolov8s.pt | Fast | Good | 4GB | Balanced performance |
| yolov8m.pt | Medium | Better | 6GB | Higher accuracy needs |
| yolov8l.pt | Slow | Best | 8GB+ | Offline, maximum accuracy |

### Storage Considerations
- **H.264 MP4:** ~1.5GB/hour
- **Database:** ~1-5MB per 1000 detections

### Optimization Tips

**Video Capture:**
- Use SSD for video storage
- Adjust resolution in source if needed
- Monitor disk space (automatic cleanup not implemented)

**AI Processing:**
- Lower confidence threshold = more detections
- Smaller batch sizes = lower memory usage
- GPU acceleration significantly faster for large videos

**System Performance:**
- Process videos during off-peak hours
- Use `--max-frames` for stream testing
- Regular database maintenance for large datasets

## 📋 Complete Workflows

### 1. Security Camera Setup
```bash
# Terminal 1: Start video capture
make run

# Terminal 2: Process live stream  
python parse_video.py 0 --stream --confidence 0.5

# Terminal 3: Auto-caption new detections
while true; do
    python caption_crops.py --batch-size 5
    sleep 30
done

# Terminal 4: Web interface
python web_viewer.py --host 0.0.0.0
```

### 2. Batch Video Analysis
```bash
# 1. Process all video files
for video in videos/*/*/*/*.mp4; do
    echo "Processing $video"
    python parse_video.py "$video" --confidence 0.6
done

# 2. Generate captions for all detections
python caption_crops.py --batch-size 10

# 3. View results
python web_viewer.py
```

### 3. Development & Testing
```bash
# Build and test
make clean && make
make test

# Test frame access
python frame_reader.py

# Test with sample data
python parse_video.py --help
python caption_crops.py --help
python web_viewer.py --help
```

## 🔧 Troubleshooting

### Video Capture Issues

**macOS Camera Permission:**
- System Preferences → Security & Privacy → Camera → Allow Terminal

**Linux Device Access:**
```bash
# Check available cameras
v4l2-ctl --list-devices

# Fix permissions
sudo usermod -a -G video $USER
# Log out and back in
```

**No Video Device:**
```bash
# List devices
ls /dev/video*

# Test different device
./video_capture /dev/video1
```

### Compilation Issues

**Missing Headers (Linux):**
```bash
sudo apt update
sudo apt install build-essential v4l-utils ffmpeg
```

**macOS Build Tools:**
```bash
xcode-select --install
brew install ffmpeg
```

### AI Processing Issues

**CUDA Out of Memory:**
```bash
# Use smaller batches
python caption_crops.py --batch-size 1

# Use smaller model
python parse_video.py video.mp4 --model yolov8n.pt
```

**Model Download Fails:**
```bash
# Manual download
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Database Locked:**
```bash
# Stop all processes
pkill -f parse_video.py
pkill -f caption_crops.py
pkill -f web_viewer.py
```

### Performance Issues

**High CPU Usage:**
- Use faster ffmpeg preset: edit source to use `-preset ultrafast`
- Reduce capture resolution in source code
- Process videos offline instead of real-time

**Storage Issues:**
```bash
# Check disk usage
du -sh videos/

# Compress older videos if needed
for video in videos/*/*/*.mp4; do
    ffmpeg -i "$video" -c:v libx264 -crf 28 "${video%.mp4}_compressed.mp4"
done
```

## 🔐 Security & Privacy

This system is designed for legitimate security and monitoring purposes:

- **Local Processing:** All AI processing happens locally
- **No Cloud Dependencies:** No external API calls required
- **Data Control:** Complete control over captured data
- **Privacy-Focused:** No automatic data sharing

## 📝 Configuration

### Video Capture Settings

Edit `video_capture.c` to customize:
```c
#define FRAME_WIDTH 640        // Frame width
#define FRAME_HEIGHT 480       // Frame height  
#define FRAMES_PER_MINUTE 1800 // Expected frames per minute
```

### AI Model Settings

Adjust confidence and model selection in commands:
```bash
# Higher confidence = fewer false positives
python parse_video.py video.mp4 --confidence 0.8

# Larger model = better accuracy
python parse_video.py video.mp4 --model yolov8l.pt
```

## 🤝 Contributing

1. **Test on both platforms** (Linux and macOS)
2. **Maintain backward compatibility** with existing database
3. **Document new features** and configuration options
4. **Include performance impact** of changes

## 📄 License

This project is provided for educational and legitimate security purposes only. The system should be used in compliance with local privacy and surveillance laws.

---

**Need Help?**
- Run `make help` for build options
- Use `--help` flag with Python scripts for detailed options
- Check logs if issues occur
- Test with sample data before production use