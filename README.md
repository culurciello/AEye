# My2ndEye - Advanced Video Surveillance & AI Detection System

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
- **Ultra-fast shared memory** - 3-5x faster frame processing vs OpenCV
- **Parallel operation** - Video saving + AI detection simultaneously
- **Reliable encoding** - Simplified pipeline for stable MP4 output

### AI Detection & Analysis
- **Advanced object detection** - Cars, persons, bicycles, motorcycles, buses, cats, dogs
- **Enhanced bounding boxes** - 25% padding for better context
- **Size filtering** - Ignores objects smaller than 100x100 pixels
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

**Automated Installation (Recommended):**
```bash
# Run the installation script
./install.sh
```

**Manual Installation:**
```bash
# Install system dependencies (Linux)
sudo apt install build-essential ffmpeg v4l-utils  

# Install system dependencies (macOS) 
brew install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Build video capture system
make clean && make

# Test the system
make check-deps
```

**Development Installation:**
```bash
# Install additional development tools
pip install -r requirements-dev.txt
```

### 3. Basic Usage

**Ultra-Fast Shared Memory Processing (Recommended):**
```bash
# Terminal 1: Start video capture
./video_capture

# Terminal 2: Start AI processing with shared memory
python3 parse_video.py --shared-memory --confidence 0.5
```

**Standard Video Processing:**
```bash
# Process video files
python parse_video.py videos/2025-09-07/07/00.mp4 --confidence 0.6

# Process live stream
python parse_video.py 0 --stream --max-frames 1000
```

**Web Interface:**
```bash
python web_viewer.py
# Visit http://localhost:3000
```

## ⚡ Shared Memory Integration (Ultra-Fast)

### Performance Comparison

| Method | Frame Access | Parallel Processing | File Saving |
|--------|--------------|-------------------|-------------|
| **Standard OpenCV** | ~10-15ms | ❌ Serial | ❌ Manual |
| **Shared Memory** | ~1-2ms | ✅ Parallel | ✅ Automatic |

### Quick Setup

1. **Build the video capture system:**
```bash
cd lib
make clean && make
```

2. **Start video capture (Terminal 1):**
```bash
cd lib
./video_capture
```
This will:
- Start capturing at 1280x720 @ 20fps
- Save to `videos/YYYY-MM-DD/HH/MM.mp4`
- Create shared memory for frame processing

3. **Start AI processing (Terminal 2):**
```bash
# Basic shared memory processing
python3 parse_video.py --shared-memory

# With custom confidence threshold
python3 parse_video.py --shared-memory --confidence 0.7

# Process limited frames for testing
python3 parse_video.py --shared-memory --max-frames 100
```

### Troubleshooting Shared Memory

**"Failed to connect to shared memory":**
- Make sure `./video_capture` is running first
- Check that camera is accessible (macOS: allow camera access)

**"Unexpected frame size":**
- Shared memory reader expects 1280x720 frames
- Check video_capture.c `FRAME_WIDTH` and `FRAME_HEIGHT` constants

## 📁 File Structure & Organization

### Video Files (Automatic)
```
videos/
├── 2025-09-07/          # Date
│   ├── 07/              # Hour
│   │   ├── 00.mp4       # Minute 00 (H.264 compressed)
│   │   ├── 01.mp4       # Minute 01
│   │   └── ...
│   └── 08/
│       ├── 00.mp4
│       └── ...
└── detections.db        # AI detection results
```

### Project Structure
```
my2ndeye/
├── README.md                     # This comprehensive guide
├── requirements.txt              # Core Python dependencies  
├── requirements-dev.txt          # Development dependencies
├── install.sh                    # Automated installation script
│
├── lib/                         # C video capture system
│   ├── Makefile                 # Build system
│   ├── video_capture.c          # Main H.264 capture program
│   └── video_capture            # Compiled binary
│
├── parse_video.py               # AI object detection
├── shared_memory_reader.py      # Shared memory interface
├── caption_crops.py             # AI captioning system
├── web_viewer.py               # Web interface
├── database.py                 # Database utilities
├── ai_agent.py                 # Semantic search capabilities
├── test_crop_export.py         # Export detection crops
├── test_integration.py         # Integration testing
│
├── models/                     # AI model storage (auto-created)
│   └── yolov8n.pt              # YOLOv8 model (downloaded)
├── templates/                  # Web UI templates
│   └── index.html              # Detection viewer
├── videos/                     # Video storage (auto-created)
├── test_crop_of_object/        # Exported crops (auto-created)
└── detections.db               # SQLite database (auto-created)
```

## 🛠 Build System

### Available Commands (lib directory)

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
- **Resolution:** 1280x720 @ 30fps (configurable in source)
- **Codec:** H.264/AVC (libx264) 
- **Quality:** CRF 22 (high quality encoding)
- **Container:** MP4 with fast-start
- **Storage:** ~3.4GB/hour at 1280x720 @ 30fps
- **Encoding:** Platform-optimized pipeline with macOS improvements

### Platform-Specific Implementation

**Linux (V4L2):**
- Direct hardware access via Video4Linux2
- Memory-mapped buffers for efficiency
- RGB24 capture → H.264 encoding

**macOS (AVFoundation via ffmpeg):**
- Native AVFoundation backend with optimized input
- Correct pixel format (uyvy422) for macOS cameras
- Audio + video input support ("0:0" format)
- Enhanced threaded capture pipeline
- Improved error handling and performance

### ffmpeg Commands

**Video Encoding (Output):**
```bash
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1280x720 -r 30 -i - \
       -c:v libx264 -crf 22 -preset medium \
       -pix_fmt yuv420p \
       -movflags +faststart output.mp4
```

**macOS Camera Input (Optimized):**
```bash
ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 \
       -video_size 1280x720 -i "0:0" \
       -pix_fmt rgb24 -f rawvideo -
```

## 🤖 AI Detection Pipeline

### 1. Object Detection (`parse_video.py`)

**Supported Objects:**
- Vehicles: Cars, buses
- People: Persons
- Animals: Cats, dogs

**Enhanced Features:**
- **Size filtering:** Ignores crops smaller than 100x100 pixels
- **Expanded bounding boxes:** 25% padding on all sides for better context
- **Smart tracking:** Groups detections across frames

**Usage Examples:**
```bash
# Ultra-fast shared memory processing
python parse_video.py --shared-memory --confidence 0.6

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
- `input`: Video file path or stream URL (required, not needed for --shared-memory)
- `--shared-memory`: Use ultra-fast shared memory processing
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
- **Smart video resolution** - Automatically finds videos for streaming sources
- Video playback (5 seconds before/after detection)
- Statistics dashboard
- Semantic search with AI-generated embeddings
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

### 4. Export Detection Crops (`test_crop_export.py`)

Export all detection crops for analysis:

```bash
# Export all crops to test_crop_of_object/ folder
python test_crop_export.py
```

## 💾 Database Schema

Each detection record contains:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type TEXT NOT NULL,
    time TIMESTAMP NOT NULL,
    crop_of_object BLOB NOT NULL,
    original_video_link TEXT NOT NULL,
    frame_num_original_video INTEGER NOT NULL,
    caption TEXT DEFAULT NULL,
    embeddings BLOB DEFAULT NULL,
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type TEXT NOT NULL,
    original_video_link TEXT NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    track_data TEXT NOT NULL,  -- JSON of bounding boxes per frame
    best_crop_detection_id INTEGER,  -- ID of detection with highest confidence
    avg_confidence REAL,
    detection_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (best_crop_detection_id) REFERENCES detections(id)
);
```

## 📊 Performance & Optimization

### Hardware Requirements
- **Minimum:** Dual-core CPU, 4GB RAM
- **Recommended:** Quad-core CPU, 8GB RAM, SSD storage
- **Optimal:** 6+ cores, 16GB RAM, CUDA GPU

### Model Selection & Performance
| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| yolov8n.pt | Fastest | Basic | 3GB | Real-time, low-power |
| yolov8s.pt | Fast | Good | 4GB | Balanced performance |
| yolov8m.pt | Medium | Better | 6GB | Higher accuracy needs |
| yolov8l.pt | Slow | Best | 8GB+ | Offline, maximum accuracy |

### Storage Considerations
- **H.264 MP4 (1280x720):** ~2.5GB/hour
- **H.264 MP4 (640x480):** ~1.5GB/hour
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
- Use shared memory for maximum performance

**System Performance:**
- Process videos during off-peak hours
- Use `--max-frames` for stream testing
- Regular database maintenance for large datasets

## 📋 Complete Workflows

### 1. Ultra-Fast Security Camera Setup (Recommended)
```bash
# Terminal 1: Start video capture
cd lib && ./video_capture

# Terminal 2: Process live stream with shared memory
python parse_video.py --shared-memory --confidence 0.5

# Terminal 3: Auto-caption new detections
while true; do
    python caption_crops.py --batch-size 5
    sleep 30
done

# Terminal 4: Web interface
python web_viewer.py --host 0.0.0.0
```

### 2. Standard Security Camera Setup
```bash
# Terminal 1: Start video capture
cd lib && ./video_capture

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

### 3. Batch Video Analysis
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

### 4. Development & Testing
```bash
# Build and test
cd lib && make clean && make
make test

# Test integration
python test_integration.py

# Test crop export
python test_crop_export.py

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
cd lib && ./video_capture /dev/video1
```

### Shared Memory Issues

**"Failed to connect to shared memory":**
- Make sure `./video_capture` is running first
- Check that camera access is granted
- Verify shared memory size limits

**Process hanging:**
- Kill all related processes: `pkill -f video_capture`
- Clean shared memory: `rm -f /tmp/video_capture_frame`

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

**Low Detection Rate:**
- Adjust `--confidence` threshold (default 0.5)
- Check lighting conditions
- Verify target object classes in parse_video.py
- Ensure objects are larger than 100x100 pixels

## 🔐 Security & Privacy

This system is designed for legitimate security and monitoring purposes:

- **Local Processing:** All AI processing happens locally
- **No Cloud Dependencies:** No external API calls required
- **Data Control:** Complete control over captured data
- **Privacy-Focused:** No automatic data sharing

## 📝 Configuration

### Video Capture Settings

Edit `lib/video_capture.c` to customize:
```c
#define FRAME_WIDTH 1280       // Frame width (current: 1280)
#define FRAME_HEIGHT 720       // Frame height (current: 720)
#define FRAMES_PER_MINUTE 1200 // Expected frames per minute (20fps)
```

### AI Model Settings

Adjust confidence and model selection in commands:
```bash
# Higher confidence = fewer false positives
python parse_video.py video.mp4 --confidence 0.8

# Larger model = better accuracy
python parse_video.py video.mp4 --model yolov8l.pt
```

### Detection Filtering

Edit `parse_video.py` to customize object classes:
```python
self.target_classes = {
    0: 'person',
    2: 'car', 
    5: 'bus',
    15: 'cat',
    16: 'dog',
    # Add more COCO classes as needed
}
```

## 🤝 Contributing

1. **Test on both platforms** (Linux and macOS)
2. **Maintain backward compatibility** with existing database
3. **Document new features** and configuration options
4. **Include performance impact** of changes
5. **Test shared memory integration** thoroughly

## 📄 License

This project is provided for educational and legitimate security purposes only. The system should be used in compliance with local privacy and surveillance laws.

---

## 🎯 Advanced Features Summary

### 🚀 **Ultra-Fast Shared Memory Processing**
- **3-5x faster** than standard OpenCV processing
- **Parallel operation:** Video saving + AI detection simultaneously
- **Zero frame drops:** Direct memory access with buffering
- **Real-time performance:** 1280x720 @ 20fps processing

### 🔍 **Enhanced Object Detection**
- **Size filtering:** Ignores objects < 100x100 pixels
- **Expanded bounding boxes:** 25% padding for better context
- **Smart tracking:** Groups detections across frames
- **Multiple object types:** Persons, cars, buses, cats, dogs

### 💾 **Intelligent Storage System**
- **Automatic organization:** `videos/YYYY-MM-DD/HH/MM.mp4`
- **1-minute segments:** Easy navigation and management
- **H.264 compression:** Universal compatibility
- **Database tracking:** Complete metadata storage

### 🌐 **Modern Web Interface**
- **Responsive design:** Works on mobile and desktop
- **Real-time updates:** See new detections immediately
- **Advanced search:** Filter by type, date, confidence
- **Video playback:** Context around each detection

---

**Need Help?**
- Run `cd lib && make help` for build options
- Use `--help` flag with Python scripts for detailed options
- Check logs if issues occur
- Test with sample data before production use
- Use shared memory mode for maximum performance



# Notes

use VLC to see stream:

```
vlc rtsp://username:password@192.168.6.225/live
```

correct ffmpeg command from mac to stream from webcam:

```
ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 -video_size 1280x720 -i "0:0" -c:v libx264 -preset ultrafast -c:a aac output.mp4
```

from screen:

```
ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 -video_size 1280x720 -i "1:0" -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/mystream
```

on VLC open:

```
rtsp://localhost:8554/mystream
```

## 📊 Latest Updates & System Status

### ✅ Recent Improvements (September 2025)

**1. Enhanced macOS Support:**
- Updated FFmpeg commands with correct macOS AVFoundation parameters
- Optimized pixel format (`uyvy422`) for native macOS camera support
- Audio + video input support (`"0:0"` format)
- Improved video quality (CRF 22, medium preset)

**2. Web Viewer Enhancements:**
- **Smart Video Resolution:** Automatically maps streaming sources to saved video files
- Works with camera indices ("0", "1"), shared memory, and RTSP streams
- Enhanced video playback for detections from any source type
- Better error handling and debugging information

**3. Video Capture Improvements:**
- Higher quality encoding (CRF 22 vs CRF 23)
- Increased frame rate (30fps vs 20fps) 
- Better compression efficiency with medium preset
- Enhanced performance monitoring

### 🔍 System Monitoring Commands

**Check if camera is running:**
```bash
ps aux | grep video_capture | grep -v grep
```

**Check video files being created:**
```bash
ls -la lib/videos/$(date +%Y-%m-%d)/$(date +%H)/
```

**Monitor encoding performance:**
```bash
# Look for frame= output showing encoding progress
tail -f /dev/null  # Replace with your capture process output
```

**Check system resources:**
```bash
# CPU usage
top -p $(pgrep video_capture)

# Disk space for videos
du -sh lib/videos/
```

### 📈 Performance Status

**Current Configuration:**
- **Resolution:** 1280x720 @ 30fps
- **Encoding:** H.264 CRF 22 (high quality)
- **Performance:** ~4x realtime encoding speed
- **Storage:** ~3.4GB per hour
- **Bitrate:** ~950 kbits/s average

**System Health Indicators:**
- ✅ Camera capture running successfully
- ✅ Video files created automatically (1-minute segments)
- ✅ Encoding performance: 120+ fps (4x realtime)  
- ✅ Shared memory integration functional
- ✅ Web viewer with streaming source support

### 🔧 Quick Troubleshooting

**Camera not working on macOS:**
```bash
# Check camera permissions
# System Preferences → Security & Privacy → Camera → Allow Terminal
```

**Video files not being created:**
```bash
# Check if video_capture is running
ps aux | grep video_capture

# Check disk space
df -h .

# Verify camera access
ls /dev/video* 2>/dev/null || echo "macOS: Camera accessed via AVFoundation"
```

**Performance issues:**
```bash
# Check system load
uptime

# Monitor memory usage
free -h  # Linux
vm_stat  # macOS
```