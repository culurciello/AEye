# m2e - Video Surveillance & AI Detection System

A comprehensive cross-platform security camera system that combines high-performance video capture with advanced AI object detection and analysis. The system captures video streams with H.264 compression, processes them with YOLO models, and provides a complete web interface for viewing and analyzing detections.


### Installation

TBD


### Usage

**Process captured video with AI:**
```bash
# Process recent video files
python processor.py videos/2024-01-15/09/00.mp4 --confidence 0.15

# Or process live frames (while capture is running)
python processor.py 0 --stream --max-frames 1000
```

**View results in web interface:**
```bash
python web_viewer.py
# Visit http://localhost:3000
```
