# My2ndEye - Advanced Video Surveillance & AI Detection System

A comprehensive cross-platform security camera system that combines high-performance video capture with advanced AI object detection and analysis. The system captures video streams with H.264/5 compression, processes them with YOLO models, and provides a complete web interface for viewing and analyzing detections.


## Usage Examples:

Default EC mode:

```
python3 smart_video_system.py -r rtsp://192.168.6.244:554/11 --enable-ai --show-frames --show-detections
```


### Basic Video Recording (No AI)
```bash
# Record from webcam
python3 smart_video_system.py -w 0

# Record from RTSP stream
python3 smart_video_system.py -r rtsp://192.168.6.244:554/11

# Record with display
python3 smart_video_system.py -w 0 --show-frames
```

### AI-Enhanced Recording
```bash
# Record with AI detection
python3 smart_video_system.py -r rtsp://192.168.6.244:554/11 --enable-ai

# Record with AI and real-time display
python3 smart_video_system.py -w 0 --enable-ai --show-frames --show-detections

# Custom AI settings
python3 smart_video_system.py -w 0 --enable-ai --model yolov8s.pt --confidence 0.3
```

### Search Existing Data
```bash
# Search without recording
python3 smart_video_system.py --search "red car"
python3 smart_video_system.py --search "person walking"
```

### Individual Service Testing
```bash
# Test recording service only
python3 video_recording_service.py -r rtsp://192.168.6.244:554/11 --show-frames

# Test detection service with image
python3 detection_service.py --test-image photo.jpg --search "car"

# Test semantic search
python3 ai_agent.py --search "blue vehicle" --db detections.db
```



#### ctronics camera:

rtsp://192.168.6.244:554/11

