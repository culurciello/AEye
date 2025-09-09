# m2e - Video Object Detection

AI-powered object detection for video files with web interface.

## Usage

**Process video/stream:**
```bash
# Video file
python processor.py video.mp4 --confidence 0.5

# Webcam (auto-detected)
python processor.py 0 --confidence 0.5

# RTSP stream (auto-detected)  
python processor.py rtsp://ipaddr:554/11 --confidence 0.5
```

**View results:**
```bash
python web_viewer.py --port 3000
# Visit http://localhost:3000
```

## Options

- `--confidence`: Detection threshold (0.0-1.0)
- `--show-live`: Display video during processing
- `--stream`: Process camera/stream input


#### ctronics camera:
rtsp://192.168.6.244:554/11