# m2e - Video Object Detection

AI-powered object detection for video files with web interface.

## Usage

**Process video:**
```bash
python processor.py video.mp4 --confidence 0.5
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