#!/bin/bash
# Quick script to check if video files are valid

if [ $# -eq 0 ]; then
    echo "Usage: $0 <video_file>"
    echo "Example: $0 data/videos/2025_09_30/20250930_194500.mp4"
    exit 1
fi

VIDEO_FILE="$1"

if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: File not found: $VIDEO_FILE"
    exit 1
fi

echo "Checking video file: $VIDEO_FILE"
echo "="

# Check file size
SIZE=$(stat -f%z "$VIDEO_FILE" 2>/dev/null || stat -c%s "$VIDEO_FILE" 2>/dev/null)
echo "File size: $SIZE bytes"

# Use ffprobe to check video info
echo ""
echo "Video info:"
ffprobe -v quiet -print_format json -show_format -show_streams "$VIDEO_FILE" | grep -E '"codec_name"|"width"|"height"|"duration"|"nb_frames"|"bit_rate"' | head -10

# Extract a test frame to check if it's black
echo ""
echo "Extracting test frame to /tmp/test_frame.jpg..."
ffmpeg -i "$VIDEO_FILE" -vframes 1 -ss 00:00:01 /tmp/test_frame.jpg -y 2>&1 | tail -5

if [ -f /tmp/test_frame.jpg ]; then
    # Check if frame is mostly black (average pixel value)
    STATS=$(ffmpeg -i /tmp/test_frame.jpg -vf "signalstats" -f null - 2>&1 | grep YAVG)
    echo ""
    echo "Frame statistics:"
    echo "$STATS"

    # Extract average Y value (brightness)
    YAVG=$(echo "$STATS" | grep -oE 'YAVG:[0-9]+' | cut -d: -f2)

    if [ -n "$YAVG" ]; then
        echo ""
        if [ "$YAVG" -lt 20 ]; then
            echo "⚠️  WARNING: Video appears to be very dark/black (Y avg: $YAVG)"
        else
            echo "✓ Video appears to have content (Y avg: $YAVG)"
        fi
    fi

    echo ""
    echo "Test frame saved to: /tmp/test_frame.jpg"
    echo "Open it to verify: open /tmp/test_frame.jpg"
fi

echo ""
echo "To play video: ffplay $VIDEO_FILE"
