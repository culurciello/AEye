#!/bin/bash

# Set full PATH (adjust based on where your Python is)
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# If using Homebrew Python, add:
export PATH="/opt/homebrew/bin:$PATH"

cd /Users/euge/Work/AEye

# Run first command in background
./ingest_ffmpeg rtsp://admin:tokkigeo1@192.168.6.254:554/Preview_01_sub data/videos &

# Run second command in background
python3 process.py --monitor &

# run web:
python3 web_viewer.py --host 0.0.0.0 --port 3000
