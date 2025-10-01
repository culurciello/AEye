#!/bin/bash
cd /Users/euge/Work/AEye

# Run first command in background
./ingest_ffmpeg rtsp://admin:tokkigeo1@192.168.6.254:554/Preview_01_sub data/videos &

# Run second command in foreground
python process.py --monitor

