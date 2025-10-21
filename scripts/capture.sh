#!/bin/bash

# ================================
# Capture 1 frame every 10 seconds from each RTSP camera
# Save into separate folders per camera
# Create a timelapse video every 1 hour
# Delete frames only if video creation succeeds
# Requires: ffmpeg installed
# ================================

# Configuration
INTERVAL=10               # seconds between snapshots
OUTPUT_BASE="./data"  # base output directory
FRAME_DIR="$OUTPUT_BASE/images"
VIDEO_DIR="$OUTPUT_BASE/videos"

# Define your cameras (name|RTSP URL)
CAMERAS=(
  "frnt|rtsp://admin:tokkigeo1@192.168.6.254:554/Preview_01_sub"
  "back|rtsp://admin:tokkigeo1@192.168.6.255:554/Preview_01_sub"
  "entr|rtsp://admin:tokkigeo1@192.168.7.1:554/Preview_01_sub"
)

# Create necessary directories
mkdir -p "$FRAME_DIR" "$VIDEO_DIR"

echo "Starting snapshot capture every $INTERVAL seconds..."
echo "A timelapse video will be created every hour."
echo "Press Ctrl+C to stop."

# Variables to track elapsed time
START_TIME=$(date +%s)

while true; do
  TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

  # Capture one frame per camera
  for ENTRY in "${CAMERAS[@]}"; do
    NAME="${ENTRY%%|*}"
    URL="${ENTRY##*|}"

    DIR="$FRAME_DIR/$NAME"
    mkdir -p "$DIR"
    OUTFILE="$DIR/${NAME}_${TIMESTAMP}.jpg"

    echo "[$(date +"%H:%M:%S")] Capturing from $NAME ..."
    ffmpeg -rtsp_transport tcp -y -i "$URL" -frames:v 1 -q:v 2 "$OUTFILE" -loglevel error
  done

  sleep "$INTERVAL"

  # Check elapsed time
  NOW=$(date +%s)
  ELAPSED=$((NOW - START_TIME))

  # If 1 hour (3600 s) has passed, create timelapse videos
  if [ "$ELAPSED" -ge 3600 ]; then
    echo "[$(date +"%H:%M:%S")] Creating hourly timelapse videos..."

    for ENTRY in "${CAMERAS[@]}"; do
      NAME="${ENTRY%%|*}"
      FRAME_PATH="$FRAME_DIR/$NAME"
      mkdir -p "$FRAME_PATH"

      # Only create video if there are frames
      if ls "$FRAME_PATH"/*.jpg >/dev/null 2>&1; then
        VIDEO_FILE="$VIDEO_DIR/${NAME}_$(date +"%Y-%m-%d_%H-%M-%S").mp4"

        echo "[$(date +"%H:%M:%S")] Creating video for $NAME ..."
        ffmpeg -y -framerate 10 -pattern_type glob -i "$FRAME_PATH/${NAME}_*.jpg" \
               -c:v libx264 -pix_fmt yuv420p "$VIDEO_FILE" -loglevel error

        # Delete frames only if video exists and is not empty
        if [ -s "$VIDEO_FILE" ]; then
          echo "[$(date +"%H:%M:%S")] Deleting old frames for $NAME ..."
          rm -f "$FRAME_PATH/${NAME}_*.jpg"
        else
          echo "[$(date +"%H:%M:%S")] ⚠️ Video not created for $NAME — keeping frames."
        fi
      else
        echo "[$(date +"%H:%M:%S")] No frames found for $NAME, skipping video."
      fi
    done

    # Reset timer for next hour
    START_TIME=$(date +%s)
  fi
done
