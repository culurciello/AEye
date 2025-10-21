#!/usr/bin/env python3
import os
import time
from datetime import datetime

# ==========================
# Configuration
# ==========================
INTERVAL = 1            # seconds between snapshots
VIDEO_INTERVAL = 3600   # 1 hour
OUTPUT_BASE = "./data"

# List your cameras
CAMERAS = [
    {"name": "frnt", "ip": "192.168.6.254", "user": "admin", "pass": "tokkigeo1", "channel": 0},
    {"name": "back", "ip": "192.168.6.255", "user": "admin", "pass": "tokkigeo1", "channel": 0},
    {"name": "entr", "ip": "192.168.7.1", "user": "admin", "pass": "tokkigeo1", "channel": 0}
]

# ==========================
# Setup directories
# ==========================
for cam in CAMERAS:
    frame_dir = os.path.join(OUTPUT_BASE, "frames", cam["name"])
    video_dir = os.path.join(OUTPUT_BASE, "videos", cam["name"])
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

print(f"Starting snapshot capture every {INTERVAL}s...")
print(f"Videos will be created every {VIDEO_INTERVAL//60} minutes.")
print("Press Ctrl+C to stop.")

start_time = time.time()

while True:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for cam in CAMERAS:
        frame_dir = os.path.join(OUTPUT_BASE, "frames", cam["name"])
        outfile = os.path.join(frame_dir, f"{cam['name']}_{timestamp}.jpg")

        # HTTP Snap URL
        snap_url = f"http://{cam['ip']}:8000/cgi-bin/api.cgi?cmd=Snap&channel={cam['channel']}&user={cam['user']}&password={cam['pass']}"

        rtsp_url = f"rtsp://{cam['user']}:{cam['pass']}@{cam['ip']}:554/Preview_01_sub"
        cmd = f"ffmpeg -rtsp_transport tcp -y -i '{rtsp_url}' -frames:v 1 '{outfile}' -loglevel error"
        ret = os.system(cmd)
        if ret != 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RTSP failed for {cam['name']}")

    time.sleep(INTERVAL)

    # Check if it's time to create videos
    elapsed = time.time() - start_time
    if elapsed >= VIDEO_INTERVAL:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating hourly timelapse videos...")

        for cam in CAMERAS:
            frame_dir = os.path.join(OUTPUT_BASE, "frames", cam["name"])
            video_dir = os.path.join(OUTPUT_BASE, "videos", cam["name"])
            video_file = os.path.join(video_dir, f"{cam['name']}_{timestamp}.mp4")

            if os.listdir(frame_dir):
                cmd = f"ffmpeg -y -framerate 10 -pattern_type glob -i '{frame_dir}/{cam['name']}_*.jpg' " \
                      f"-vf \"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:" \
                      f"text='%{{localtime\\:%Y-%m-%d %H\\\\:%M\\\\:%S}}': x=10: y=10: fontsize=24: fontcolor=white: box=1: boxcolor=0x00000099\" " \
                      f"-c:v libx264 -pix_fmt yuv420p '{video_file}' -loglevel error"
                ret = os.system(cmd)
                if ret == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Video saved: {video_file}")
                    # Delete frames after video creation
                    for f in os.listdir(frame_dir):
                        os.remove(os.path.join(frame_dir, f))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Deleted frames for {cam['name']}")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] FFmpeg failed for {cam['name']}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No frames to create video for {cam['name']}")

        start_time = time.time()
