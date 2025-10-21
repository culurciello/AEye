#!/bin/bash

# Set full PATH (adjust based on where your Python is)
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# If using Homebrew Python, add:
export PATH="/opt/homebrew/bin:$PATH"

cd /Users/euge/Work/AEye-simple

# Run first command in background
python3 capture.py &

# Run second command in background
python3 process.py --monitor &

# run web:
python3 web_viewer.py --host 0.0.0.0 --port 3000
