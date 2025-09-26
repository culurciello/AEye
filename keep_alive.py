#!/usr/bin/env python3
"""
Keep-alive script for AEye motion detection system.

This script monitors the main.py process and automatically restarts it
if it crashes due to camera failures or other issues.
"""

import subprocess
import time
import logging
import signal
import sys
import os
from datetime import datetime

# Configuration
COMMAND = ["python3", "main.py", "--video-source", "rtsp://192.168.6.244:554/11"]#, "--headless"]
RESTART_DELAY = 5  # seconds to wait before restarting
MAX_RESTART_ATTEMPTS = 5  # max restarts within RESTART_WINDOW
RESTART_WINDOW = 300  # 5 minutes window for counting restarts
LOG_FILE = "data/keep_alive.log"

# Global variables
process = None
restart_times = []
shutdown_requested = False

def setup_logging():
    """Setup logging configuration."""
    os.makedirs("data", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested, process

    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")

    shutdown_requested = True

    if process and process.poll() is None:
        logger.info("Terminating main.py process...")
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate gracefully, killing it...")
            process.kill()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")

    logger.info("Keep-alive script stopped")
    sys.exit(0)

def check_restart_rate():
    """Check if we're restarting too frequently."""
    global restart_times

    now = time.time()
    # Remove restart times older than the window
    restart_times = [t for t in restart_times if now - t < RESTART_WINDOW]

    if len(restart_times) >= MAX_RESTART_ATTEMPTS:
        return False

    restart_times.append(now)
    return True

def start_process():
    """Start the main.py process."""
    global process

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting process: {' '.join(COMMAND)}")
        process = subprocess.Popen(
            COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        logger.info(f"Process started with PID: {process.pid}")
        return True

    except Exception as e:
        logger.error(f"Failed to start process: {e}")
        return False

def monitor_process():
    """Monitor the running process and handle its output."""
    global process

    logger = logging.getLogger(__name__)

    try:
        # Read output in real-time
        while process.poll() is None and not shutdown_requested:
            output = process.stdout.readline()
            if output:
                # Log output from main.py (strip timestamp to avoid duplication)
                output_line = output.strip()
                if output_line:
                    logger.info(f"[main.py] {output_line}")
            time.sleep(0.1)

        # Get the exit code
        exit_code = process.poll()

        if not shutdown_requested:
            if exit_code == 0:
                logger.info("Process exited normally")
            else:
                logger.warning(f"Process exited with code: {exit_code}")

        return exit_code

    except Exception as e:
        logger.error(f"Error monitoring process: {e}")
        return -1

def main():
    """Main keep-alive loop."""
    global process, shutdown_requested

    setup_logging()
    logger = logging.getLogger(__name__)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("AEye Keep-Alive Script Started")
    logger.info(f"Command: {' '.join(COMMAND)}")
    logger.info(f"Restart delay: {RESTART_DELAY}s")
    logger.info(f"Max restarts: {MAX_RESTART_ATTEMPTS} in {RESTART_WINDOW}s")
    logger.info("Press Ctrl+C to stop")

    while not shutdown_requested:
        try:
            # Check if we can restart (rate limiting)
            if not check_restart_rate():
                logger.error(f"Too many restarts ({MAX_RESTART_ATTEMPTS}) in {RESTART_WINDOW}s window")
                logger.error("Waiting 5 minutes before allowing more restarts...")
                time.sleep(300)  # Wait 5 minutes
                restart_times.clear()  # Reset the counter
                continue

            # Start the process
            if not start_process():
                logger.error(f"Failed to start process, retrying in {RESTART_DELAY}s...")
                time.sleep(RESTART_DELAY)
                continue

            # Monitor the process
            exit_code = monitor_process()

            if shutdown_requested:
                break

            # Log the restart reason
            if exit_code != 0:
                logger.warning(f"Process crashed with exit code {exit_code}")
            else:
                logger.info("Process exited normally")

            # Wait before restarting
            if not shutdown_requested:
                logger.info(f"Restarting in {RESTART_DELAY} seconds...")
                time.sleep(RESTART_DELAY)

        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            if not shutdown_requested:
                time.sleep(RESTART_DELAY)

if __name__ == "__main__":
    main()