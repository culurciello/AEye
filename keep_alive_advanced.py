#!/usr/bin/env python3
"""
Advanced Keep-alive script for AEye motion detection system.

This script provides advanced monitoring and recovery features:
- Configurable restart strategies
- Health checks based on log output
- Email notifications (optional)
- Different restart delays based on failure type
- Process health monitoring
"""

import subprocess
import time
import logging
import signal
import sys
import os
import json
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

class AEyeKeepAlive:
    def __init__(self, config_file="keep_alive_config.json"):
        self.config = self.load_config(config_file)
        self.process = None
        self.restart_times = []
        self.shutdown_requested = False
        self.last_heartbeat = None
        self.consecutive_failures = 0

        self.setup_logging()
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file):
        """Load configuration from JSON file or create default."""
        default_config = {
            "command": ["python3", "main.py", "--video-source", "rtsp://192.168.6.244:554/11"],# "--headless"],
            "restart_delay": 5,
            "max_restart_attempts": 5,
            "restart_window": 300,
            "health_check": {
                "enabled": True,
                "timeout": 120,
                "heartbeat_pattern": "Motion-triggered processor started|Motion detected|Face recognition|Object detection"
            },
            "failure_strategies": {
                "camera_failure": {
                    "delay": 30,
                    "max_attempts": 10,
                    "backoff_multiplier": 1.5
                },
                "general_failure": {
                    "delay": 5,
                    "max_attempts": 5,
                    "backoff_multiplier": 1.2
                }
            },
            "notifications": {
                "enabled": False,
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "to_address": ""
                }
            },
            "logging": {
                "level": "INFO",
                "file": "data/keep_alive.log",
                "max_size_mb": 10,
                "backup_count": 5
            }
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")

        # Save config for reference
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

        return default_config

    def setup_logging(self):
        """Setup logging with rotation."""
        log_config = self.config["logging"]
        os.makedirs(os.path.dirname(log_config["file"]), exist_ok=True)

        from logging.handlers import RotatingFileHandler

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_config["file"],
            maxBytes=log_config["max_size_mb"] * 1024 * 1024,
            backupCount=log_config["backup_count"]
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_config["level"]))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_requested = True

        if self.process and self.process.poll() is None:
            self.logger.info("Terminating main.py process...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Process didn't terminate gracefully, killing it...")
                self.process.kill()
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")

        self.logger.info("Keep-alive script stopped")
        sys.exit(0)

    def check_restart_rate(self, failure_type="general_failure"):
        """Check if we're restarting too frequently."""
        strategy = self.config["failure_strategies"][failure_type]
        max_attempts = strategy["max_attempts"]

        now = time.time()
        window = self.config["restart_window"]

        # Remove restart times older than the window
        self.restart_times = [t for t in self.restart_times if now - t < window]

        if len(self.restart_times) >= max_attempts:
            return False

        self.restart_times.append(now)
        return True

    def get_restart_delay(self, failure_type="general_failure"):
        """Calculate restart delay with exponential backoff."""
        strategy = self.config["failure_strategies"][failure_type]
        base_delay = strategy["delay"]
        multiplier = strategy["backoff_multiplier"]

        # Apply exponential backoff based on consecutive failures
        delay = base_delay * (multiplier ** min(self.consecutive_failures, 5))
        return min(delay, 300)  # Cap at 5 minutes

    def detect_failure_type(self, exit_code, last_output):
        """Detect the type of failure based on exit code and output."""
        if last_output:
            # Check for camera-related errors
            camera_errors = [
                "Could not open video source",
                "Failed to read frame",
                "Connection timed out",
                "RTSP",
                "Camera error",
                "Video capture failed"
            ]

            for error in camera_errors:
                if error.lower() in last_output.lower():
                    return "camera_failure"

        return "general_failure"

    def send_notification(self, subject, message):
        """Send email notification if configured."""
        if not self.config["notifications"]["enabled"]:
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            email_config = self.config["notifications"]["email"]

            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = email_config["to_address"]
            msg['Subject'] = f"AEye Alert: {subject}"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()

            self.logger.info("Notification sent successfully")

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")

    def start_process(self):
        """Start the main.py process."""
        try:
            self.logger.info(f"Starting process: {' '.join(self.config['command'])}")
            self.process = subprocess.Popen(
                self.config["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.logger.info(f"Process started with PID: {self.process.pid}")
            self.last_heartbeat = time.time()
            return True

        except Exception as e:
            self.logger.error(f"Failed to start process: {e}")
            return False

    def monitor_process(self):
        """Monitor the running process with health checks."""
        last_output = ""
        health_config = self.config["health_check"]

        try:
            while self.process.poll() is None and not self.shutdown_requested:
                output = self.process.stdout.readline()
                if output:
                    output_line = output.strip()
                    if output_line:
                        self.logger.info(f"[main.py] {output_line}")
                        last_output = output_line

                        # Check for heartbeat patterns
                        if health_config["enabled"]:
                            heartbeat_pattern = health_config["heartbeat_pattern"]
                            if re.search(heartbeat_pattern, output_line, re.IGNORECASE):
                                self.last_heartbeat = time.time()

                # Check for health timeout
                if (health_config["enabled"] and self.last_heartbeat and
                    time.time() - self.last_heartbeat > health_config["timeout"]):
                    self.logger.warning("Health check timeout - no heartbeat detected")
                    self.process.terminate()
                    break

                time.sleep(0.1)

            return self.process.poll(), last_output

        except Exception as e:
            self.logger.error(f"Error monitoring process: {e}")
            return -1, last_output

    def run(self):
        """Main keep-alive loop."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info("AEye Advanced Keep-Alive Script Started")
        self.logger.info(f"Command: {' '.join(self.config['command'])}")
        self.logger.info("Press Ctrl+C to stop")

        while not self.shutdown_requested:
            try:
                # Start the process
                if not self.start_process():
                    self.logger.error("Failed to start process, retrying...")
                    time.sleep(self.config["restart_delay"])
                    continue

                # Monitor the process
                exit_code, last_output = self.monitor_process()

                if self.shutdown_requested:
                    break

                # Detect failure type
                failure_type = self.detect_failure_type(exit_code, last_output)

                # Check restart rate
                if not self.check_restart_rate(failure_type):
                    self.logger.error(f"Too many {failure_type} restarts, backing off...")
                    self.send_notification("Restart Limit Reached",
                                         f"AEye process has failed too many times ({failure_type})")
                    time.sleep(300)  # Wait 5 minutes
                    self.restart_times.clear()
                    continue

                # Calculate restart delay
                restart_delay = self.get_restart_delay(failure_type)

                if exit_code != 0:
                    self.consecutive_failures += 1
                    self.logger.warning(f"Process crashed ({failure_type}) - consecutive failures: {self.consecutive_failures}")

                    if self.consecutive_failures >= 3:
                        self.send_notification("Multiple Failures",
                                             f"AEye process has failed {self.consecutive_failures} times consecutively")
                else:
                    self.consecutive_failures = 0
                    self.logger.info("Process exited normally")

                if not self.shutdown_requested:
                    self.logger.info(f"Restarting in {restart_delay:.1f} seconds...")
                    time.sleep(restart_delay)

            except KeyboardInterrupt:
                self.signal_handler(signal.SIGINT, None)
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                if not self.shutdown_requested:
                    time.sleep(self.config["restart_delay"])

def main():
    parser = argparse.ArgumentParser(description='Advanced Keep-alive script for AEye')
    parser.add_argument('--config', default='keep_alive_config.json',
                       help='Configuration file path')
    parser.add_argument('--video-source',
                       help='Override video source in config')

    args = parser.parse_args()

    keeper = AEyeKeepAlive(args.config)

    # Override video source if provided
    if args.video_source:
        # Find and replace video source in command
        cmd = keeper.config["command"]
        try:
            idx = cmd.index("--video-source")
            cmd[idx + 1] = args.video_source
            keeper.logger.info(f"Video source overridden to: {args.video_source}")
        except ValueError:
            keeper.logger.warning("Could not find --video-source in command")

    keeper.run()

if __name__ == "__main__":
    main()