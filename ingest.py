#!/usr/bin/env python3
"""
Stable RTSP Ingest Wrapper
Uses C-based FFmpeg ingest for maximum stability
"""

import subprocess
import argparse
import logging
import os
import sys
import signal
import time

logger = logging.getLogger(__name__)


class StableIngester:
    """Wrapper for C-based FFmpeg ingest"""

    def __init__(self, video_source: str, output_dir: str = "data/videos"):
        self.video_source = video_source
        self.output_dir = output_dir
        self.process = None
        self.ingest_binary = self._find_ingest_binary()

    def _find_ingest_binary(self):
        """Find the ingest_ffmpeg binary"""
        # Check current directory
        if os.path.exists("./ingest_ffmpeg"):
            return "./ingest_ffmpeg"

        # Check in /usr/local/bin
        if os.path.exists("/usr/local/bin/ingest_ffmpeg"):
            return "/usr/local/bin/ingest_ffmpeg"

        # Check if it's in PATH
        from shutil import which
        binary = which("ingest_ffmpeg")
        if binary:
            return binary

        # Not found
        logger.error("ingest_ffmpeg binary not found. Please compile it first:")
        logger.error("  make")
        logger.error("or")
        logger.error("  gcc -o ingest_ffmpeg ingest_ffmpeg.c -lavformat -lavcodec -lavutil -lswscale -O3")
        return None

    def run(self):
        """Run the C-based ingest process"""
        if not self.ingest_binary:
            logger.error("Cannot run: ingest binary not found")
            return 1

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Starting stable RTSP ingest")
        logger.info(f"Binary: {self.ingest_binary}")
        logger.info(f"Source: {self.video_source}")
        logger.info(f"Output: {self.output_dir}")

        try:
            # Run the C binary
            cmd = [self.ingest_binary, self.video_source, self.output_dir]
            logger.info(f"Command: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Stream output in real-time
            for line in self.process.stdout:
                print(line, end='')
                sys.stdout.flush()

            # Wait for process to complete
            return_code = self.process.wait()

            if return_code != 0:
                logger.error(f"Ingest process exited with code: {return_code}")
                return return_code

            logger.info("Ingest process completed successfully")
            return 0

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            if self.process:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            return 0

        except Exception as e:
            logger.error(f"Error running ingest: {e}")
            if self.process:
                self.process.kill()
            return 1


def check_ffmpeg_libs():
    """Check if FFmpeg libraries are available"""
    try:
        # Try to run pkg-config to check for FFmpeg
        result = subprocess.run(
            ['pkg-config', '--exists', 'libavformat', 'libavcodec', 'libavutil', 'libswscale'],
            capture_output=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        # pkg-config not available, assume libs might be installed
        return True


def compile_binary():
    """Attempt to compile the ingest_ffmpeg binary"""
    logger.info("Attempting to compile ingest_ffmpeg...")

    if not os.path.exists("ingest_ffmpeg.c"):
        logger.error("Source file ingest_ffmpeg.c not found")
        return False

    if not check_ffmpeg_libs():
        logger.error("FFmpeg development libraries not found")
        logger.error("Install them with:")
        logger.error("  macOS:   brew install ffmpeg")
        logger.error("  Ubuntu:  sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev")
        return False

    try:
        # Try using make first
        if os.path.exists("Makefile"):
            result = subprocess.run(['make'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Compiled successfully using make")
                return True

        # Fallback to direct gcc compilation
        cmd = [
            'gcc', '-o', 'ingest_ffmpeg', 'ingest_ffmpeg.c',
            '-lavformat', '-lavcodec', '-lavutil', '-lswscale', '-O3'
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Compiled successfully")
            return True
        else:
            logger.error(f"Compilation failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Stable RTSP video ingest using C-based FFmpeg'
    )
    parser.add_argument('--video-source', required=True,
                       help='RTSP URL or video source')
    parser.add_argument('--output-dir', default='data/videos',
                       help='Output directory for video segments (default: data/videos)')
    parser.add_argument('--compile', action='store_true',
                       help='Compile the ingest_ffmpeg binary before running')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    os.makedirs('data', exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/ingest_stable.log')
        ]
    )

    # Compile if requested
    if args.compile:
        if not compile_binary():
            logger.error("Failed to compile binary")
            return 1

    # Run ingester
    try:
        ingester = StableIngester(
            video_source=args.video_source,
            output_dir=args.output_dir
        )

        return ingester.run()

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
