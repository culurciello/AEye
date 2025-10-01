# Makefile for AEyeMon C components

CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lavformat -lavcodec -lavutil

# Try to use pkg-config if available
PKG_CONFIG := $(shell command -v pkg-config 2> /dev/null)
ifdef PKG_CONFIG
    CFLAGS += $(shell pkg-config --cflags libavformat libavcodec libavutil 2>/dev/null)
    LDFLAGS = $(shell pkg-config --libs libavformat libavcodec libavutil 2>/dev/null)
endif

TARGET = ingest_ffmpeg
SRC = ingest_ffmpeg.c

.PHONY: all clean install help

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling $(TARGET)..."
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "Build complete: ./$(TARGET)"
	@echo ""
	@echo "Usage: ./$(TARGET) <rtsp_url> [output_dir]"
	@echo "Example: ./$(TARGET) rtsp://192.168.1.100:554/stream data/videos"

clean:
	@echo "Cleaning build files..."
	rm -f $(TARGET)
	@echo "Clean complete"

install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin..."
	install -m 755 $(TARGET) /usr/local/bin/
	@echo "Install complete"

help:
	@echo "AEyeMon C Components Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the ingest_ffmpeg binary"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make install  - Install to /usr/local/bin (requires sudo)"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - gcc compiler"
	@echo "  - FFmpeg development libraries (libavformat, libavcodec, libavutil, libswscale)"
	@echo ""
	@echo "Install FFmpeg libraries:"
	@echo "  macOS:   brew install ffmpeg"
	@echo "  Ubuntu:  sudo apt-get install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev"
	@echo "  CentOS:  sudo yum install ffmpeg-devel"
