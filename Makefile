CC=gcc
CFLAGS=-Wall -Wextra -O2 -std=c99
LDFLAGS=-lpthread

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -lrt
    PLATFORM_FLAGS = -DPLATFORM_LINUX
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM_FLAGS = -DPLATFORM_MACOS
endif

TARGET=video_capture
SOURCES=video_capture.c

.PHONY: all clean install test run check-deps help

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(PLATFORM_FLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
ifeq ($(UNAME_S),Darwin)
	cp $(TARGET) /usr/local/bin/
	chmod +x /usr/local/bin/$(TARGET)
else
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/$(TARGET)
endif

# Development targets
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

test: $(TARGET)
	@echo "Testing H.264 video capture..."
	@echo "Platform: $(UNAME_S)"
ifeq ($(UNAME_S),Darwin)
	@echo "To test: make run"
else
	@echo "To test: make run"
endif

run: $(TARGET)
ifeq ($(UNAME_S),Darwin)
	./$(TARGET)
else
	./$(TARGET) /dev/video0
endif

check-deps:
	@echo "Checking dependencies..."
ifeq ($(UNAME_S),Darwin)
	@echo "macOS detected"
	@which ffmpeg > /dev/null || echo "WARNING: ffmpeg not found. Install with: brew install ffmpeg"
	@ffmpeg -encoders 2>/dev/null | grep libx264 > /dev/null || echo "ERROR: libx264 encoder not found"
	@echo "Required: ffmpeg (for camera access and H.264 encoding)"
else
	@echo "Linux detected"
	@ls /dev/video* 2>/dev/null || echo "WARNING: No video devices found"
	@which ffmpeg > /dev/null || echo "WARNING: ffmpeg not found. Install with: sudo apt install ffmpeg"
	@ffmpeg -encoders 2>/dev/null | grep libx264 > /dev/null || echo "ERROR: libx264 encoder not found"
	@echo "Required: v4l2 compatible camera + ffmpeg (for H.264 encoding)"
endif
	@echo "Dependencies check complete"

help:
	@echo "Video Capture System - Available targets:"
	@echo ""
	@echo "  make                Build H.264 video capture (default)"
	@echo "  make clean          Remove build files"
	@echo "  make install        Install to /usr/local/bin"
	@echo "  make check-deps     Check system dependencies"
	@echo "  make test           Show test command"
	@echo "  make run            Run with platform defaults"
	@echo "  make debug          Build with debug symbols"
	@echo "  make help           Show this help"
	@echo ""
	@echo "Platform: $(UNAME_S)"
	@echo ""
	@echo "Output: H.264 MP4 files in videos/YYYY-MM-DD/HH/MM.mp4"