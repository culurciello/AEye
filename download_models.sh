#!/bin/bash

# Script to download YOLO models
# Usage: ./download_models.sh [model_size]
# If no model_size specified, downloads yolov8s (small) as a good default

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

BASE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0"

# Model mapping function
get_model_file() {
    case "$1" in
        "nano"|"n") echo "yolov8n.pt" ;;
        "small"|"s") echo "yolov8s.pt" ;;
        "medium"|"m") echo "yolov8m.pt" ;;
        "large"|"l") echo "yolov8l.pt" ;;
        "extra-large"|"x") echo "yolov8x.pt" ;;
        *) echo "" ;;
    esac
}

get_description() {
    case "$1" in
        "yolov8n.pt") echo "YOLOv8 Nano - Fastest, smallest (6.2MB)" ;;
        "yolov8s.pt") echo "YOLOv8 Small - Good balance (21.5MB)" ;;
        "yolov8m.pt") echo "YOLOv8 Medium - Better accuracy (49.7MB)" ;;
        "yolov8l.pt") echo "YOLOv8 Large - High accuracy (83.7MB)" ;;
        "yolov8x.pt") echo "YOLOv8 Extra Large - Best accuracy (131.4MB)" ;;
        *) echo "Unknown model" ;;
    esac
}

download_model() {
    local model_file=$1
    local model_path="$MODELS_DIR/$model_file"
    
    if [ -f "$model_path" ]; then
        echo "✓ $model_file already exists"
        return 0
    fi
    
    local description=$(get_description "$model_file")
    echo "📥 Downloading $model_file ($description)..."
    if wget -q --show-progress "$BASE_URL/$model_file" -O "$model_path"; then
        echo "✅ Downloaded $model_file successfully"
    else
        echo "❌ Failed to download $model_file"
        rm -f "$model_path"
        return 1
    fi
}

if [ $# -eq 0 ]; then
    echo "🚀 Downloading recommended YOLOv8 Small model..."
    download_model "yolov8s.pt"
elif [ "$1" = "all" ]; then
    echo "🚀 Downloading all YOLOv8 models..."
    for model in yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt; do
        download_model "$model"
    done
else
    model_file=$(get_model_file "$1")
    if [ -n "$model_file" ]; then
        echo "🚀 Downloading $model_file..."
        download_model "$model_file"
    else
        echo "❌ Unknown model size: $1"
        echo ""
        echo "Available options:"
        echo "  nano, n       - YOLOv8 Nano (6.2MB)"
        echo "  small, s      - YOLOv8 Small (21.5MB) [recommended]"
        echo "  medium, m     - YOLOv8 Medium (49.7MB)"
        echo "  large, l      - YOLOv8 Large (83.7MB)"
        echo "  extra-large, x - YOLOv8 Extra Large (131.4MB)"
        echo "  all           - Download all models"
        echo ""
        echo "Usage: $0 [model_size]"
        echo "Example: $0 small"
        exit 1
    fi
fi

echo ""
echo "🎯 Available models:"
python3 processor.py --list-models 2>/dev/null || echo "Run 'python3 processor.py --list-models' to see available models"