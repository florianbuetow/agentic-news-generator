#!/bin/bash

# Download FastText language detection models
# This script downloads the compressed FastText model (lid.176.ftz) if it doesn't exist
# Model location is read from config.yaml (data_models_dir path)

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read data_models_dir from config.yaml using Python
cd "$PROJECT_ROOT"
MODELS_DIR=$(uv run python -c "
import yaml
from pathlib import Path

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

models_dir = config.get('paths', {}).get('data_models_dir')
if not models_dir:
    raise ValueError('data_models_dir not found in config.yaml')

# Resolve to absolute path
path = Path(models_dir).resolve()
print(path)
")

# Create fasttext subdirectory
FASTTEXT_DIR="$MODELS_DIR/fasttext"
mkdir -p "$FASTTEXT_DIR"

# Model file path
MODEL_FILE="$FASTTEXT_DIR/lid.176.ftz"
MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "✓ FastText model already exists at $MODEL_FILE"
    exit 0
fi

# Download the model
echo "Downloading FastText language detection model (lid.176.ftz)..."
echo "  URL: $MODEL_URL"
echo "  Destination: $MODEL_FILE"

# Download with curl (shows progress)
if command -v curl &> /dev/null; then
    curl -L "$MODEL_URL" -o "$MODEL_FILE" --progress-bar
    DOWNLOAD_STATUS=$?
elif command -v wget &> /dev/null; then
    wget "$MODEL_URL" -O "$MODEL_FILE" --show-progress
    DOWNLOAD_STATUS=$?
else
    echo "Error: Neither curl nor wget is available. Please install one of them."
    exit 1
fi

# Validate download succeeded
if [ $DOWNLOAD_STATUS -ne 0 ]; then
    echo "Error: Failed to download FastText model"
    rm -f "$MODEL_FILE"  # Clean up partial download
    exit 1
fi

# Verify file exists and has content
if [ ! -f "$MODEL_FILE" ] || [ ! -s "$MODEL_FILE" ]; then
    echo "Error: Downloaded file is missing or empty"
    rm -f "$MODEL_FILE"
    exit 1
fi

echo "✓ FastText model downloaded successfully"
echo "  Location: $MODEL_FILE"
echo "  Size: $(du -h "$MODEL_FILE" | cut -f1)"
