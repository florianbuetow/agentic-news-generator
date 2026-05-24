#!/bin/bash

# Download FastText language detection models
# This script downloads the compressed FastText model (lid.176.ftz) if it doesn't exist
# Model location is read from config.yaml (data_models_dir path)

# Get the absolute path to the project root
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

# Read data_models_dir from config.yaml using Python
cd "$project_root"
models_dir=$(uv run python -c "
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
fasttext_dir="$models_dir/fasttext"
mkdir -p "$fasttext_dir"

# Model file path
model_file="$fasttext_dir/lid.176.ftz"
model_url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Check if model already exists
if [ -f "$model_file" ]; then
    echo "✓ FastText model already exists at $model_file"
    exit 0
fi

# Download the model
echo "Downloading FastText language detection model (lid.176.ftz)..."
echo "  URL: $model_url"
echo "  Destination: $model_file"

# Download with curl (shows progress)
if command -v curl &> /dev/null; then
    curl -L "$model_url" -o "$model_file" --progress-bar
    download_status=$?
elif command -v wget &> /dev/null; then
    wget "$model_url" -O "$model_file" --show-progress
    download_status=$?
else
    echo "Error: Neither curl nor wget is available. Please install one of them."
    exit 1
fi

# Validate download succeeded
if [ $download_status -ne 0 ]; then
    echo "Error: Failed to download FastText model"
    rm -f "$model_file"  # Clean up partial download
    exit 1
fi

# Verify file exists and has content
if [ ! -f "$model_file" ] || [ ! -s "$model_file" ]; then
    echo "Error: Downloaded file is missing or empty"
    rm -f "$model_file"
    exit 1
fi

echo "✓ FastText model downloaded successfully"
echo "  Location: $model_file"
echo "  Size: $(du -h "$model_file" | cut -f1)"
