#!/bin/bash

# Initialize all required directories from config.yaml
# This script is the single source of truth for directory creation

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read all directory paths from config.yaml using Python
cd "$PROJECT_ROOT"
uv run python -c "
import yaml
import os
from pathlib import Path

# Read config.yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get all directory paths from config
paths = config.get('paths', {})

# Create each directory listed in config.yaml
for key, path_str in paths.items():
    if key.endswith('_dir'):
        # Convert relative path to absolute and create directory
        path = Path(path_str).resolve()
        path.mkdir(parents=True, exist_ok=True)
        print(f'✓ Created: {path}')

# Create additional project-specific directories
additional_dirs = [
    'reports/coverage',
    'reports/security',
    'reports/pyright',
    'reports/deptry',
    '.cache',
]

for dir_path in additional_dirs:
    path = Path(dir_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    print(f'✓ Created: {path}')
"
