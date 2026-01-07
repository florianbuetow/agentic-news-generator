#!/usr/bin/env python3
"""Validate config.yaml matches config.yaml.template structure."""

import sys
from pathlib import Path

import yaml


def extract_keys(data: dict | None, prefix: str = "") -> set[str]:
    """Recursively extract all key paths from nested dict."""
    if data is None:
        return set()

    keys = set()
    for key, value in data.items():
        current_key = f"{prefix}.{key}" if prefix else key
        keys.add(current_key)
        if isinstance(value, dict):
            keys.update(extract_keys(value, current_key))
    return keys


def main() -> int:
    """Compare config.yaml and config.yaml.template structures."""
    # Load template (required)
    template_path = Path("config/config.yaml.template")
    if not template_path.exists():
        print(f"✗ ERROR: Template file not found: {template_path}")
        return 1

    template_data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    template_keys = extract_keys(template_data)

    # Load config (optional)
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("⚠ WARNING: config.yaml does not exist")
        return 0

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_keys = extract_keys(config_data)

    # Compare
    missing_in_config = template_keys - config_keys
    extra_in_config = config_keys - template_keys

    # Print results
    has_diff = False
    if missing_in_config:
        print("✗ Keys in template but missing in config.yaml:")
        for key in sorted(missing_in_config):
            print(f"  - {key}")
        has_diff = True

    if extra_in_config:
        print("✗ Keys in config.yaml but not in template:")
        for key in sorted(extra_in_config):
            print(f"  - {key}")
        has_diff = True

    if has_diff:
        return 1

    print(f"✓ Config structure in '{config_path}' matches template '{template_path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
