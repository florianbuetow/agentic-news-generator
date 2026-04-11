"""Filter files for processing based on config-defined base directories."""

import json
from pathlib import Path
from typing import Any, cast

from src.config import Config


class FileProcessingFilter:
    """Skip files listed in the file filter configuration."""

    def __init__(self, config: Config) -> None:
        """Initialize the filter from config paths and the filter file."""
        config_paths = self.extract_config_paths(config)
        self._path_to_key_lookup = self.build_path_to_key_lookup(config_paths)
        filter_path = config.getConfigPath().parent / "filefilter.json"
        self._filter = self.load_filter_file(filter_path)

        invalid_keys = sorted(set(self._filter.keys()) - set(config_paths.keys()))
        if invalid_keys:
            valid_keys = ", ".join(sorted(config_paths.keys()))
            invalid_keys_str = ", ".join(invalid_keys)
            raise ValueError(f"Invalid filter keys in {filter_path}: {invalid_keys_str}. Valid config path keys: {valid_keys}")

    @staticmethod
    def extract_config_paths(config: Config) -> dict[str, str]:
        """Extract all configured path fields from the config."""
        raw_paths: dict[str, Any] = config.get_paths_config().model_dump()
        return {str(field_name): str(field_value) for field_name, field_value in raw_paths.items()}

    @staticmethod
    def build_path_to_key_lookup(config_paths: dict[str, str]) -> dict[str, str]:
        """Build a reverse lookup from resolved path string to config key."""
        return {str(Path(path_value).resolve()): config_key for config_key, path_value in config_paths.items()}

    @staticmethod
    def load_filter_file(filter_path: Path) -> dict[str, set[str]]:
        """Load the JSON filter file and normalize values to sets."""
        if not filter_path.exists():
            raise FileNotFoundError(f"Filter file not found: {filter_path}")

        with filter_path.open(encoding="utf-8") as filter_file:
            raw: object = json.load(filter_file)

        if not isinstance(raw, dict):
            raise ValueError(f"Filter file must contain a JSON object: {filter_path}")

        raw_filter = cast(dict[str, object], raw)

        loaded_filter: dict[str, set[str]] = {}
        for config_key, file_list_value in raw_filter.items():
            if not isinstance(file_list_value, list):
                raise ValueError(f"Filter file values must be lists of strings for key '{config_key}'")

            file_list = cast(list[object], file_list_value)
            file_names: list[str] = []
            for entry in file_list:
                if not isinstance(entry, str):
                    raise ValueError(f"Filter file values must be lists of strings for key '{config_key}'")
                file_names.append(entry)

            loaded_filter[config_key] = set(file_names)

        return loaded_filter

    def _resolve_config_key(self, base_dir: str) -> str:
        """Resolve a base directory path back to its config path key."""
        resolved_base_dir = str(Path(base_dir).resolve())
        config_key = self._path_to_key_lookup.get(resolved_base_dir)
        if config_key is None:
            known_paths = ", ".join(sorted(self._path_to_key_lookup.keys()))
            raise ValueError(f"Unknown base_dir: {base_dir}. Known config paths: {known_paths}")
        return config_key

    def should_skip_file(self, file_path: str, base_dir: str) -> bool:
        """Return whether the file path is listed for skipping under the base directory."""
        config_key = self._resolve_config_key(base_dir)
        relative_path = str(Path(file_path).resolve().relative_to(Path(base_dir).resolve()))
        if config_key not in self._filter:
            return False
        return relative_path in self._filter[config_key]
