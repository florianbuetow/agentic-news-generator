"""Structural tests verifying config path conventions across the codebase."""

import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class TestConfigYamlPathFormatting:
    """TS-17: Path values in config.yaml follow formatting rules."""

    def test_no_trailing_slashes_in_paths(self) -> None:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        paths = config["paths"]
        violations = [k for k, v in paths.items() if isinstance(v, str) and v.endswith("/")]
        assert violations == [], f"Paths with trailing slashes: {violations}"

    def test_no_dot_slash_prefix_in_paths(self) -> None:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        paths = config["paths"]
        violations = [k for k, v in paths.items() if isinstance(v, str) and v.startswith("./")]
        assert violations == [], f"Paths with ./ prefix: {violations}"

    def test_paths_start_with_slash_or_letter(self) -> None:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        paths = config["paths"]
        violations = []
        for k, v in paths.items():
            if isinstance(v, str) and v and not v[0].isalpha() and not v[0].isdigit() and v[0] != "/":
                violations.append(k)
        assert violations == [], f"Paths with invalid prefix: {violations}"


class TestScriptsUseConfigGetters:
    """TS-16: Scripts use Config getters without manual path construction."""

    def test_no_project_root_prefix_on_config_getters(self) -> None:
        violations = []
        for py_file in SCRIPTS_DIR.glob("*.py"):
            content = py_file.read_text()
            if re.search(r"(project_root|base_dir)\s*/\s*config\.get", content):
                violations.append(py_file.name)
        assert violations == [], f"Scripts with project_root/base_dir prefix on config getters: {violations}"

    def test_no_manual_data_dir_joins(self) -> None:
        violations = []
        for py_file in SCRIPTS_DIR.glob("*.py"):
            content = py_file.read_text()
            if re.search(r"config\.getDataDir\(\)\s*/\s*td_config\.", content):
                violations.append(py_file.name)
            if re.search(r"data_dir\s*/\s*td_config\.", content):
                violations.append(py_file.name)
        assert violations == [], f"Scripts with manual data_dir / td_config joins: {violations}"

    def test_no_direct_config_data_access(self) -> None:
        violations = []
        for py_file in SCRIPTS_DIR.glob("*.py"):
            content = py_file.read_text()
            if "config._data" in content:
                violations.append(py_file.name)
        assert violations == [], f"Scripts accessing config._data directly: {violations}"
