# Config Path Resolution - Test Implementation Specification

## Test Framework & Conventions

- **Framework:** pytest (project standard)
- **Assertion style:** plain `assert` statements
- **Config creation:** write a temp YAML file via `yaml.dump()`, load with `Config(path)`
- **Temp directories:** pytest `tmp_path` fixture
- **Pydantic validation errors:** caught with `pytest.raises(ValueError)` or
  `pytest.raises(ValidationError)`
- **No mocking of Config internals** — tests exercise the full `Config.__init__`
  and getter chain. Only the filesystem (config file location) is controlled.
- **Existing test pattern:** `tests/test_config_data_dir.py` uses
  `get_valid_paths_config()` helper returning a dict, then writes to a temp
  YAML file. We follow the same pattern.

## Test Structure

**File:** `tests/test_config_path_resolution.py`

All tests live in one file since they test a single concern (path resolution).
Tests are grouped into classes by test specification category:

| Class | Covers |
|-------|--------|
| `TestAbsolutePathResolution` | TS-1, TS-4b, TS-10, TS-12 |
| `TestRelativePathResolution` | TS-2, TS-3, TS-4 |
| `TestProjectRootDerivation` | TS-6, TS-11 |
| `TestSchemaValidation` | TS-7, TS-9 |
| `TestEdgeCases` | TS-8, TS-13 |
| `TestExhaustiveGetterCoverage` | TS-18, TS-19, TS-20 |
| `TestDomainSpecificGetters` | TS-5, TS-21, TS-22, TS-23, TS-24, TS-25 |

**Structural tests** (TS-14 through TS-17) are in a separate file:

**File:** `tests/test_config_path_conventions.py`

These are grep/static-analysis tests that scan `config.yaml` and `scripts/`
for convention violations. They do not load Config — they read files as text.

## Fixtures & Test Data

### Config Writer Helper

A module-level helper function writes a minimal valid `config.yaml` to a temp
directory and returns the `Config` object. All path keys can be overridden
per test.

```
def write_config(tmp_path: Path, path_overrides: dict[str, str] | None = None) -> Config:
    """Write a minimal config.yaml under tmp_path/config/ and return Config.

    The config file is placed at tmp_path/config/config.yaml so that
    project_root = tmp_path/config/../ = tmp_path.

    Args:
        tmp_path: pytest tmp_path fixture.
        path_overrides: dict of paths: keys to override (e.g., {"data_dir": "/abs/path"}).

    Returns:
        Loaded Config instance.
    """
```

This helper:
1. Creates `tmp_path/config/` directory
2. Builds a minimal valid config dict (channels=[], valid defaults, valid paths)
3. Merges `path_overrides` into the paths section
4. Writes YAML to `tmp_path/config/config.yaml`
5. Returns `Config(tmp_path / "config" / "config.yaml")`

The default paths dict uses relative values so the baseline exercises relative
path resolution. Tests that need absolute values override specific keys.

### Getter Registry

A list of `(getter_name, paths_key)` tuples used by parameterized tests in
`TestExhaustiveGetterCoverage`. This list is the single source of truth for
which getters exist and which config key they read from. It must be updated
whenever a getter is added or removed.

```
GETTER_REGISTRY: list[tuple[str, str]] = [
    ("getDataDir", "data_dir"),
    ("getDataModelsDir", "data_models_dir"),
    ("getDataDownloadsDir", "data_downloads_dir"),
    ("getDataDownloadsVideosDir", "data_downloads_videos_dir"),
    ("getDataDownloadsTranscriptsDir", "data_downloads_transcripts_dir"),
    ("getDataDownloadsTranscriptsHallucinationsDir", "data_downloads_transcripts_hallucinations_dir"),
    ("getDataDownloadsTranscriptsCleanedDir", "data_downloads_transcripts_cleaned_dir"),
    ("getDataTranscriptsTopicsDir", "data_transcripts_topics_dir"),
    ("getDataDownloadsAudioDir", "data_downloads_audio_dir"),
    ("getDataDownloadsMetadataDir", "data_downloads_metadata_dir"),
    ("getDataOutputDir", "data_output_dir"),
    ("getDataInputDir", "data_input_dir"),
    ("getDataTempDir", "data_temp_dir"),
    ("getDataArchiveDir", "data_archive_dir"),
    ("getDataArchiveVideosDir", "data_archive_videos_dir"),
    ("getDataLogsDir", "data_logs_dir"),
    ("getDataOutputArticlesDir", "data_output_articles_dir"),
    ("getDataArticlesInputDir", "data_articles_input_dir"),
    ("getReportsDir", "reports_dir"),
    # New domain-specific getters (added by this feature):
    # ("getTopicDetectionOutputDir", "data_topic_detection_output_dir"),
    # ("getArticleGenerationOutputDir", "data_article_generation_output_dir"),
    # ... etc — uncomment as getters are implemented
]
```

## Test Scenario Mapping

### TestAbsolutePathResolution

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-1 | `test_absolute_path_returned_as_is` | `write_config(tmp_path, {"data_dir": "/Volumes/data/project"})` | `config.getDataDir()` | `result == Path("/Volumes/data/project")` and `result.is_absolute()` |
| TS-4b | `test_absolute_path_trailing_slash_stripped` | `write_config(tmp_path, {"data_dir": "/Volumes/data/"})` | `config.getDataDir()` | `result == Path("/Volumes/data")` and `not str(result).endswith("/")` |
| TS-10 | `test_root_path_is_valid_absolute` | `write_config(tmp_path, {"data_dir": "/"})` | `config.getDataDir()` | `result == Path("/")` and `result.is_absolute()` |
| TS-12 | `test_nonexistent_path_no_error_at_load` | `write_config(tmp_path, {"data_dir": "/nonexistent/path"})` | `config.getDataDir()` | No error raised; `result == Path("/nonexistent/path")` |

### TestRelativePathResolution

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-2 | `test_relative_path_resolved_via_project_root` | `write_config(tmp_path, {"data_dir": "data"})` | `config.getDataDir()` | `result == tmp_path / "data"` and `result.is_absolute()` |
| TS-3 | `test_project_root_derived_from_config_location` | `write_config(tmp_path, {"data_dir": "data/output"})` | `config.getDataDir()` | `result == tmp_path / "data" / "output"` |
| TS-4 | `test_relative_path_trailing_slash_stripped` | `write_config(tmp_path, {"data_dir": "data/output/"})` | `config.getDataDir()` | `result == tmp_path / "data" / "output"` and `not str(result).endswith("/")` |

### TestProjectRootDerivation

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-6 | `test_project_root_independent_of_cwd` | `write_config(tmp_path, {"data_dir": "data"})`, then `os.chdir("/tmp")` (restored via fixture) | `config.getDataDir()` | `result == tmp_path / "data"`, not `/tmp/data` |
| TS-11 | `test_config_from_nonstandard_location` | Write config directly at `tmp_path / "test_config.yaml"` (no `/config/` subdirectory), set `data_dir` to `"mydata"` | `config.getDataDir()` | `result == tmp_path.parent / "mydata"` (project root = config_path.parent.parent = tmp_path's grandparent) |

**Note on TS-6:** The test must save and restore the original CWD. Use a
`monkeypatch.chdir()` or a try/finally block.

**Note on TS-11:** When the config file is at `tmp_path / "test_config.yaml"`,
the project root is `tmp_path.parent.parent`. The relative path `"mydata"` resolves
to `tmp_path.parent.parent / "mydata"`. The test verifies the path does NOT
incorporate the CWD or `tmp_path` itself, only the config file's grandparent.

### TestSchemaValidation

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-7 | `test_unknown_path_key_rejected` | Build paths dict with extra key `"nonexistent_dir": "foo"` | `Config(config_path)` | Raises `ValueError` with "nonexistent_dir" or "extra" in message |
| TS-9 | `test_empty_string_path_rejected` | `write_config(tmp_path, {"data_dir": ""})` | `Config(config_path)` | Raises `ValueError` with indication of empty/min_length |

### TestEdgeCases

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-8 | `test_relative_path_with_parent_segments` | `write_config(tmp_path, {"data_dir": "../sibling/data"})` | `config.getDataDir()` | `result == tmp_path / ".." / "sibling" / "data"` and `result.is_absolute()` |
| TS-13 | `test_symlinks_not_resolved` | `write_config(tmp_path, {"data_dir": "/some/symlinked/path"})` | `config.getDataDir()` | `str(result) == "/some/symlinked/path"` (no `.resolve()` applied) |

### TestExhaustiveGetterCoverage

These are **parameterized** tests using `@pytest.mark.parametrize` over the
`GETTER_REGISTRY` list.

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-18 | `test_getter_returns_absolute_with_absolute_value[<getter>]` | For each `(getter, key)`, `write_config(tmp_path, {key: "/abs/<key>"})` | `getattr(config, getter)()` | `result == Path("/abs/<key>")` and `result.is_absolute()` |
| TS-19 | `test_getter_returns_absolute_with_relative_value[<getter>]` | For each `(getter, key)`, `write_config(tmp_path, {key: "rel/<key>"})` | `getattr(config, getter)()` | `result == tmp_path / "rel" / key` and `result.is_absolute()` |
| TS-20 | `test_mixed_absolute_and_relative_in_same_config` | `write_config(tmp_path, {"data_dir": "/absolute/data", "data_models_dir": "data/models", "data_output_dir": "/absolute/output", "reports_dir": "reports"})` | Call all four getters | `getDataDir() == Path("/absolute/data")`, `getDataModelsDir() == tmp_path / "data/models"`, `getDataOutputDir() == Path("/absolute/output")`, `getReportsDir() == tmp_path / "reports"` |

### TestDomainSpecificGetters

These test the new getters that replace the manual `config.getDataDir() / td_config.output_dir`
pattern. The exact getter names depend on implementation, so the test function
names use descriptive behavior names.

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-5 | `test_sub_path_fragment_composed_by_getter` | Config with `data_article_generation_institutional_memory_dir: "data/im"` and `article_generation.institutional_memory.fact_checking_subdir: "fact_checking"` | Call the composed getter | `result == tmp_path / "data" / "im" / "fact_checking"` and `result.is_absolute()` |
| TS-21 | `test_topic_detection_output_dir_relative` | `data_topic_detection_output_dir: "data/output/topics"` | Getter call | `result == tmp_path / "data" / "output" / "topics"` |
| TS-22 | `test_article_generation_output_dir_absolute` | `data_article_generation_output_dir: "/Volumes/data/output/articles"` | Getter call | `result == Path("/Volumes/data/output/articles")` |
| TS-23 | `test_article_generation_artifacts_dir_relative` | `data_article_generation_artifacts_dir: "data/output/article_editor_runs"` | Getter call | `result == tmp_path / "data" / "output" / "article_editor_runs"` |
| TS-24 | `test_knowledge_base_dir_relative` | `data_article_generation_kb_dir: "data/knowledgebase"` | Getter call | `result == tmp_path / "data" / "knowledgebase"` |
| TS-25 | `test_topic_detection_taxonomies_dir_relative` | `data_topic_detection_taxonomies_dir: "data/input/taxonomies"` | Getter call | `result == tmp_path / "data" / "input" / "taxonomies"` |

### Structural Convention Tests (test_config_path_conventions.py)

These tests read files as text and do not load Config.

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-14 | `test_no_directory_paths_in_domain_sections` | Read `config/config.yaml` | Parse YAML, inspect `article_generation`, `topic_detection`, `hallucination_detection`, `article_compiler` sections for values starting with `/` | Zero absolute path values found in domain sections (excluding `api_base` and similar non-filesystem URLs) |
| TS-15 | `test_domain_path_keys_follow_naming_convention` | Read `config/config.yaml` | Collect all `paths:` keys that are not in the base set (`data_dir`, `data_models_dir`, `data_downloads_*`, `data_output_dir`, `data_input_dir`, `data_temp_dir`, `data_archive_*`, `data_logs_dir`, `reports_dir`) | Each remaining key matches `data_<domain>_*_dir` pattern |
| TS-16 | `test_scripts_use_config_getters_without_manual_construction` | Read all `scripts/*.py` files | Search for patterns: `project_root /` or `base_dir /` followed by `config.get`, `config._data[`, `config.getDataDir() / td_config.`, `config.getDataDir() / .*config.` | Zero matches found |
| TS-17 | `test_config_yaml_path_values_follow_formatting_rules` | Read `config/config.yaml` | Parse YAML, extract all values from `paths:` section | No value ends with `/`, no value starts with `./`, each value starts with `/` or a letter/digit |

## Alignment Check

**Full alignment.** Every test scenario (TS-1 through TS-25) is mapped to a
test function with setup, action, and assertion defined. No gaps.

**Initial failure expectation:**
- TS-1 through TS-4b, TS-7, TS-9, TS-10, TS-12, TS-13: These test existing
  getters but with the NEW resolution behavior (relative -> absolute). They will
  FAIL because `Config` currently returns `Path(raw_string)` without resolution.
- TS-2, TS-3, TS-4, TS-6, TS-8, TS-11: These specifically test relative path
  resolution, which does not exist yet. They will FAIL.
- TS-18, TS-19, TS-20: Parameterized over all getters. The relative-value
  variants (TS-19, parts of TS-20) will FAIL.
- TS-5, TS-21 through TS-25: These test new getters that do not exist yet.
  They will FAIL with `AttributeError`.
- TS-14 through TS-17: These are structural tests. They will FAIL because
  `config.yaml` still has the old structure with directory paths in domain
  sections.

**Design concerns:** None. All tests verify observable behavior (getter return
values and config file structure). No internal implementation details are tested.
