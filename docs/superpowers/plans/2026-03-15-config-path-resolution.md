# Config Path Resolution Implementation Plan

> **For agentic workers:** REQUIRED: Read `CLAUDE.md` first. Then read the spec at `docs/specs/config-path-resolution-specification.md` and the test spec at `docs/specs/config-path-resolution-test-specification.md`. Use `uv run` for all Python execution. Never use `python` directly. Never use `git add -A`. Never include AI attribution in commits.

**Goal:** Make all Config path getters resolve relative paths against the project root (`config_path.parent.parent`) and return absolute `Path` objects. Move all directory paths to the `paths:` section. Eliminate manual path construction in scripts.

**Architecture:** Add a private `_resolve_path(raw: str) -> Path` method to `Config` that checks if the raw string starts with `/` (absolute, return as-is) or not (relative, join with project root). All existing getters call this method. New domain-specific path keys are added to `PathsConfig` and `paths:` in config.yaml. Domain config sections lose their directory path fields, keeping only sub-path fragments. Scripts are updated to use getters exclusively.

**Tech Stack:** Python 3.12, Pydantic, pytest, pathlib, uv

---

## Chunk 1: Core Path Resolution + Tests

### Task 1: Add `_resolve_path` method and update all existing getters

**Files:**
- Modify: `src/config.py` (lines 505-520 for `__init__`, lines 832-978 for getters)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_path_resolution.py`:

```python
"""Tests for Config path resolution: absolute vs relative paths."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config


def _default_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Build a valid paths dict with all required keys. Override specific keys."""
    base = {
        "data_dir": "data",
        "data_models_dir": "data/models",
        "data_downloads_dir": "data/downloads",
        "data_downloads_videos_dir": "data/downloads/videos",
        "data_downloads_transcripts_dir": "data/downloads/transcripts",
        "data_downloads_transcripts_hallucinations_dir": "data/downloads/transcripts-hallucinations",
        "data_downloads_transcripts_cleaned_dir": "data/downloads/transcripts_cleaned",
        "data_transcripts_topics_dir": "data/downloads/transcripts-topics",
        "data_downloads_audio_dir": "data/downloads/audio",
        "data_downloads_metadata_dir": "data/downloads/metadata",
        "data_output_dir": "data/output",
        "data_input_dir": "data/input",
        "data_temp_dir": "data/temp",
        "data_archive_dir": "data/archive",
        "data_archive_videos_dir": "data/archive/videos",
        "data_logs_dir": "data/logs",
        "data_output_articles_dir": "data/output/articles",
        "data_articles_input_dir": "data/articles/input",
        "reports_dir": "reports",
    }
    if overrides:
        base.update(overrides)
    return base


def _write_config(tmp_path: Path, path_overrides: dict[str, str] | None = None) -> Config:
    """Write a minimal config.yaml at tmp_path/config/config.yaml and return Config.

    Project root = tmp_path (since config is at tmp_path/config/config.yaml,
    project_root = config_path.parent.parent = tmp_path).
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "paths": _default_paths(path_overrides),
        "channels": [],
        "defaults": {
            "encoding_name": "o200k_base",
            "repetition_min_k": 1,
            "repetition_min_repetitions": 5,
            "detect_min_k": 3,
        },
    }

    config_path = config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return Config(config_path)


# Registry: (getter_method_name, paths_key)
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
]


class TestAbsolutePathResolution:
    """TS-1, TS-4b, TS-10, TS-12: Absolute paths returned as-is."""

    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/Volumes/data/project"})
        result = config.getDataDir()
        assert result == Path("/Volumes/data/project")
        assert result.is_absolute()

    def test_absolute_path_trailing_slash_stripped(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/Volumes/data/"})
        result = config.getDataDir()
        assert result == Path("/Volumes/data")
        assert not str(result).endswith("/")

    def test_root_path_is_valid_absolute(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/"})
        result = config.getDataDir()
        assert result == Path("/")
        assert result.is_absolute()

    def test_nonexistent_path_no_error_at_load(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/nonexistent/path"})
        result = config.getDataDir()
        assert result == Path("/nonexistent/path")


class TestRelativePathResolution:
    """TS-2, TS-3, TS-4: Relative paths resolved via project root."""

    def test_relative_path_resolved_via_project_root(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data"})
        result = config.getDataDir()
        assert result == tmp_path / "data"
        assert result.is_absolute()

    def test_project_root_derived_from_config_location(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data/output"})
        result = config.getDataDir()
        assert result == tmp_path / "data" / "output"

    def test_relative_path_trailing_slash_stripped(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "data/output/"})
        result = config.getDataDir()
        assert result == tmp_path / "data" / "output"
        assert not str(result).endswith("/")


class TestProjectRootDerivation:
    """TS-6, TS-11: Project root from config file location."""

    def test_project_root_independent_of_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _write_config(tmp_path, {"data_dir": "data"})
        monkeypatch.chdir("/tmp")
        result = config.getDataDir()
        assert result == tmp_path / "data"

    def test_config_from_nonstandard_location(self, tmp_path: Path) -> None:
        """Config at tmp_path/test_config.yaml -> project_root = tmp_path.parent."""
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            "paths": _default_paths({"data_dir": "mydata"}),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        config = Config(config_path)
        # project_root = config_path.parent.parent = tmp_path.parent
        expected = tmp_path.parent / "mydata"
        assert config.getDataDir() == expected


class TestSchemaValidation:
    """TS-7, TS-9: Pydantic validation for paths."""

    def test_unknown_path_key_rejected(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        paths = _default_paths()
        paths["nonexistent_dir"] = "foo"
        config_data = {
            "paths": paths,
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        with pytest.raises(ValueError, match="nonexistent_dir|extra"):
            Config(config_path)

    def test_empty_string_path_rejected(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_data = {
            "paths": _default_paths({"data_dir": ""}),
            "channels": [],
            "defaults": {
                "encoding_name": "o200k_base",
                "repetition_min_k": 1,
                "repetition_min_repetitions": 5,
                "detect_min_k": 3,
            },
        }
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        with pytest.raises(ValueError):
            Config(config_path)


class TestEdgeCases:
    """TS-8, TS-13: Edge cases for path resolution."""

    def test_relative_path_with_parent_segments(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "../sibling/data"})
        result = config.getDataDir()
        assert result == tmp_path / ".." / "sibling" / "data"
        assert result.is_absolute()

    def test_symlinks_not_resolved(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_dir": "/some/symlinked/path"})
        result = config.getDataDir()
        assert str(result) == "/some/symlinked/path"


class TestExhaustiveGetterCoverage:
    """TS-18, TS-19, TS-20: Every getter works with absolute and relative values."""

    @pytest.mark.parametrize("getter_name,paths_key", GETTER_REGISTRY)
    def test_getter_returns_absolute_with_absolute_value(
        self, tmp_path: Path, getter_name: str, paths_key: str,
    ) -> None:
        config = _write_config(tmp_path, {paths_key: f"/abs/{paths_key}"})
        result = getattr(config, getter_name)()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result == Path(f"/abs/{paths_key}")

    @pytest.mark.parametrize("getter_name,paths_key", GETTER_REGISTRY)
    def test_getter_returns_absolute_with_relative_value(
        self, tmp_path: Path, getter_name: str, paths_key: str,
    ) -> None:
        config = _write_config(tmp_path, {paths_key: f"rel/{paths_key}"})
        result = getattr(config, getter_name)()
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert result == tmp_path / "rel" / paths_key

    def test_mixed_absolute_and_relative_in_same_config(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {
            "data_dir": "/absolute/data",
            "data_models_dir": "data/models",
            "data_output_dir": "/absolute/output",
            "reports_dir": "reports",
        })
        assert config.getDataDir() == Path("/absolute/data")
        assert config.getDataModelsDir() == tmp_path / "data" / "models"
        assert config.getDataOutputDir() == Path("/absolute/output")
        assert config.getReportsDir() == tmp_path / "reports"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_path_resolution.py -v`
Expected: FAIL — getters currently return `Path(raw_string)` without resolution

- [ ] **Step 3: Implement `_resolve_path` and `_project_root` in Config**

In `src/config.py`, add to `Config.__init__` (after line 520):

```python
self._project_root = self.config_path.parent.parent
```

Add a private method to `Config` (before the getters, around line 828):

```python
def _resolve_path(self, raw: str) -> Path:
    """Resolve a config path value to an absolute Path.

    If the value starts with '/', it is treated as absolute.
    Otherwise, it is resolved relative to the project root
    (config_path.parent.parent).
    """
    stripped = raw.rstrip("/")
    if stripped.startswith("/"):
        return Path(stripped)
    return self._project_root / stripped
```

Then update **every** getter from `return Path(self._paths.xxx)` to `return self._resolve_path(self._paths.xxx)`. There are 19 getters to update (lines 832-978). Example:

```python
def getDataDir(self) -> Path:
    """Get the data directory path."""
    return self._resolve_path(self._paths.data_dir)
```

Apply the same pattern to all 19 getters: `getDataModelsDir`, `getDataDownloadsDir`, `getDataDownloadsVideosDir`, `getDataDownloadsTranscriptsDir`, `getDataDownloadsTranscriptsHallucinationsDir`, `getDataDownloadsTranscriptsCleanedDir`, `getDataTranscriptsTopicsDir`, `getDataDownloadsAudioDir`, `getDataDownloadsMetadataDir`, `getDataOutputDir`, `getDataInputDir`, `getDataTempDir`, `getDataArchiveDir`, `getDataArchiveVideosDir`, `getDataLogsDir`, `getDataOutputArticlesDir`, `getDataArticlesInputDir`, `getReportsDir`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_path_resolution.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: Some existing tests may fail because they use `./data/` style paths and assert raw string equality. That is expected and will be fixed in Task 2.

- [ ] **Step 6: Commit**

```bash
git add src/config.py tests/test_config_path_resolution.py
git commit -m "feat(config): add path resolution — relative paths resolved via project root"
```

---

### Task 2: Fix existing tests for new resolution behavior

**Files:**
- Modify: `tests/test_config_data_dir.py`
- Modify: `tests/test_config.py`
- Modify: any other test files that break from Task 1

- [ ] **Step 1: Run full test suite to identify failures**

Run: `uv run pytest tests/ -v --ignore=tests/e2e 2>&1 | head -100`
Note which tests fail and why.

- [ ] **Step 2: Update test helpers**

In `tests/test_config_data_dir.py` and `tests/test_config.py`, update `get_valid_paths_config()` to remove `./` prefixes and trailing slashes. Change assertions from raw string equality to Path-based equality or remove assertions that test raw string values.

The key change: tests that assert `paths.data_dir == "./data/"` must now understand that the Pydantic model stores raw strings, but the Config getter returns resolved Paths. Tests should either:
- Test the Pydantic model directly (raw strings are still stored)
- Test via Config getters (which now resolve)

- [ ] **Step 3: Run full test suite to verify all pass**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_config_data_dir.py tests/test_config.py
git commit -m "test: update existing config tests for path resolution behavior"
```

---

## Chunk 2: Normalize config.yaml + Move Domain Paths

### Task 3: Normalize path formatting in config.yaml and config.yaml.template

**Files:**
- Modify: `config/config.yaml` (lines 1-24, paths section)
- Modify: `config/config.yaml.template`

- [ ] **Step 1: Clean up config.yaml paths section**

Remove all trailing slashes and `./` prefixes from every value in the `paths:` section. The absolute paths stay absolute; just strip trailing `/`.

Before:
```yaml
data_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/
data_models_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/models/
data_downloads_videos_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/downloads/videos/
data_output_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/output/
data_input_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/input/
```

After:
```yaml
data_dir: /Volumes/2TB/data/projects/agentic-news-generator/data
data_models_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/models
data_downloads_videos_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/downloads/videos
data_output_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/output
data_input_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/input
```

- [ ] **Step 2: Update config.yaml.template to use relative paths**

Change the template to use relative paths (no `./` prefix) as portable examples.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add config/config.yaml config/config.yaml.template
git commit -m "chore(config): normalize path formatting — remove trailing slashes"
```

---

### Task 4: Add domain-specific path keys to PathsConfig and config.yaml

**Files:**
- Modify: `src/config.py` (PathsConfig model, ~lines 477-502)
- Modify: `config/config.yaml` (paths section)
- Modify: `config/config.yaml.template` (paths section)

The following paths currently live in domain sections and must move to `paths:`:

| Current location | New paths: key |
|---|---|
| `article_generation.editor.output.final_articles_dir` | `data_article_generation_output_dir` |
| `article_generation.editor.output.run_artifacts_dir` | `data_article_generation_artifacts_dir` |
| `article_generation.knowledge_base.data_dir` | `data_article_generation_kb_dir` |
| `article_generation.knowledge_base.index_dir` | `data_article_generation_kb_index_dir` |
| `article_generation.institutional_memory.data_dir` | `data_article_generation_institutional_memory_dir` |
| `article_generation.editor.prompts.root_dir` | `data_article_generation_prompts_dir` |
| `topic_detection.output_dir` | `data_topic_detection_output_dir` |
| `topic_detection.taxonomy.acm_ccs_2012_xml_path` (dir part) | `data_topic_detection_taxonomies_dir` |
| `topic_detection.taxonomy.cache_dir` (dir part) | `data_topic_detection_taxonomy_cache_dir` |

- [ ] **Step 1: Write failing tests for new path keys**

Add to `tests/test_config_path_resolution.py`:

```python
class TestDomainSpecificGetters:
    """TS-21 through TS-25: New domain-specific getters."""

    def test_topic_detection_output_dir_relative(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_topic_detection_output_dir": "data/output/topics"})
        result = config.getTopicDetectionOutputDir()
        assert result == tmp_path / "data" / "output" / "topics"
        assert result.is_absolute()

    def test_article_generation_output_dir_absolute(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_article_generation_output_dir": "/Volumes/data/output/articles"})
        result = config.getArticleGenerationOutputDir()
        assert result == Path("/Volumes/data/output/articles")

    def test_article_generation_artifacts_dir_relative(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_article_generation_artifacts_dir": "data/output/article_editor_runs"})
        result = config.getArticleGenerationArtifactsDir()
        assert result == tmp_path / "data" / "output" / "article_editor_runs"

    def test_article_generation_kb_dir_relative(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_article_generation_kb_dir": "data/knowledgebase"})
        result = config.getArticleGenerationKbDir()
        assert result == tmp_path / "data" / "knowledgebase"

    def test_topic_detection_taxonomies_dir_relative(self, tmp_path: Path) -> None:
        config = _write_config(tmp_path, {"data_topic_detection_taxonomies_dir": "data/input/taxonomies"})
        result = config.getTopicDetectionTaxonomiesDir()
        assert result == tmp_path / "data" / "input" / "taxonomies"
```

Update `_default_paths()` to include the new keys with default relative values.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_path_resolution.py::TestDomainSpecificGetters -v`
Expected: FAIL — new keys and getters don't exist

- [ ] **Step 3: Add new fields to PathsConfig**

In `src/config.py`, add to `PathsConfig`:

```python
data_article_generation_output_dir: str = Field(..., description="Article generation final output directory", min_length=1)
data_article_generation_artifacts_dir: str = Field(..., description="Article generation run artifacts directory", min_length=1)
data_article_generation_kb_dir: str = Field(..., description="Knowledge base data directory", min_length=1)
data_article_generation_kb_index_dir: str = Field(..., description="Knowledge base index directory", min_length=1)
data_article_generation_institutional_memory_dir: str = Field(..., description="Institutional memory directory", min_length=1)
data_article_generation_prompts_dir: str = Field(..., description="Article generation prompts directory", min_length=1)
data_topic_detection_output_dir: str = Field(..., description="Topic detection output directory", min_length=1)
data_topic_detection_taxonomies_dir: str = Field(..., description="Topic detection taxonomies directory", min_length=1)
data_topic_detection_taxonomy_cache_dir: str = Field(..., description="Topic detection taxonomy cache directory", min_length=1)
```

- [ ] **Step 4: Add corresponding getters to Config**

```python
def getArticleGenerationOutputDir(self) -> Path:
    """Get article generation final output directory."""
    return self._resolve_path(self._paths.data_article_generation_output_dir)

def getArticleGenerationArtifactsDir(self) -> Path:
    """Get article generation run artifacts directory."""
    return self._resolve_path(self._paths.data_article_generation_artifacts_dir)

def getArticleGenerationKbDir(self) -> Path:
    """Get knowledge base data directory."""
    return self._resolve_path(self._paths.data_article_generation_kb_dir)

def getArticleGenerationKbIndexDir(self) -> Path:
    """Get knowledge base index directory."""
    return self._resolve_path(self._paths.data_article_generation_kb_index_dir)

def getArticleGenerationInstitutionalMemoryDir(self) -> Path:
    """Get institutional memory directory."""
    return self._resolve_path(self._paths.data_article_generation_institutional_memory_dir)

def getArticleGenerationPromptsDir(self) -> Path:
    """Get article generation prompts directory."""
    return self._resolve_path(self._paths.data_article_generation_prompts_dir)

def getTopicDetectionOutputDir(self) -> Path:
    """Get topic detection output directory."""
    return self._resolve_path(self._paths.data_topic_detection_output_dir)

def getTopicDetectionTaxonomiesDir(self) -> Path:
    """Get topic detection taxonomies directory."""
    return self._resolve_path(self._paths.data_topic_detection_taxonomies_dir)

def getTopicDetectionTaxonomyCacheDir(self) -> Path:
    """Get topic detection taxonomy cache directory."""
    return self._resolve_path(self._paths.data_topic_detection_taxonomy_cache_dir)
```

- [ ] **Step 5: Add new keys to config.yaml and config.yaml.template**

Add to the `paths:` section of `config/config.yaml`:

```yaml
  # Article generation paths
  data_article_generation_output_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/output/articles
  data_article_generation_artifacts_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/output/article_editor_runs
  data_article_generation_kb_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/knowledgebase
  data_article_generation_kb_index_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/knowledgebase_index
  data_article_generation_institutional_memory_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/institutional_memory
  data_article_generation_prompts_dir: prompts/article_editor

  # Topic detection paths
  data_topic_detection_output_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/output/topics
  data_topic_detection_taxonomies_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/input/taxonomies
  data_topic_detection_taxonomy_cache_dir: /Volumes/2TB/data/projects/agentic-news-generator/data/input/taxonomies/cache
```

Also update `config/config.yaml.template` with relative path equivalents.

- [ ] **Step 6: Update _default_paths in tests and GETTER_REGISTRY**

Add the new keys to `_default_paths()` and `GETTER_REGISTRY` in `tests/test_config_path_resolution.py`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/config.py config/config.yaml config/config.yaml.template tests/test_config_path_resolution.py
git commit -m "feat(config): add domain-specific path keys to paths section"
```

---

## Chunk 3: Remove Domain Paths from Config Sections + Update Consumers

### Task 5: Remove directory paths from article_generation config section

**Files:**
- Modify: `src/config.py` — `ArticleGenerationOutputConfig`, `KnowledgeBaseConfig`, `InstitutionalMemoryConfig`
- Modify: `src/agents/article_generation/agent.py` — `build_chief_editor_orchestrator()`
- Modify: `config/config.yaml` — article_generation section
- Modify: `config/config.yaml.template` — article_generation section

- [ ] **Step 1: Remove `final_articles_dir` and `run_artifacts_dir` from ArticleGenerationOutputConfig**

Delete these two fields from the Pydantic model. The model becomes empty or is removed entirely if no other fields remain.

- [ ] **Step 2: Remove `data_dir` and `index_dir` from KnowledgeBaseConfig**

These are now at `paths.data_article_generation_kb_dir` and `paths.data_article_generation_kb_index_dir`.

- [ ] **Step 3: Remove `data_dir` from InstitutionalMemoryConfig**

Now at `paths.data_article_generation_institutional_memory_dir`.

- [ ] **Step 4: Remove `root_dir` from ArticleGenerationPromptConfig**

Now at `paths.data_article_generation_prompts_dir`.

- [ ] **Step 5: Update `build_chief_editor_orchestrator()` in agent.py**

Replace:
```python
Path(article_generation_config.editor.output.final_articles_dir)
```
With:
```python
config.getArticleGenerationOutputDir()
```

Apply the same pattern for `run_artifacts_dir`, `institutional_memory.data_dir`, `knowledge_base.data_dir`, `knowledge_base.index_dir`, and `prompts.root_dir`.

- [ ] **Step 6: Remove the directory path values from config.yaml article_generation section**

Remove `final_articles_dir`, `run_artifacts_dir`, `data_dir`, `index_dir`, and `root_dir` from their respective sub-sections. Keep sub-path fragments like `fact_checking_subdir`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Fix any test failures from changed config structure (especially `tests/test_article_generation_config.py` and `tests/test_agent_factory.py`).

- [ ] **Step 8: Commit**

```bash
git add src/config.py src/agents/article_generation/agent.py config/config.yaml config/config.yaml.template tests/
git commit -m "refactor(config): move article generation dir paths to paths section"
```

---

### Task 6: Remove directory paths from topic_detection config section

**Files:**
- Modify: `src/config.py` — `TopicDetectionConfig`, `TopicDetectionTaxonomyConfig`
- Modify: `config/config.yaml` — topic_detection section
- Modify: `config/config.yaml.template`

- [ ] **Step 1: Remove `output_dir` from TopicDetectionConfig**

Now at `paths.data_topic_detection_output_dir`.

- [ ] **Step 2: Remove `acm_ccs_2012_xml_path` and `cache_dir` from TopicDetectionTaxonomyConfig**

Replace `acm_ccs_2012_xml_path` with just a filename field (`acm_ccs_2012_xml_file`). Replace `cache_dir` — it's now at `paths.data_topic_detection_taxonomy_cache_dir`.

- [ ] **Step 3: Remove these values from config.yaml topic_detection section**

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Fix any failures in `tests/test_topic_detection_config.py`, `tests/test_topic_tree_e2e.py`.

- [ ] **Step 5: Commit**

```bash
git add src/config.py config/config.yaml config/config.yaml.template tests/
git commit -m "refactor(config): move topic detection dir paths to paths section"
```

---

### Task 7: Update scripts to use new Config getters

**Files:**
- Modify: 9+ scripts in `scripts/` that currently do `config.getDataDir() / td_config.output_dir`
- Modify: 3 scripts that prepend `project_root /` or `base_dir /` to config getters

- [ ] **Step 1: Replace manual topic detection path joins in all scripts**

In each of these files, replace `config.getDataDir() / td_config.output_dir` with `config.getTopicDetectionOutputDir()`:

- `scripts/topic-tree.py` (line 364)
- `scripts/delete-old-topic-trees.py` (line 56)
- `scripts/extract-topics.py` (line 175)
- `scripts/status.py` (line 168)
- `scripts/visualize-embeddings.py` (line 558)
- `scripts/export-to-minirag.py` (line 352)
- `scripts/prepare-article-input-data.py` (line 151)
- `scripts/detect-boundaries.py` (line 186)
- `scripts/generate-embeddings.py` (line 226)

- [ ] **Step 2: Remove dead `project_root /` and `base_dir /` prefixes**

- `scripts/transcribe_audio.py` (lines 280-282): Remove `project_root /` prefix
- `scripts/transcript-hallucination-detection.py` (lines 233, 297, 303): Remove `Path(__file__).parent.parent /` prefix
- `scripts/transcript-hallucination-removal.py` (lines 76-79): Remove `base_dir /` prefix

- [ ] **Step 3: Update scripts/generate-articles.py for new config paths**

Replace any `article_generation_config.editor.output.*` path access with Config getters.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "refactor(scripts): use Config getters for all path access"
```

---

## Chunk 4: Structural Convention Tests

### Task 8: Write structural convention tests

**Files:**
- Create: `tests/test_config_path_conventions.py`

- [ ] **Step 1: Write the structural tests**

```python
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
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_config_path_conventions.py -v`
Expected: ALL PASS (if Tasks 3-7 are complete)

- [ ] **Step 3: Commit**

```bash
git add tests/test_config_path_conventions.py
git commit -m "test: add structural convention tests for config paths"
```

---

### Task 9: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: ALL PASS

- [ ] **Step 2: Verify no regressions with a spot-check**

Run: `uv run pytest tests/test_config.py tests/test_config_data_dir.py tests/test_article_generation_config.py tests/test_agent_factory.py tests/test_mock_agents.py tests/test_topic_detection_config.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit any remaining fixes**

Only if needed. Otherwise, the implementation is complete.
