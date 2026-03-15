# Config Path Resolution: Remaining Domain Sections - Test Implementation Specification

## Test Framework & Conventions

Same as parent spec: pytest, plain `assert`, `_write_config()` helper with
`tmp_path` fixture, `GETTER_REGISTRY` for parameterized tests.

## Test Structure

**No new test files.** All changes go into existing files:

- `tests/test_config_path_resolution.py` — add new getters to `GETTER_REGISTRY`
  and `_default_paths()`, add `TestRemainingDomainGetters` class
- `tests/test_config_path_conventions.py` — update structural tests if needed
  (the existing TS-14 test should already cover the new sections)

## Test Scenario Mapping

### TestRemainingDomainGetters (new class in test_config_path_resolution.py)

| Scenario | Test Function | Setup | Action | Assertion |
|----------|---------------|-------|--------|-----------|
| TS-1 | `test_hallucination_detection_output_dir_absolute` | `_write_config(tmp_path, {"data_hallucination_detection_output_dir": "/Volumes/data/hallucinations"})` | `config.getHallucinationDetectionOutputDir()` | `result == Path("/Volumes/data/hallucinations")` and `result.is_absolute()` |
| TS-2 | `test_hallucination_detection_output_dir_relative` | `_write_config(tmp_path, {"data_hallucination_detection_output_dir": "data/downloads/transcripts-hallucinations"})` | `config.getHallucinationDetectionOutputDir()` | `result == tmp_path / "data" / "downloads" / "transcripts-hallucinations"` and `result.is_absolute()` |
| TS-3 | `test_article_compiler_input_dir_relative` | `_write_config(tmp_path, {"data_article_compiler_input_dir": "data/input/newspaper/articles"})` | `config.getArticleCompilerInputDir()` | `result == tmp_path / "data" / "input" / "newspaper" / "articles"` and `result.is_absolute()` |
| TS-4 | `test_article_compiler_output_file_relative` | `_write_config(tmp_path, {"data_article_compiler_output_file": "data/input/newspaper/articles.js"})` | `config.getArticleCompilerOutputFile()` | `result == tmp_path / "data" / "input" / "newspaper" / "articles.js"` and `result.is_absolute()` |
| TS-5 | `test_article_compiler_output_file_absolute` | `_write_config(tmp_path, {"data_article_compiler_output_file": "/Volumes/data/newspaper/articles.js"})` | `config.getArticleCompilerOutputFile()` | `result == Path("/Volumes/data/newspaper/articles.js")` |

### GETTER_REGISTRY updates

Add to `GETTER_REGISTRY` in `test_config_path_resolution.py`:

```python
("getHallucinationDetectionOutputDir", "data_hallucination_detection_output_dir"),
("getArticleCompilerInputDir", "data_article_compiler_input_dir"),
("getArticleCompilerOutputFile", "data_article_compiler_output_file"),
```

This automatically includes TS-6 (parameterized tests TS-18/TS-19 exercise them).

### _default_paths() updates

Add to `_default_paths()`:

```python
"data_hallucination_detection_output_dir": "data/downloads/transcripts-hallucinations",
"data_article_compiler_input_dir": "data/input/newspaper/articles",
"data_article_compiler_output_file": "data/input/newspaper/articles.js",
```

### Structural tests (TS-7, TS-8)

The existing `test_no_directory_paths_in_domain_sections` (TS-7) already
inspects `hallucination_detection` and `article_compiler` sections. It will
pass once the path values are removed from those sections.

For TS-8, update `test_no_manual_data_dir_joins` or add a new assertion in
`TestScriptsUseConfigGetters` to search for `compiler_config.input_dir`,
`compiler_config.output_file`, and `hallucination_config.*output_dir`.

### ArticleCompilerConfig validation (TS-9)

Add to an existing test class or create inline:

```python
def test_article_compiler_config_has_no_path_fields(self, tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    # Add article_compiler section to config (without input_dir/output_file)
    # Verify it loads and the model has no input_dir or output_file attributes
```

This test validates AC-2.3 — the Pydantic model no longer has path fields.

## Fixtures & Test Data

Same `_write_config()` helper as parent spec. Updated `_default_paths()` with
3 new keys. No new fixtures needed.

## Alignment Check

**Full alignment.** Every test scenario (TS-1 through TS-9) is mapped to a
test function. The parameterized tests (TS-18/TS-19 from parent spec) will
automatically cover the new getters via `GETTER_REGISTRY`.

**Initial failure expectation:** TS-1 through TS-5 will fail because the new
getters and PathsConfig fields don't exist yet. TS-7 will fail because
config.yaml still has path values in domain sections. TS-8 will fail because
scripts still use the old pattern.
