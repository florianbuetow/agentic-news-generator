# Config Path Resolution: Remaining Domain Sections - Behavioral Specification

## Objective

Complete the path consolidation started in `config-path-resolution-specification.md`
by moving the remaining directory paths from `hallucination_detection` and
`article_compiler` config sections into the `paths:` section. After this work,
zero domain sections in `config.yaml` contain directory path values — all
directory paths are in `paths:` and accessed through `Config` getters that
return absolute `Path` objects.

## User Stories & Acceptance Criteria

### US-1: As a developer, I want the `hallucination_detection.output_dir` path moved to `paths:`, so that hallucination detection follows the same path convention as all other domains.

Acceptance Criteria:

  AC-1.1: A new key `data_hallucination_detection_output_dir` exists in the
           `paths:` section of `config.yaml` and `config.yaml.template`.

  AC-1.2: The `hallucination_detection` section in `config.yaml` no longer
           contains the `output_dir` key.

  AC-1.3: A new `Config` getter method `getHallucinationDetectionOutputDir()`
           exists, returns an absolute `Path`, and uses `_resolve_path`.

  AC-1.4: The `PathsConfig` Pydantic model has a corresponding
           `data_hallucination_detection_output_dir` field with
           `min_length=1`.

  AC-1.5: `scripts/transcript-hallucination-detection.py` uses
           `config.getHallucinationDetectionOutputDir()` instead of reading
           `hallucination_config["output_dir"]` and manually joining with
           `data_dir`.

  AC-1.6: Any other script or test that reads `hallucination_detection.output_dir`
           is updated to use the new getter.

### US-2: As a developer, I want the `article_compiler.input_dir` and `article_compiler.output_file` paths moved to `paths:`, so that the article compiler follows the same path convention as all other domains.

Acceptance Criteria:

  AC-2.1: Two new keys exist in the `paths:` section:
           - `data_article_compiler_input_dir`
           - `data_article_compiler_output_file`

  AC-2.2: The `article_compiler` section in `config.yaml` no longer contains
           `input_dir` or `output_file` keys.

  AC-2.3: The `ArticleCompilerConfig` Pydantic model no longer has `input_dir`
           or `output_file` fields.

  AC-2.4: Two new `Config` getter methods exist:
           - `getArticleCompilerInputDir()` — returns absolute `Path`
           - `getArticleCompilerOutputFile()` — returns absolute `Path`
           Both use `_resolve_path`.

  AC-2.5: The `PathsConfig` Pydantic model has corresponding fields with
           `min_length=1`.

  AC-2.6: `scripts/compile_articles.py` uses the new getters instead of
           `Path(compiler_config.input_dir)` and `Path(compiler_config.output_file)`.

  AC-2.7: `src/processing/article_compiler.py` uses the new getters if it
           accesses these paths, or receives them as constructor parameters
           from a caller that uses getters.

### US-3: As a developer, I want the structural convention test TS-14 to pass, confirming zero directory paths remain in any domain config section.

Acceptance Criteria:

  AC-3.1: The existing test `test_no_directory_paths_in_domain_sections` in
           `tests/test_config_path_conventions.py` passes after the changes,
           confirming that `hallucination_detection` and `article_compiler`
           sections contain no directory path values.

  AC-3.2: `just ci-quiet` passes with zero failures.

## Constraints

  C-1: Same resolution rules as the parent spec: absolute paths (starting with
       `/`) returned as-is; relative paths resolved against project root
       (`config_path.parent.parent`). Trailing slashes stripped.

  C-2: Breaking changes to config.yaml structure are acceptable (per CLAUDE.md).

  C-3: New `paths:` keys follow the `data_<domain>_<purpose>_dir` naming
       convention. Exception: `data_article_compiler_output_file` uses `_file`
       suffix since it points to a file, not a directory.

  C-4: The `hallucination_detection` section is not backed by a Pydantic model —
       it is read via `self._data["hallucination_detection"]` dict access. The
       `output_dir` key removal only affects YAML and the script that reads it.

## Edge Cases

  EC-1: `data_article_compiler_output_file` points to a file path, not a
        directory. The same `_resolve_path` logic applies — if relative, resolve
        against project root. The `_file` suffix in the key name distinguishes
        it from `_dir` keys.

  EC-2: `hallucination_detection.output_dir` was previously documented as
        "relative to data_dir" (config.yaml comment, line 68). Under the new
        convention it becomes relative to project root. The absolute value in
        `config.yaml` must be updated accordingly, or the relative value must
        be adjusted to be relative to project root instead of data_dir.

## Non-Goals

  NG-1: This spec does NOT move non-path config from `hallucination_detection`
        (SVM coefficients, window settings) or `article_compiler` (paragraph
        counts, image settings, link config). Only directory/file path values
        move.

  NG-2: This spec does NOT create a Pydantic model for `hallucination_detection`.
        It remains dict-accessed. Only the `output_dir` key is removed from it.

## Open Questions

None.
