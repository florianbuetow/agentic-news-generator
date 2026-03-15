# Config Path Resolution - Behavioral Specification

## Objective

Establish a single, consistent mechanism for defining and resolving filesystem
paths across the entire project. All directory paths are defined in one place
(`paths:` section of `config.yaml`), resolved to absolute `Path` objects by
the `Config` class, and returned to callers through getter methods. No script
or module ever constructs, guesses, or prefixes a base directory on its own.

## User Stories & Acceptance Criteria

### US-1: As a developer, I want all config paths resolved to absolute `Path` objects, so that scripts never need to reason about relative vs absolute paths.

Acceptance Criteria:

  AC-1.1: Every getter method on `Config` that returns a directory path returns
           a `pathlib.Path` object that satisfies `path.is_absolute() == True`.

  AC-1.2: If a path value in `config.yaml` starts with `/`, `Config` treats it
           as an absolute path and returns it as-is (wrapped in `Path`).

  AC-1.3: If a path value in `config.yaml` does NOT start with `/`, `Config`
           resolves it relative to the **project root** and returns the resulting
           absolute path.

  AC-1.4: The project root is defined as `config_path.parent.parent` (the
           grandparent of the config file). For the standard layout
           `<project>/config/config.yaml`, this yields `<project>/`.

  AC-1.5: Path resolution strips any trailing `/` from the raw config value
           before constructing the `Path` object. A value `data/output/` and
           `data/output` both resolve to the same absolute path.

### US-2: As a developer, I want all directory paths defined in the `paths:` section, so that there is a single source of truth for every directory the project uses.

Acceptance Criteria:

  AC-2.1: The `paths:` section is the **only** place in `config.yaml` where
           directory paths are defined.

  AC-2.2: Domain-specific config sections (`article_generation`,
           `topic_detection`, `hallucination_detection`, `article_compiler`)
           must NOT contain directory path values that duplicate or override
           entries in `paths:`.

  AC-2.3: Domain-specific sections MAY contain **sub-path fragments** (e.g.,
           a subdirectory name like `fact_checking`) that are relative to the
           domain's base path defined in `paths:`. These sub-path fragments
           are not standalone paths and are not resolved by the path resolution
           mechanism in US-1. They are joined by the `Config` getter that
           serves the complete path.

  AC-2.4: Every domain-specific path key in `paths:` follows the naming
           convention `data_<domain>_<purpose>_dir` so that the key name
           identifies the domain it belongs to. Examples:
           - `data_article_generation_output_dir`
           - `data_article_generation_artifacts_dir`
           - `data_topic_detection_output_dir`
           - `data_topic_detection_taxonomies_dir`

### US-3: As a developer, I want scripts to get paths exclusively through `Config` getter methods, so that path construction logic is never duplicated across scripts.

Acceptance Criteria:

  AC-3.1: Every script in `scripts/` obtains directory paths by calling a
           method on the `Config` object (e.g., `config.getDataOutputDir()`).
           No script reads `config._data` directly for path values.

  AC-3.2: No script prepends `project_root /`, `base_dir /`,
           `Path(__file__).parent.parent /`, or any other base directory to
           a path returned by a `Config` getter. Config getters already return
           absolute paths.

  AC-3.3: No script manually joins `config.getDataDir() / td_config.output_dir`
           or similar. If a composed path is needed (base + sub-path), a
           dedicated `Config` getter provides the complete resolved path.

  AC-3.4: The only path a script may construct from `Config` data using the `/`
           operator is an **ad-hoc sub-path below a getter result** for
           script-specific purposes (e.g., `config.getDataOutputDir() / "language_analysis"`).
           The base directory itself always comes from a getter.

### US-4: As a developer, I want `config.yaml` path values to have consistent formatting, so that the config file is predictable and unambiguous.

Acceptance Criteria:

  AC-4.1: No path value in `config.yaml` ends with a trailing `/`.

  AC-4.2: No path value in `config.yaml` starts with `./`. A relative path
           is written without a leading dot (e.g., `data/output`, not
           `./data/output`).

  AC-4.3: If a path value in `config.yaml` is absolute, it starts with `/`.
           If relative, it starts with a letter or digit.

## Constraints

  C-1: Python 3.12, all paths use `pathlib.Path`.

  C-2: No environment variables are used for configuration (per CLAUDE.md).

  C-3: Project root derivation must be deterministic and not depend on the
       current working directory at runtime. It is always
       `config_path.parent.parent`.

  C-4: The `Config` class is the only component that performs path resolution.
       No other module, script, or test may implement its own resolution logic.

  C-5: Breaking changes to config.yaml structure are acceptable per project
       rules (no backwards compatibility).

  C-6: Existing `PathsConfig` Pydantic model must enforce `extra="forbid"` so
       that unknown keys are rejected.

## Edge Cases

  EC-1: A relative path value that contains `..` segments (e.g., `../data/output`)
        is resolved normally by `Path.resolve()` or `/` operator. The resulting
        absolute path may be outside the project root. This is valid.

  EC-2: A path value that is an empty string must be rejected by Pydantic
        validation (`min_length=1` on `Field`). This is the existing behavior
        and must be preserved.

  EC-3: A path value that is just `/` (root directory) is technically absolute
        and valid. No special handling needed.

  EC-4: On macOS, paths starting with `/Volumes/` are absolute. On Linux,
        paths starting with `/` are absolute. The check is simply
        `value.startswith("/")`.

  EC-5: If `config.yaml` is loaded from a non-standard location (e.g.,
        `/tmp/test_config.yaml`), the project root becomes `/tmp/`'s parent
        (`/`). Relative paths resolve against `/`. This is acceptable for
        testing but the caller is responsible for providing sensible paths
        in the config.

  EC-6: `config.yaml.template` must be updated to match the new `paths:`
        structure. Its values should use relative paths to serve as portable
        examples.

## Non-Goals

  NG-1: This feature does NOT add path existence validation at config load time.
        Paths are resolved but not checked for existence on disk. Existence
        checking is the caller's responsibility.

  NG-2: This feature does NOT change the Pydantic model types. Path values in
        config.yaml remain `str` fields in Pydantic models. The `Config` getter
        methods are what perform the resolution and return `Path` objects.

  NG-3: This feature does NOT introduce symbolic link resolution. `Path()` is
        used, not `Path.resolve()`. Symlinks are preserved.

  NG-4: This feature does NOT address Windows path support. Only POSIX paths
        (macOS/Linux) are in scope.

  NG-5: This feature does NOT consolidate ALL config sections. Only directory
        path values are moved to `paths:`. Non-path config (LLM settings,
        thresholds, flags) stays in its domain section.

## Open Questions

None. All design decisions have been resolved.
