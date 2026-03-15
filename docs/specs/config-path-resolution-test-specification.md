# Config Path Resolution - Test Specification

## Coverage Matrix

Note: TS-18 through TS-20 verify that **every** path key defined in the
`paths:` section is readable through a corresponding `Config` getter and
that the resolution logic (absolute vs relative) applies uniformly across
all keys — not just the one or two keys used in the focused scenarios above.

| Spec Requirement | Test Scenario(s) |
|------------------|------------------|
| AC-1.1: Getters return absolute Path | TS-1, TS-2, TS-18, TS-19, TS-20 |
| AC-1.2: Absolute path returned as-is | TS-1, TS-18, TS-20, TS-22 |
| AC-1.3: Relative path resolved via project root | TS-2, TS-19, TS-20, TS-21, TS-23, TS-24, TS-25 |
| AC-1.4: Project root is config_path.parent.parent | TS-3 |
| AC-1.5: Trailing slash stripped | TS-4 |
| AC-2.1: paths: is single source of truth | TS-14 (structural, verified by review) |
| AC-2.2: Domain sections have no directory paths | TS-14 (structural, verified by review) |
| AC-2.3: Sub-path fragments joined by getter | TS-5, TS-21 through TS-25 |
| AC-2.4: Domain keys follow naming convention | TS-15 (structural, verified by review) |
| AC-3.1: Scripts use Config getters only | TS-16 (structural, verified by review) |
| AC-3.2: No base-dir prefix in scripts | TS-16 (structural, verified by review) |
| AC-3.3: No manual data_dir / sub-path joins | TS-16 (structural, verified by review) |
| AC-3.4: Ad-hoc sub-paths below getter result OK | (convention, not testable in isolation) |
| AC-4.1: No trailing slash in config values | TS-17 (structural, verified by review) |
| AC-4.2: No ./ prefix in config values | TS-17 (structural, verified by review) |
| AC-4.3: Absolute starts with /, relative with letter/digit | TS-17 (structural, verified by review) |
| C-3: Project root independent of CWD | TS-6 |
| C-6: PathsConfig rejects unknown keys | TS-7 |
| EC-1: Relative path with .. segments | TS-8 |
| EC-2: Empty string path rejected | TS-9 |
| EC-3: Root path "/" is valid absolute | TS-10 |
| EC-5: Config from non-standard location | TS-11 |
| NG-1: No path existence validation | TS-12 |
| NG-3: No symlink resolution | TS-13 |

## Test Scenarios

### Path Resolution (US-1)

```
TS-1: Absolute path returned as-is
  Given a config.yaml with paths.data_dir set to "/Volumes/data/project"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("/Volumes/data/project")
  And the returned Path satisfies is_absolute() == True
```

```
TS-2: Relative path resolved via project root
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_dir is set to "data"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("<tmp>/data")
  And the returned Path satisfies is_absolute() == True
```

```
TS-3: Project root derived from config file location
  Given a config.yaml located at "/home/user/myproject/config/config.yaml"
  And paths.data_dir is set to "data/output"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("/home/user/myproject/data/output")
```

```
TS-4: Trailing slash stripped from path value
  Given a config.yaml with paths.data_dir set to "data/output/"
  And a config.yaml located at "<tmp>/config/config.yaml"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("<tmp>/data/output")
  And the path string does not end with "/"
```

```
TS-4b: Trailing slash stripped from absolute path value
  Given a config.yaml with paths.data_dir set to "/Volumes/data/"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("/Volumes/data")
```

### Sub-Path Fragment Composition (US-2)

```
TS-5: Getter composes base path with sub-path fragment
  Given a config.yaml with paths.data_article_generation_institutional_memory_dir
        set to "data/institutional_memory"
  And article_generation.institutional_memory.fact_checking_subdir set to
      "fact_checking"
  And the config file is at "<tmp>/config/config.yaml"
  When a Config getter for the fact-checking institutional memory directory is called
  Then the returned Path equals Path("<tmp>/data/institutional_memory/fact_checking")
  And the returned Path satisfies is_absolute() == True
```

### Project Root Independence (C-3)

```
TS-6: Project root does not depend on current working directory
  Given a config.yaml located at "/opt/project/config/config.yaml"
  And paths.data_dir is set to "data"
  And the current working directory is "/home/user"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("/opt/project/data")
  And the returned Path does NOT contain "/home/user"
```

### Schema Validation (C-6)

```
TS-7: Unknown path key rejected by PathsConfig
  Given a config.yaml with an extra key "paths.nonexistent_dir" set to "foo"
  When Config is loaded
  Then a validation error is raised mentioning "extra" or "nonexistent_dir"
```

### Edge Cases

```
TS-8: Relative path with parent directory segments
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_dir is set to "../sibling/data"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("<tmp>/../sibling/data")
  And the returned Path satisfies is_absolute() == True
```

```
TS-9: Empty string path rejected
  Given a config.yaml with paths.data_dir set to ""
  When Config is loaded
  Then a validation error is raised
  And the error message indicates the value is too short or empty
```

```
TS-10: Root path "/" treated as valid absolute path
  Given a config.yaml with paths.data_dir set to "/"
  When Config is loaded and getDataDir() is called
  Then the returned Path equals Path("/")
  And the returned Path satisfies is_absolute() == True
```

```
TS-11: Config loaded from non-standard location
  Given a config.yaml located at "/tmp/test_config.yaml"
  And paths.data_dir is set to "mydata"
  When Config is loaded
  Then the project root is Path("/tmp").parent which is Path("/")
  And getDataDir() returns Path("/mydata")
```

### Non-Goal Verification

```
TS-12: Non-existent path does not cause error at load time
  Given a config.yaml with paths.data_dir set to "/nonexistent/path/that/does/not/exist"
  When Config is loaded
  Then no error is raised
  And getDataDir() returns Path("/nonexistent/path/that/does/not/exist")
```

```
TS-13: Symlinks are not resolved
  Given a config.yaml with paths.data_dir set to "/some/symlinked/path"
  When Config is loaded and getDataDir() is called
  Then the returned Path string is "/some/symlinked/path"
  And Path.resolve() is NOT called on the value
```

### Structural Validation (US-2, US-3, US-4)

These scenarios verify conventions across the codebase rather than single
function calls. They are implemented as static analysis or grep-based checks.

```
TS-14: No directory path values in domain config sections
  Given the config.yaml file
  When the article_generation, topic_detection, hallucination_detection,
       and article_compiler sections are inspected
  Then none of these sections contain values that are absolute directory paths
       (values starting with "/" and ending with a directory-like pattern)
  And the only path-like values in domain sections are sub-path fragments
       (single directory names or short relative fragments without "/" prefix)
```

```
TS-15: Domain path keys follow naming convention
  Given the paths: section in config.yaml
  When all keys that serve domain-specific directories are listed
  Then each such key matches the pattern "data_<domain>_<purpose>_dir"
```

```
TS-16: Scripts use Config getters without manual path construction
  Given all Python files in scripts/
  When the source code is searched for patterns:
       - "project_root /" or "base_dir /" followed by "config.get"
       - "config._data"
       - "config.getDataDir() / td_config." or similar manual joins
  Then zero matches are found
```

```
TS-17: Config.yaml path values follow formatting rules
  Given all path values in the paths: section of config.yaml
  When each value is inspected
  Then no value ends with "/"
  And no value starts with "./"
  And each value either starts with "/" (absolute) or with a letter/digit (relative)
```

### Exhaustive Getter Coverage (AC-1.1)

These scenarios verify that EVERY path getter on the Config class returns
an absolute Path that matches the expected resolved value. They are
parameterized: each path key in the `paths:` section is tested with both
an absolute and a relative config value.

```
TS-18: Every paths: getter returns absolute Path with absolute config values
  Given a config.yaml where every key in paths: is set to a unique absolute
        path (e.g., data_dir="/abs/data", data_models_dir="/abs/models", ...)
  When Config is loaded
  And each of the following getters is called:
        - getDataDir()
        - getDataModelsDir()
        - getDataDownloadsDir()
        - getDataDownloadsVideosDir()
        - getDataDownloadsTranscriptsDir()
        - getDataDownloadsTranscriptsHallucinationsDir()
        - getDataDownloadsTranscriptsCleanedDir()
        - getDataTranscriptsTopicsDir()
        - getDataDownloadsAudioDir()
        - getDataDownloadsMetadataDir()
        - getDataOutputDir()
        - getDataInputDir()
        - getDataTempDir()
        - getDataArchiveDir()
        - getDataArchiveVideosDir()
        - getDataLogsDir()
        - getDataOutputArticlesDir()
        - getDataArticlesInputDir()
        - getReportsDir()
        (and any new domain-specific getters added as part of this feature)
  Then every returned value is a Path object
  And every returned value satisfies is_absolute() == True
  And every returned value equals the absolute path set in config
```

```
TS-19: Every paths: getter returns absolute Path with relative config values
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And every key in paths: is set to a unique relative path
        (e.g., data_dir="data", data_models_dir="data/models", ...)
  When Config is loaded
  And each of the getters listed in TS-18 is called
  Then every returned value is a Path object
  And every returned value satisfies is_absolute() == True
  And every returned value equals Path("<tmp>") / <relative_value>
```

```
TS-20: Mixed absolute and relative paths in same config
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_dir is set to "/absolute/data" (absolute)
  And paths.data_models_dir is set to "data/models" (relative)
  And paths.data_output_dir is set to "/absolute/output" (absolute)
  And paths.reports_dir is set to "reports" (relative)
  When Config is loaded
  Then getDataDir() returns Path("/absolute/data")
  And getDataModelsDir() returns Path("<tmp>/data/models")
  And getDataOutputDir() returns Path("/absolute/output")
  And getReportsDir() returns Path("<tmp>/reports")
```

### Domain-Specific Getters (New Getters from US-2)

These scenarios cover the new getters that replace the manual path
construction patterns currently used in scripts (e.g., the former
`config.getDataDir() / td_config.output_dir` pattern).

```
TS-21: Topic detection output dir getter with relative path
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_topic_detection_output_dir is set to "data/output/topics"
  When the Config getter for the topic detection output directory is called
  Then the returned Path equals Path("<tmp>/data/output/topics")
  And the returned Path satisfies is_absolute() == True
```

```
TS-22: Article generation output dir getter with absolute path
  Given a config.yaml with paths.data_article_generation_output_dir
        set to "/Volumes/data/output/articles"
  When the Config getter for the article generation output directory is called
  Then the returned Path equals Path("/Volumes/data/output/articles")
```

```
TS-23: Article generation artifacts dir getter with relative path
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_article_generation_artifacts_dir is set to
      "data/output/article_editor_runs"
  When the Config getter for the article generation artifacts directory is called
  Then the returned Path equals Path("<tmp>/data/output/article_editor_runs")
```

```
TS-24: Knowledge base dir getter
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_article_generation_kb_dir is set to "data/knowledgebase"
  When the Config getter for the knowledge base data directory is called
  Then the returned Path equals Path("<tmp>/data/knowledgebase")
```

```
TS-25: Topic detection taxonomies dir getter
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_topic_detection_taxonomies_dir is set to "data/input/taxonomies"
  When the Config getter for the topic detection taxonomies directory is called
  Then the returned Path equals Path("<tmp>/data/input/taxonomies")
```

## Edge Case Scenarios

All edge cases from the specification are covered:

| Edge Case | Scenario |
|-----------|----------|
| EC-1: `..` segments | TS-8 |
| EC-2: Empty string | TS-9 |
| EC-3: Root path `/` | TS-10 |
| EC-4: macOS `/Volumes/` | TS-1 (uses `/Volumes/` in Given) |
| EC-5: Non-standard config location | TS-11 |
| EC-6: config.yaml.template update | (verified during implementation review, not a code test) |

## Traceability

Every acceptance criterion (AC-1.1 through AC-4.3), constraint (C-3, C-6),
edge case (EC-1 through EC-5), and non-goal (NG-1, NG-3) has at least one
corresponding test scenario.

TS-18 through TS-20 provide exhaustive coverage of every existing path getter
on the Config class with both absolute and relative values. TS-21 through TS-25
cover the new domain-specific getters introduced by this feature. Together,
these ensure that the path resolution mechanism is verified for every config
key, not just a representative sample.

Coverage is complete.
