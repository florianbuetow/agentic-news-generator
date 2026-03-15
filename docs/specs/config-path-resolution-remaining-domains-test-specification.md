# Config Path Resolution: Remaining Domain Sections - Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|------------------|------------------|
| AC-1.1: `data_hallucination_detection_output_dir` key in paths: | TS-1, TS-2 |
| AC-1.2: `hallucination_detection` section has no `output_dir` | TS-7 |
| AC-1.3: `getHallucinationDetectionOutputDir()` getter exists, returns absolute Path | TS-1, TS-2 |
| AC-1.4: PathsConfig has `data_hallucination_detection_output_dir` field | TS-1 |
| AC-1.5: `transcript-hallucination-detection.py` uses getter | TS-8 |
| AC-1.6: No other code reads `hallucination_detection.output_dir` | TS-8 |
| AC-2.1: `data_article_compiler_input_dir` and `data_article_compiler_output_file` keys in paths: | TS-3, TS-4, TS-5 |
| AC-2.2: `article_compiler` section has no `input_dir` or `output_file` | TS-7 |
| AC-2.3: `ArticleCompilerConfig` has no `input_dir` or `output_file` fields | TS-9 |
| AC-2.4: `getArticleCompilerInputDir()` and `getArticleCompilerOutputFile()` getters exist | TS-3, TS-4, TS-5 |
| AC-2.5: PathsConfig has corresponding fields | TS-3, TS-4 |
| AC-2.6: `compile_articles.py` uses new getters | TS-8 |
| AC-3.1: Structural test TS-14 passes | TS-7 |
| AC-3.2: `just ci-quiet` passes | (verified during implementation) |
| C-1: Same resolution rules (absolute/relative, trailing slash) | TS-1 through TS-5 |
| EC-1: `_file` suffix for file path key | TS-4, TS-5 |
| EC-2: Hallucination output_dir relative base changed to project root | TS-2 |

## Test Scenarios

### Hallucination Detection Path (US-1)

```
TS-1: Hallucination detection output dir getter with absolute path
  Given a config.yaml with paths.data_hallucination_detection_output_dir
        set to "/Volumes/data/hallucinations"
  When Config is loaded and getHallucinationDetectionOutputDir() is called
  Then the returned Path equals Path("/Volumes/data/hallucinations")
  And the returned Path satisfies is_absolute() == True
```

```
TS-2: Hallucination detection output dir getter with relative path
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_hallucination_detection_output_dir is set to
      "data/downloads/transcripts-hallucinations"
  When Config is loaded and getHallucinationDetectionOutputDir() is called
  Then the returned Path equals
       Path("<tmp>/data/downloads/transcripts-hallucinations")
  And the returned Path satisfies is_absolute() == True
```

### Article Compiler Paths (US-2)

```
TS-3: Article compiler input dir getter with relative path
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_article_compiler_input_dir is set to
      "data/input/newspaper/articles"
  When Config is loaded and getArticleCompilerInputDir() is called
  Then the returned Path equals
       Path("<tmp>/data/input/newspaper/articles")
  And the returned Path satisfies is_absolute() == True
```

```
TS-4: Article compiler output file getter with relative path
  Given a config.yaml located at "<tmp>/config/config.yaml"
  And paths.data_article_compiler_output_file is set to
      "data/input/newspaper/articles.js"
  When Config is loaded and getArticleCompilerOutputFile() is called
  Then the returned Path equals
       Path("<tmp>/data/input/newspaper/articles.js")
  And the returned Path satisfies is_absolute() == True
```

```
TS-5: Article compiler output file getter with absolute path
  Given a config.yaml with paths.data_article_compiler_output_file
        set to "/Volumes/data/newspaper/articles.js"
  When Config is loaded and getArticleCompilerOutputFile() is called
  Then the returned Path equals Path("/Volumes/data/newspaper/articles.js")
```

### Exhaustive Getter Coverage (extends parent spec TS-18/TS-19)

```
TS-6: New getters included in parameterized absolute/relative tests
  Given the GETTER_REGISTRY in test_config_path_resolution.py
  When the registry is inspected
  Then it contains entries for:
       - ("getHallucinationDetectionOutputDir", "data_hallucination_detection_output_dir")
       - ("getArticleCompilerInputDir", "data_article_compiler_input_dir")
       - ("getArticleCompilerOutputFile", "data_article_compiler_output_file")
  And the parameterized tests TS-18 and TS-19 exercise these getters
```

### Structural Validation (US-3)

```
TS-7: No directory paths remain in any domain config section
  Given the config.yaml file
  When the hallucination_detection and article_compiler sections are inspected
  Then neither section contains values that are directory or file paths
  And the existing structural test test_no_directory_paths_in_domain_sections passes
```

```
TS-8: Scripts use Config getters without manual path construction
  Given all Python files in scripts/
  When the source code is searched for:
       - "hallucination_config" followed by "output_dir"
       - "compiler_config.input_dir"
       - "compiler_config.output_file"
  Then zero matches are found
```

### Non-Goal Verification

```
TS-9: ArticleCompilerConfig retains non-path fields
  Given a valid config.yaml with article_compiler section
  When Config is loaded and get_article_compiler_config() is called
  Then ArticleCompilerConfig has fields: min_articles, date_format,
       paragraphs, images, links
  And ArticleCompilerConfig does NOT have fields: input_dir, output_file
```

## Traceability

Every acceptance criterion (AC-1.1 through AC-3.2), constraint (C-1),
and edge case (EC-1, EC-2) has at least one corresponding test scenario.
Coverage is complete.
