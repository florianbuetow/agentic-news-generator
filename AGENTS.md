# Development Rules for Agentic News Generator

## General Coding Principles
- **Never assume any default values anywhere**
- Always be explicit about values, paths, and configurations
- If a value is not provided, handle it explicitly (raise error, use null, or prompt for input)
- **NEVER MAINTAIN BACKWARDS COMPATIBILITY WHEN CHANGING CODE**
  - **ABSOLUTELY NO backwards compatibility - EVER**
  - Backwards compatibility accumulates technical debt
  - Make breaking changes cleanly rather than adding compatibility layers
  - Delete unused code completely instead of keeping it for compatibility
  - **NO legacy format support, NO fallback logic, NO "handle both old and new" code**
  - When changing a format or structure, the old format becomes immediately unsupported
  - If old data exists, it must be regenerated or migrated - never supported in-place
- **NEVER delete any files without explicitly asking the user first**
- **ALWAYS verify .gitignore before running destructive commands that might delete files**

## Git Commit Guidelines
- **NEVER include AI attribution in commit messages**
- **NEVER add "Generated with AI assistance" or similar phrases**
- **NEVER add "Co-Authored-By: [AI Agent Name]" or similar attribution**
- **NEVER mention authors, creators, or external people by name in commit messages**
- **NEVER mention source URLs, video links, or external sources in commit messages**
- **NEVER run `git add -A` or `git add .` - always stage files explicitly**
- Keep commit messages professional and focused on the changes made
- Commit messages should describe what changed and why, without mentioning AI assistance, authors, or sources
- Note: this rule applies to commit messages only — guides, docs, and code comments may credit authors and link sources as needed

## Testing
- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `just run` after modifications

## Python Execution Rules
- Python code must be executed **only** via `uv run ...`
  - Example: `uv run src/main.py`
  - **Never** use: `python src/main.py` or `python3 src/main.py`
- The virtual environment must be created and updated **only** via `uv sync`
  - **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies must be managed through `uv` and declared in `pyproject.toml`

## Justfile Rules
- All Python execution in the justfile uses `uv run`, never `python` directly
- Use `just init` to set up the project
- Use `just run` to execute the main program
- Use `just` (without arguments) to see all available targets

## Project Structure
- All source code lives in `src/`
- Test scripts and utilities go in `scripts/`
- Prompt templates go in `prompts/`
- **Input data**: `data/input/`
- **Output data**: `data/output/`
- **Temporary debug scripts**: `debug/` (see `TROUBLESHOOTING.md` for usage guidelines)
- **Never create Python files in the project root directory**
  - Wrong: `./test.py`, `./helper.py`
  - Correct: `./src/helper.py`, `./scripts/test.py`, `./debug/test_something.py`

## News Generation
- Use OpenAI compatible API (LM Studio) for news generation
- Store prompt templates in `prompts/` directory
- Input topics/sources should be stored in `data/input/`
- Generated articles should be saved to `data/output/`
- Each generated article should include:
  - Title
  - Content
  - Timestamp
  - Source information (if applicable)

## Error Handling
- Scripts should continue processing other items even if one fails
- Failed/invalid outputs must be logged appropriately
- Scripts should track and report success/failure counts
- Exit with code 1 if any items failed, 0 if all succeeded

## Configuration Management
- **Never use environment variables** for configuration
- All config parameters must be loaded through `config.py` from `config.yaml`
- Never hardcode configuration values in source code
- **Never mention specific config values in code comments or docstrings** - they will get out of sync with actual config and cause confusion
- Use `config/config.yaml` for all settings (add sensitive values to `.gitignore` or use `config.yaml.local`)
- Document configuration options in `config/config.yaml.template`

## Output Organization
- Organize output files by date or topic as appropriate
- Use clear, descriptive filenames
- Include metadata files alongside generated content when useful
- Keep raw outputs separate from processed/refined outputs

## Troubleshooting & Debugging
- See `TROUBLESHOOTING.md` at project root for catalogue of diagnostic helper scripts in `scripts/` and `tools/`
- Covers: finding files by video ID, detecting corrupt videos, empty transcripts, hallucination analysis, LM Studio status checks, AI-powered test/shell reviewers, and code quality diagnostics
- Contains playbooks for common failure modes (empty transcripts, corrupt videos, LLM pipeline stalls)
- Check there first before writing new debug scripts

<!-- progressive-disclosure:index:start -->
## Documentation Index

### Introduction
- For project overview, features, and usage instructions, see [`README.md`](README.md)
- For the newspaper frontend and its content system, see [`frontend/newspaper/README.md`](frontend/newspaper/README.md)
- For a high-level technical walkthrough of the project, see [`docs/lightningtalk/project-blitz.md`](docs/lightningtalk/project-blitz.md)

### Architecture
- For the topic detection pipeline design, see [`docs/topic-detection.md`](docs/topic-detection.md)
- For topic pipeline input/output specifications, see [`docs/topics_pipeline_specifications.md`](docs/topics_pipeline_specifications.md)
- For the magnetic segment merging algorithm, see [`docs/magnetic_topics.md`](docs/magnetic_topics.md)
- For the MiniSeg text segmentation approach, see [`docs/how to implement MiniSeg with python on text files.md`](docs/how%20to%20implement%20MiniSeg%20with%20python%20on%20text%20files.md)
- For the transcription limit and date ordering design, see [`docs/superpowers/specs/2026-04-09-transcription-limit-and-date-ordering-design.md`](docs/superpowers/specs/2026-04-09-transcription-limit-and-date-ordering-design.md)
- For the transcription limit implementation plan, see [`docs/superpowers/plans/2026-04-09-transcription-limit-and-date-ordering-plan.md`](docs/superpowers/plans/2026-04-09-transcription-limit-and-date-ordering-plan.md)

### Research
- For audio speech vs music detection approaches, see [`docs/research_music_vs_speech_detection.md`](docs/research_music_vs_speech_detection.md)
- For text segmentation by semantic boundaries research, see [`reports/research_text_segmentation_semantic_boundaries.md`](reports/research_text_segmentation_semantic_boundaries.md)
- For audio speech detection implementation research, see [`reports/research_audio_speech_detection.md`](reports/research_audio_speech_detection.md)

### API & Reference
- For the transcript summarization prompt template, see [`prompts/summarize-transcript.md`](prompts/summarize-transcript.md)

### Development
- For the shell script parameter detector tool, see [`tools/shellscript_analyzer/README.md`](tools/shellscript_analyzer/README.md)
- For notebook restructuring tasks, see [`notebooks/shellscript_analyzer/NOTEBOOK_RESTRUCTURING_TODO.md`](notebooks/shellscript_analyzer/NOTEBOOK_RESTRUCTURING_TODO.md)

### Operations
- For diagnostic scripts and failure playbooks, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- For the Gemini topics task investigation, see [`debug/gemini_topics_task.md`](debug/gemini_topics_task.md)

### Reports
- For topic detection quality comparison across LLMs, see [`reports/llm-topic-extraction-comparison/quality-review.md`](reports/llm-topic-extraction-comparison/quality-review.md)
- For gap analysis between research and current pipeline, see [`reports/gap_analysis_merged.md`](reports/gap_analysis_merged.md)
- For Codex-specific gap analysis, see [`reports/gap_analysis_codex.md`](reports/gap_analysis_codex.md)
- For Gemini-specific gap analysis, see [`reports/gap_analysis_gemini.md`](reports/gap_analysis_gemini.md)
- For agentic code evaluation results, see [`reports/agentic_code_evals.md`](reports/agentic_code_evals.md)
- For model performance visualizations, see [`reports/visualizations/README.md`](reports/visualizations/README.md)
- For shell environment variable violations, see [`reports/shell_env_var_violations.md`](reports/shell_env_var_violations.md)
- For shell script parameter analysis, see [`reports/shell_script_params.md`](reports/shell_script_params.md)
- For shell env detector test results, see [`reports/shell_env_detector_test_results.md`](reports/shell_env_detector_test_results.md)

### Appendix
- For the project changelog, see [`CHANGELOG.md`](CHANGELOG.md)
<!-- progressive-disclosure:index:end -->

