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
- Always verify changes with `just ci` after modifications
- **NEVER** run a full-pipeline target (`just pipelines-all`, `just video-all`, `just url-all`) as a routine verification step; they perform long-running, side-effectful work (downloads, transcription, LLM calls)

## Python Execution Rules
- Python code must be executed **only** via `uv run ...`
  - Example: `uv run scripts/summarize-transcripts.py`
  - **Never** use: `python scripts/summarize-transcripts.py` or `python3 scripts/summarize-transcripts.py`
- The virtual environment must be created and updated **only** via `uv sync`
  - **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies must be managed through `uv` and declared in `pyproject.toml`

## Justfile Rules
- All Python execution in the justfile uses `uv run`, never `python` directly
- Use `just init` to set up the project
- Use `just pipelines-all` to run the full URL + video pipelines (or `just video-all` / `just url-all` for one)
- Use `just` (without arguments) to see all available targets

## Project Structure
- All source code lives in `src/`
- Test scripts and utilities go in `scripts/`
- Prompt templates go in `prompts/`
- **Input data**: `data/input/`
- **Output data**: `data/output/`
- **Temporary debug scripts**: `debug/` (see `TROUBLESHOOTING-GUIDE.md` for usage guidelines)
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
- Resilient Processing: Scripts must continue processing remaining items even if individual items fail. A single failure should never halt an entire batch or pipeline.
- Failure Visibility: All failures must be logged appropriately.
- Reporting Requirement: At the end of execution (and thus at the end of a just target), a summary of all failures must be printed to the screen. This summary should include the specific items that failed and the reasons, allowing for immediate inspection.
- Exit Codes: Scripts must track success/failure counts. Exit with code 1 if any items failed (even if others succeeded), and 0 only if all items succeeded.
- **Implementation Pattern**:
  ```python
  failures = []
  for item in items:
      try:
          process(item)
      except Exception as e:
          logger.error(f"Failed to process {item}: {e}")
          failures.append((item, str(e)))
  
  if failures:
      print("\n--- Failure Summary ---")
      for item, error in failures:
          print(f"❌ {item}: {error}")
      sys.exit(1)
  ```
- Goal: This ensures processing runs through completely while ensuring all errors are visibly flagged for remediation. Both are required.

## Status Output in Long-Running Loops
Applies to any loop that processes many items (files, URLs, records) or where a single item takes a noticeable amount of time. The terminal must always show what is happening — never leave the operator staring at a silent prompt while work is in progress.

Rules:
- **Skipped items get exactly one line.** If an item needs no work (already processed, etc.), print a single `Skipping: <item>` line and move on. Do not print intermediate detail, sub-steps, or file paths for skips.
- **Announce work before doing it.** Print `Processing: <item>` before starting a slow item, and print each slow step (`<step>...`) *before* running it — so progress is visible while the step runs, not only after it finishes. This is the difference between a live status display and a terminal that looks hung.
- **Close each processed item** with a short `done` line.
- **Aggregate errors and report them at the end**, then exit non-zero. The end-of-run summary is authoritative; see [Error Handling](#error-handling) for the exact summary format and exit-code rules.

**Pattern**:
```text
for job in jobs:
    if can_skip(job):
        print(f"Skipping: {job}")
        continue
    print(f"Processing: {job}")
    print("step 1...")          # announce BEFORE the slow step, not after
    do_step_1(job)
    print("step 2...")
    do_step_2(job)
    print("done")

if errors:
    for err in errors:
        print(f"ERROR: {err}")
    print(f"Encountered {len(errors)} errors")
    sys.exit(1)
sys.exit(0)
```

Goal: the operator can always tell, in real time, which item is being worked on and which step is running, while skipped items stay quiet and on one line.

## Configuration Management
- **Never use environment variables** for configuration
- All config parameters must be loaded through `config.py` from `config.yaml`
- Never hardcode configuration values in source code
- **Never mention specific config values in code comments or docstrings** - they will get out of sync with actual config and cause confusion
- Use `config/config.yaml` for all settings (add sensitive values to `.gitignore` or use `config.yaml.local`)
- Document configuration options in `config/config.yaml.template`

### How to extend Configuration
1. Add the new setting to `config/config.yaml` and `config/config.yaml.template`.
2. Add the corresponding field to the appropriate `BaseModel` in `src/config.py`.
3. Add a getter method to the `Config` class in `src/config.py`.
4. **NEVER** access `os.environ` or the YAML file directly in your script.

### How to use Configuration in Shell Scripts
Extract values from `config/config.yaml` using a one-liner:
```bash
# Example extraction in shell
DATA_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('config/config.yaml'))['paths']['data_dir'])")
```

## Output Organization
- Organize output files by date or topic as appropriate
- Use clear, descriptive filenames
- Include metadata files alongside generated content when useful
- Keep raw outputs separate from processed/refined outputs

## Troubleshooting & Debugging
- See `TROUBLESHOOTING-GUIDE.md` at project root for catalogue of diagnostic helper scripts in `scripts/` and `tools/`
- Covers: finding files by video ID, detecting corrupt videos, empty transcripts, hallucination analysis, LM Studio status checks, AI-powered test/shell reviewers, and code quality diagnostics
- Contains playbooks for common failure modes (empty transcripts, corrupt videos, LLM pipeline stalls)
- Check there first before writing new debug scripts
