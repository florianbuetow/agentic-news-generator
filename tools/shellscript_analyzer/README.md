# Shell Script Parameter Detector

An AI-powered tool using Autogen v0.7.5 to analyze shell scripts and determine whether they rely on environment variables or accept parameters via command line arguments.

## Purpose

This tool helps identify shell scripts that could benefit from better parameter handling by:
- Detecting scripts that rely heavily on undocumented environment variables
- Identifying scripts that should accept CLI arguments for better reusability
- Providing recommendations for improving parameter handling
- Generating detailed reports of parameter usage across all scripts

## What It Analyzes

The tool automatically discovers and analyzes all `.sh` files in:
- `scripts/` directory (recursively, including subdirectories)
- Project root directory (non-recursive)

The set of scripts is determined at run time by discovery — there is no hardcoded list. It analyzes every `.sh` file found under `scripts/` (recursively) plus any `.sh` files in the project root.

## Usage

### Run Full Analysis

```bash
just ai-review-shell-scripts
```

This will:
1. Find all `.sh` files in `scripts/` directory and root level
2. Extract script content
3. Analyze each script with a local LLM via Autogen
4. Generate a markdown report at `reports/shell_env_var_violations.md`

### Run Without Cache

```bash
just ai-review-shell-scripts-nocache
```

Forces a complete re-scan of all scripts, ignoring the hash-based cache.

### Direct Python Invocation

```bash
# Analyze all scripts
uv run python tools/shellscript_analyzer/shellscript_analyzer.py

# Analyze a specific script
uv run python tools/shellscript_analyzer/shellscript_analyzer.py --file scripts/convert_to_audio.sh

# Run discovery test (find all scripts)
uv run python tools/shellscript_analyzer/shellscript_analyzer.py --test discover

# Run extraction test (extract content from first script)
uv run python tools/shellscript_analyzer/shellscript_analyzer.py --test extract

# Run analysis test (analyze first script)
uv run python tools/shellscript_analyzer/shellscript_analyzer.py --test analyze
```

## Requirements

- LM Studio (or a compatible OpenAI-compatible LLM server) running locally.
- The base URL, model, and API key are read from `config/config.yaml` (via `Config.get_agentic_shell_script_reviews_config()`). The tool does **not** read environment variables and does **not** accept connection details as CLI arguments — configure the server in `config.yaml` and make sure the model is loaded (`just status`).

## Analysis Criteria

The tool evaluates scripts based on:

1. **Environment Variable Usage**
   - Detects `$VAR`, `${VAR}`, `${VAR:-default}` patterns
   - Lists all environment variables used
   - Checks for documentation

2. **CLI Argument Handling**
   - Detects `$1`, `$2`, `$@`, `getopts` patterns
   - Analyzes argument parsing logic
   - Checks for usage/help functions

3. **Violation Status (PASS / FAIL / ERROR)**
   - **PASS**: Uses no environment variables, or every environment-derived value is passed in as a CLI argument
   - **FAIL**: Depends on environment variables (or sourced configuration) that are not passed as CLI arguments
   - **ERROR**: The script could not be analyzed (e.g. the LLM API was unavailable)

4. **Recommendations**
   - For FAIL scripts, identifies the offending variables and recommends passing them as CLI arguments

## Output

### Report Location

`reports/shell_env_var_violations.md`

### Report Sections

1. **Summary**: How many scripts passed, failed, or errored
2. **Scripts With Violations**: Scripts that FAIL, with their offending variables
3. **All Scripts**: Complete PASS / FAIL / ERROR breakdown

### Example Report Entry

```markdown
### scripts/example.sh

**Status:** ❌ FAIL

**Environment Variables:** Yes
  - `SOME_VAR`

**Recommendation:** Pass `SOME_VAR` as a CLI argument instead of reading it from the environment.
```

## Caching

The tool uses hash-based caching to skip unchanged files:
- Cache location: `.cache/shell_script_hashes.json`
- Tracks file hash, last check time, and analysis status
- Use `--no-cache` flag to force re-scan

## Integration with CI

This tool is part of the AI-based CI pipeline:

```bash
just ci-ai
```

This runs:
1. `ai-review-unit-tests-nocache` - Detects fake unit tests
2. `ai-review-shell-scripts-nocache` - Analyzes shell script parameters

## Exit Codes

- `0`: All scripts pass (no environment-variable violations)
- `1`: One or more scripts fail (violations found)
- `2`: Error occurred during analysis or LLM API unavailable
- `130`: Interrupted by user (Ctrl+C)

## Architecture

The tool consists of 5 main classes:

1. **ScriptFileHashCache**: Manages hash-based caching
2. **ShellScriptFinder**: Discovers `.sh` files in the project
3. **ShellScriptExtractor**: Extracts script content
4. **ShellScriptParameterAnalyzer**: Analyzes scripts with Autogen v0.7.5
5. **ShellScriptAnalysisOrchestrator**: Orchestrates the complete workflow

## Example Workflow

```
1. Find shell scripts
   └─> scripts/convert_to_audio.sh
   └─> scripts/yt-downloader.sh
   └─> ...

2. Check cache
   └─> scripts/convert_to_audio.sh (unchanged, skip)
   └─> scripts/yt-downloader.sh (changed, analyze)

3. Extract content
   └─> Read file, count lines

4. Analyze with LLM
   └─> Build prompt with script content
   └─> Call Autogen model client
   └─> Parse JSON response

5. Generate report
   └─> Markdown report with all findings
   └─> Summary statistics
   └─> Recommendations for improvement

6. Update cache
   └─> Store file hash and status
```

## Tips

- Run with `--test discover` first to verify it finds the expected scripts
- Use `--file` to test analysis on a single script before running full analysis
- Keep LM Studio running to avoid API connection errors
- Review `reports/shell_env_var_violations.md` for detailed findings
