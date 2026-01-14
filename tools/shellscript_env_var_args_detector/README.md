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

Currently analyzes **8 shell scripts** in this project:
- `scripts/archive-videos.sh`
- `scripts/config.sh`
- `scripts/convert_to_audio.sh`
- `scripts/init-directories.sh`
- `scripts/move-metadata.sh`
- `scripts/move-transcript-metadata.sh`
- `scripts/transcribe_single.sh`
- `scripts/yt-downloader.sh`

## Usage

### Run Full Analysis

```bash
just ai-review-shell-scripts
```

This will:
1. Find all `.sh` files in `scripts/` directory and root level
2. Extract script content
3. Analyze each script with a local LLM via Autogen
4. Generate a markdown report at `reports/shell_script_params.md`

### Run Without Cache

```bash
just ai-review-shell-scripts-nocache
```

Forces a complete re-scan of all scripts, ignoring the hash-based cache.

### Direct Python Invocation

```bash
# Analyze all scripts
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py

# Analyze a specific script
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py --file scripts/convert_to_audio.sh

# Run discovery test (find all scripts)
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py --test discover

# Run extraction test (extract content from first script)
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py --test extract

# Run analysis test (analyze first script)
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py --test analyze
```

## Requirements

- LM Studio (or compatible OpenAI-compatible LLM server) running locally
- Default: `http://localhost:1234/v1`
- Default model: `qwen2.5-7b-instruct-mlx`

You can override these with environment variables:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export LM_STUDIO_MODEL="qwen2.5-7b-instruct-mlx"
export LM_STUDIO_API_KEY="local"
```

Or with command-line arguments:

```bash
uv run python tools/shellscript_env_var_args_detector/detect_shell_params.py \
  --base-url http://localhost:1234/v1 \
  --model qwen2.5-7b-instruct-mlx \
  --api-key local
```

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

3. **Parameter Quality Score (0.0-1.0)**
   - **0.8-1.0 (High)**: Primarily uses CLI args with good defaults, env vars as optional overrides
   - **0.5-0.7 (Medium)**: Mixed approach or well-documented env var usage
   - **0.0-0.4 (Low)**: Relies heavily on undocumented env vars without CLI arg alternatives

4. **Recommendations**
   - Suggests improvements for scripts scoring below 0.7
   - Identifies missing documentation
   - Recommends adding CLI argument support

## Output

### Report Location

`reports/shell_script_params.md`

### Report Sections

1. **Summary**: Overall statistics and average quality score
2. **Scripts Needing Improvement**: Scripts with score < 0.7
3. **All Scripts Analysis**: Complete breakdown of all analyzed scripts

### Example Report Entry

```markdown
### scripts/convert_to_audio.sh

**Parameter Quality Score:** 0.85/1.0

✅ **Status:** Good parameter handling

**Environment Variables:** Yes
  - `ENABLE_SILENCE_REMOVAL`
  - `SILENCE_THRESHOLD_DB`

**CLI Arguments:** Yes
  - Accepts input video files as positional arguments

**Recommendation:**
```

## Caching

The tool uses hash-based caching to skip unchanged files:
- Cache location: `.cache/shell_script_hashes.json`
- Tracks file hash, last check time, and parameter score
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

- `0`: All scripts have good parameter handling (score >= 0.7)
- `1`: Some scripts need improvement (score < 0.7)
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
   └─> Store file hash and scores
```

## Tips

- Run with `--test discover` first to verify it finds the expected scripts
- Use `--file` to test analysis on a single script before running full analysis
- Keep LM Studio running to avoid API connection errors
- Review `reports/shell_script_params.md` for detailed findings
