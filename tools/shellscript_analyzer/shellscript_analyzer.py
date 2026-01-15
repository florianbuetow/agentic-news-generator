#!/usr/bin/env python3
"""Shell Script Parameter Detector using Autogen v0.7.5.

This tool uses a local LLM via Autogen to analyze shell scripts and determine
whether they rely on environment variables or accept parameters via command line
arguments. It helps identify scripts that should be more configurable or explicit
about their dependencies.

Usage:
    python tools/shellscript_analyzer/detect_shell_params.py                    # Run full detection
    python tools/shellscript_analyzer/detect_shell_params.py --test discover    # Test file discovery
    python tools/shellscript_analyzer/detect_shell_params.py --test extract     # Test script extraction
    python tools/shellscript_analyzer/detect_shell_params.py --test analyze     # Test agent analysis
"""

import argparse
import asyncio
import glob
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import requests
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Set our logger to INFO but keep third-party loggers quiet
logger.setLevel(logging.INFO)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ShellScript:
    """Represents a shell script file to be analyzed."""

    file_path: str
    content: str  # Full script content
    line_count: int


@dataclass
class AnalysisResult:
    """Result of analyzing a shell script for environment variable violations."""

    file_path: str
    status: str  # "PASS" or "FAIL" or "ERROR"
    env_vars: list[str]  # Environment variables detected
    summary: str  # Explanation of the analysis


# =============================================================================
# Class 0: ScriptFileHashCache
# =============================================================================


class ScriptFileHashCache:
    """Manages hash-based caching to skip unchanged shell script files."""

    def __init__(self, cache_file: str = ".cache/shell_script_hashes.json") -> None:
        """Initialize the hash cache.

        Args:
            cache_file: Path to the JSON cache file
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, str | float]]:
        """Load the hash cache from JSON file.

        Returns:
            Dictionary mapping file paths to hash info
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load cache file: {e}. Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save the hash cache to JSON file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex digest of the file hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_file_changed(self, file_path: str) -> bool:
        """Check if a file has changed since last scan.

        Args:
            file_path: Path to the file

        Returns:
            True if file is new or has changed, False if unchanged
        """
        current_hash = self._calculate_file_hash(file_path)
        cached_entry = self.cache.get(file_path)

        if cached_entry is None:
            # New file
            return True

        return cached_entry.get("hash") != current_hash

    def update_file_hash(self, file_path: str, status: str = "UNKNOWN") -> None:
        """Update the hash for a file after scanning.

        Args:
            file_path: Path to the file
            status: Analysis status (PASS/FAIL)
        """
        file_hash = self._calculate_file_hash(file_path)
        self.cache[file_path] = {
            "hash": file_hash,
            "last_checked": datetime.now().isoformat(),
            "status": status,
        }
        self._save_cache()

    def get_cached_info(self, file_path: str) -> dict[str, str | float] | None:
        """Get cached information for a file.

        Args:
            file_path: Path to the file

        Returns:
            Cached info dictionary or None if not cached
        """
        return self.cache.get(file_path)

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the cache.

        Args:
            file_path: Path to the file to remove
        """
        if file_path in self.cache:
            del self.cache[file_path]
            self._save_cache()


# =============================================================================
# Class 1: ShellScriptFinder
# =============================================================================


class ShellScriptFinder:
    """Finds all shell script files in the project."""

    def find_shell_scripts(self, root_path: str = ".") -> list[str]:
        """Find all .sh files in the scripts/ directory and root level.

        Args:
            root_path: Root path to search from

        Returns:
            Sorted list of relative paths to shell script files
        """
        shell_scripts: list[str] = []

        # Find all .sh files in scripts/ directory
        scripts_pattern = os.path.join(root_path, "scripts", "**", "*.sh")
        shell_scripts.extend(glob.glob(scripts_pattern, recursive=True))

        # Find .sh files in root directory (non-recursive)
        root_pattern = os.path.join(root_path, "*.sh")
        shell_scripts.extend(glob.glob(root_pattern, recursive=False))

        # Filter out any __pycache__ or hidden files
        shell_scripts = [f for f in shell_scripts if "__pycache__" not in f and not os.path.basename(f).startswith(".")]

        # Keep as relative paths and sort
        shell_scripts.sort()

        return shell_scripts


# =============================================================================
# Class 2: ShellScriptExtractor
# =============================================================================


class ShellScriptExtractor:
    """Extracts content from shell script files."""

    def extract(self, file_path: str) -> ShellScript | None:
        """Extract content from a shell script file.

        Args:
            file_path: Path to the shell script file

        Returns:
            ShellScript object or None if extraction failed
        """
        try:
            # Read file content
            content = self._read_file(file_path)

            # Count lines
            line_count = len(content.splitlines())

            return ShellScript(
                file_path=file_path,
                content=content,
                line_count=line_count,
            )

        except Exception as e:
            logger.warning(f"Error extracting from {file_path}: {e}")
            return None

    def _read_file(self, file_path: str) -> str:
        """Read file handling encoding issues."""
        try:
            with open(file_path, encoding="utf-8-sig") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fall back to latin-1
            with open(file_path, encoding="latin-1") as f:
                return f.read()


# =============================================================================
# API Health Check
# =============================================================================


def check_api_available(base_url: str, timeout: int = 5) -> bool:
    """Check if the LLM API is available.

    Args:
        base_url: Base URL of the API (e.g., http://localhost:1234/v1)
        timeout: Request timeout in seconds

    Returns:
        True if API is available, False otherwise
    """
    try:
        # Try to reach the models endpoint
        models_url = base_url.rstrip("/") + "/models"
        response = requests.get(models_url, timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.RequestException):
        return False


# =============================================================================
# Report Writing
# =============================================================================


class MarkdownReportWriter:
    """Generates markdown reports for shell script parameter analysis."""

    def __init__(self, report_path: str = "reports/shell_env_var_violations.md") -> None:
        """Initialize the report writer.

        Args:
            report_path: Path to write the report (default: reports/shell_env_var_violations.md)
        """
        self.report_path = report_path

    def write_report(self, results: list[AnalysisResult], model: str) -> None:
        """Write a markdown report of shell script environment variable violations.

        Args:
            results: List of AnalysisResult objects
            model: Model name used for detection
        """
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        with open(self.report_path, "w", encoding="utf-8") as f:
            # Header
            _ = f.write("# Shell Script Environment Variable Violations Report\n\n")
            _ = f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            _ = f.write(f"**Model:** {model}\n\n")

            # Summary
            _ = f.write("## Summary\n\n")
            _ = f.write(f"- **Total Scripts Analyzed:** {len(results)}\n")

            passed_scripts = [r for r in results if r.status == "PASS"]
            failed_scripts = [r for r in results if r.status == "FAIL"]
            error_scripts = [r for r in results if r.status == "ERROR"]

            _ = f.write(f"- **Scripts Passed:** {len(passed_scripts)}\n")
            _ = f.write(f"- **Scripts Failed:** {len(failed_scripts)}\n")
            if error_scripts:
                _ = f.write(f"- **Scripts with Errors:** {len(error_scripts)}\n")
            _ = f.write("\n")

            # Failed scripts details
            if failed_scripts:
                _ = f.write("## ❌ Failed Scripts (Environment Variable Violations)\n\n")
                _ = f.write(f"Found {len(failed_scripts)} scripts using environment variables without CLI args or file reading.\n\n")

                for result in sorted(failed_scripts, key=lambda r: r.file_path):
                    _ = f.write(f"### {result.file_path}\n\n")

                    _ = f.write("**Status:** ❌ FAIL\n\n")

                    _ = f.write("**Environment Variables:**\n")
                    for var in result.env_vars:
                        _ = f.write(f"- `{var}`\n")
                    _ = f.write("\n")

                    _ = f.write(f"**Summary:** {result.summary}\n\n")
                    _ = f.write("---\n\n")
            else:
                _ = f.write("No violations found.\n\n")

            # Passed scripts
            if passed_scripts:
                _ = f.write("## ✅ Passed Scripts\n\n")
                _ = f.write(f"Found {len(passed_scripts)} scripts with proper parameter handling.\n\n")

                for result in sorted(passed_scripts, key=lambda r: r.file_path):
                    _ = f.write(f"### {result.file_path}\n\n")
                    _ = f.write("**Status:** ✅ PASS\n\n")

                    if result.env_vars:
                        _ = f.write(f"**Environment Variables:** {', '.join([f'`{v}`' for v in result.env_vars])}\n\n")

                    _ = f.write(f"**Summary:** {result.summary}\n\n")

                    _ = f.write("---\n\n")
            else:
                _ = f.write("## ✅ All Scripts Passed\n\n")
                _ = f.write("No environment variable violations detected. All scripts either:\n")
                _ = f.write("- Use no environment variables, OR\n")
                _ = f.write("- All environment variables are passed as CLI arguments or read from files\n\n")

        # Silent - no log needed

        # Silent - no log needed


# =============================================================================
# Class 3: ShellScriptParameterAnalyzer (Autogen v0.7.5 Agent)
# =============================================================================


class ShellScriptParameterAnalyzer:
    """Uses Autogen v0.7.5 to analyze shell scripts for parameter handling."""

    # System prompt defining analysis criteria
    SYSTEM_PROMPT = """You are a security expert specializing in code reviews of POSIX sh and bash scripts.
Your task is to analyze the script provided in <shell_script> and determine whether it relies on environment variables.
You must strictly follow <rules> and execute <steps> in order, and you must output exactly the JSON
described in <output_format> (no extra keys, no extra text, no extra markdown around it).

<rules>
- Treat any variable expansion ($VAR, ${VAR}, ${VAR:-...}, ${!VAR}) as an environment read unless the variable is
  proven defined/assigned earlier in the script in the same execution path or is a shell special parameter.
- Do NOT consider variables "safe" if they are introduced via sourcing (. file, source file), eval, command
  substitution that imports the environment, or any external command output; treat these as violations.
- Do NOT allow configuration from files; any dependence on environment variables or sourced configuration is a violation.
- A variable is "proven defined" only if it is assigned a value in the script before its first use
  (e.g., VAR=..., local VAR=..., declare VAR=..., VAR+=..., read VAR, for VAR in ..., VAR=($@), etc.).
- Assignments from command-line arguments are allowed only if the value is derived from
  $1..$N, $@, $*, or $OPTARG within explicit argument parsing (e.g., getopts/case/shift loop).
- Variables defined by loop constructs (for VAR in ..., while read VAR, select VAR) count as
  defined starting at the loop header/body, but any use before that is a violation.
- Treat uppercase names (^[A-Z][A-Z0-9_]*$) as especially likely environment variables;
  include them in the suspected list unless proven defined earlier.
- Treat references to common env vars (PATH, HOME, USER, SHELL, TMPDIR, LANG, LC_*, CI, GITHUB_*,
  AWS_*, etc.) as violations unless proven defined earlier in the script.
- Shell special parameters are NOT environment variables and must NOT be reported:
  $0, $1..$9, ${10}, $#, $?, $, $!, $@, $*, $_, $-, $IFS
  (note: IFS is a normal variable; if used without assignment, treat as violation).
- If a variable is only set conditionally and later used outside/after the conditional,
  and you cannot prove it is always set before use, treat that use as a violation.
- If any of these features appear, automatically FAIL and list them as "dynamic/unsafe":
  source, ., eval, declare -n, export, set -a, env, printenv.
- Your analysis must be conservative: when uncertain, assume it is an environment dependency and FAIL.
- Do not execute the script; reason purely from the text.
- Your output must be valid JSON and must follow <output_format> exactly (no extra keys, no commentary outside JSON).
</rules>

<steps>
1) Read the entire script in <shell_script>.
2) Identify and record all variable expansions and indirect expansions
   (e.g., $VAR, ${VAR}, ${VAR:-x}, ${!VAR}) and ignore shell special parameters.
3) Identify all definitions/assignments of variables that occur in the script text
   (VAR=, local/declare/typeset, read VAR, for VAR in, while read VAR, etc.) and
   note the earliest point each variable is definitely defined.
4) Detect argument parsing and CLI-derived assignments (getopts, case "$1", shift loops) and
   mark variables as "CLI-bound" only if the RHS is derived from $1..$N, $@, $*, or $OPTARG.
5) Detect any forbidden/dynamic constructs (source, ., eval, declare -n, export, set -a, env,
   printenv). If present, mark FAIL.
6) For each variable expansion, decide whether it is proven defined before first use.
   If not proven defined, classify it as a suspected environment variable read.
7) If any suspected environment variable reads exist (or any forbidden/dynamic constructs exist),
   the script FAILS; otherwise PASS.
8) Produce the JSON output exactly as in <output_format>, including: pass/fail score,
   the sorted unique list of suspected environment variables, and the required summary text.
</steps>

<output_format>
{
  "score": "PASS|FAIL",
  "env_vars": ["VAR1", "VAR2"],
  "summary": "The script uses several environment variables: VAR1, VAR2. These variables should either be provided via \
command-line arguments, sourced from a configuration file, or removed if they are not required for the script's operation."
}
</output_format>

<example_output>
{
  "score": "FAIL",
  "env_vars": ["HOME", "TOKEN"],
  "summary": "The script uses several environment variables: HOME, TOKEN. These variables should either be provided via \
command-line arguments, sourced from a configuration file, or removed if they are not required for the script's operation."
}
</example_output>

<shell_script>
{{PASTE_SHELL_SCRIPT_HERE}}
</shell_script>"""

    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct-mlx",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "local",
    ) -> None:
        """Initialize the Autogen model client.

        Args:
            model: Model name in LM Studio
            base_url: LM Studio base URL
            api_key: API key (any value for local LLMs)
        """
        # Create OpenAI-compatible client for local LLM (direct, no agent wrapper)
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_capabilities={
                "vision": False,
                "function_calling": False,
                "json_output": False,
            },
        )
        self.model = model

        # Silent initialization

    async def analyze(self, script: ShellScript) -> AnalysisResult:
        """Analyze a shell script to determine parameter handling.

        Args:
            script: The shell script to analyze

        Returns:
            AnalysisResult with analysis
        """
        try:
            # Build prompt by replacing placeholder with script content
            prompt = self.SYSTEM_PROMPT.replace("{{PASTE_SHELL_SCRIPT_HERE}}", script.content)

            # Create single user message with complete prompt
            messages = [
                UserMessage(content=prompt, source="user"),
            ]

            # Call model directly (no agent overhead)
            response = await self.model_client.create(
                messages=messages,
                extra_create_args={"max_tokens": 32768},
            )

            # Extract response text
            response_text = response.content
            if isinstance(response_text, list):
                # Handle function calls - shouldn't happen but type system requires handling
                response_text = ""

            # Parse JSON response
            return self._parse_response(response_text, script)

        except Exception as e:
            logger.warning(f"Error analyzing {script.file_path}: {e}")
            return AnalysisResult(
                file_path=script.file_path,
                status="ERROR",
                env_vars=[],
                summary=f"Analysis failed: {str(e)}",
            )

    def _parse_response(self, response_text: str, script: ShellScript) -> AnalysisResult:
        """Parse JSON response from LLM.

        Args:
            response_text: Raw response from LLM
            script: Original script

        Returns:
            AnalysisResult
        """
        try:
            # Clean response - remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]  # Remove ```json
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]  # Remove ```
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]  # Remove trailing ```
            cleaned = cleaned.strip()

            # Try to find JSON in response
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = cleaned[start_idx:end_idx]

            # Try to clean common JSON issues
            import re

            # Remove trailing commas before } or ]
            json_str = re.sub(r",(\\s*[}\\]])", r"\\1", json_str)

            data = json.loads(json_str)

            return AnalysisResult(
                file_path=script.file_path,
                status=data.get("score", "UNKNOWN"),  # Map "score" to "status"
                env_vars=data.get("env_vars", []),
                summary=data.get("summary", ""),
            )

        except json.JSONDecodeError as e:
            # Log the actual response for debugging
            logger.warning(f"JSON decode error: {e}")
            logger.warning(f"Raw response (first 500 chars): {response_text[:500]}")

            # Default response if parsing fails
            return AnalysisResult(
                file_path=script.file_path,
                status="ERROR",
                env_vars=[],
                summary="Failed to parse LLM response",
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.warning(f"Response was (first 500 chars): {response_text[:500]}")
            # Default response if parsing fails
            return AnalysisResult(
                file_path=script.file_path,
                status="ERROR",
                env_vars=[],
                summary="Failed to parse LLM response",
            )

    async def analyze_batch(self, scripts: list[ShellScript]) -> list[AnalysisResult]:
        """Analyze multiple shell scripts.

        Args:
            scripts: List of scripts to analyze

        Returns:
            List of analysis results
        """
        results: list[AnalysisResult] = []

        for i, script in enumerate(scripts):
            print(f"Analyzing [{i + 1}/{len(scripts)}]: {script.file_path} ... ", end="", flush=True)

            result = await self.analyze(script)
            results.append(result)

            # Print status
            status_symbol = "✅" if result.status == "PASS" else "❌"
            print(f"{status_symbol} {result.status}")

        print()  # Final newline
        return results


# =============================================================================
# Class 4: ShellScriptAnalysisOrchestrator
# =============================================================================


class ShellScriptAnalysisOrchestrator:
    """Main orchestrator for shell script parameter analysis."""

    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct-mlx",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "local",
        use_cache: bool = True,
    ) -> None:
        """Initialize orchestrator.

        Args:
            model: Model name in LM Studio
            base_url: LM Studio base URL
            api_key: API key
            use_cache: Whether to use hash-based caching to skip unchanged files
        """
        self.finder = ShellScriptFinder()
        self.extractor = ShellScriptExtractor()
        self.analyzer = ShellScriptParameterAnalyzer(model=model, base_url=base_url, api_key=api_key)
        self.use_cache = use_cache
        self.cache = ScriptFileHashCache() if use_cache else None

    def run(self, root_path: str = ".") -> tuple[list[AnalysisResult], list[ShellScript]]:
        """Run complete shell script parameter analysis.

        Args:
            root_path: Root path to search for scripts

        Returns:
            Tuple of (all results, all scripts)
        """
        # Step 1: Find shell scripts
        script_files = self.finder.find_shell_scripts(root_path)

        # Step 1.5: Filter unchanged files if caching is enabled
        if self.use_cache and self.cache is not None:
            files_to_scan: list[str] = []
            skipped_files: list[str] = []
            for script_file in script_files:
                if self.cache.has_file_changed(script_file):
                    files_to_scan.append(script_file)
                    # Remove from cache before testing - will be re-added after analysis
                    self.cache.remove_file(script_file)
                else:
                    skipped_files.append(script_file)

            if skipped_files:
                print(f"ℹ️  Skipped {len(skipped_files)} unchanged files (use --no-cache to force re-scan)\n")

            script_files = files_to_scan

        # Step 2: Extract script content
        all_scripts: list[ShellScript] = []
        for script_file in script_files:
            script = self.extractor.extract(script_file)
            if script:
                all_scripts.append(script)

        # Step 3: Analyze with AI agent
        print(f"Analyzing {len(all_scripts)} shell script(s)...\n")
        results = asyncio.run(self.analyzer.analyze_batch(all_scripts))

        # Step 4: Update cache for scanned files (only cache files that passed)
        if self.use_cache and self.cache is not None:
            for result in results:
                # Only cache files with no violations - files with failures must be re-scanned
                if result.status == "PASS":
                    self.cache.update_file_hash(result.file_path, result.status)
                    print(f"✓ Cached clean file: {result.file_path}")

        return results, all_scripts


# =============================================================================
# Test Mode Handlers
# =============================================================================


def test_discover(root_path: str) -> None:
    """Test Phase 1: Discover shell script files."""
    print("=== Test Phase 1: Discover Shell Script Files ===")
    finder = ShellScriptFinder()
    script_files = finder.find_shell_scripts(root_path)
    print(f"\nFound {len(script_files)} shell script files:")
    for f in script_files:
        print(f"  - {f}")
    sys.exit(0)


def test_extract(root_path: str) -> None:
    """Test Phase 2: Extract script content from one file."""
    print("=== Test Phase 2: Extract Script Content ===")
    finder = ShellScriptFinder()
    script_files = finder.find_shell_scripts(root_path)
    if not script_files:
        print("No shell script files found")
        sys.exit(1)

    # Use first script file
    script_file = script_files[0]
    print(f"\nExtracting from: {script_file}")

    extractor = ShellScriptExtractor()
    script = extractor.extract(script_file)
    if script:
        print("\nScript extracted successfully:")
        print(f"  - File path: {script.file_path}")
        print(f"  - Line count: {script.line_count}")
        print(f"  - Content length: {len(script.content)} chars")
        print("\nFirst 500 characters of content:")
        print(script.content[:500])
    else:
        print("Failed to extract script")
    sys.exit(0)


def test_analyze(root_path: str, model: str, base_url: str, api_key: str) -> None:
    """Test Phase 3: Analyze one script."""
    print("=== Test Phase 3: Analyze Shell Script ===")
    finder = ShellScriptFinder()
    script_files = finder.find_shell_scripts(root_path)
    if not script_files:
        print("No shell script files found")
        sys.exit(1)

    # Extract from first file
    extractor = ShellScriptExtractor()
    script = extractor.extract(script_files[0])
    if not script:
        print("Failed to extract script")
        sys.exit(1)

    # Analyze first script
    analyzer = ShellScriptParameterAnalyzer(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    result = asyncio.run(analyzer.analyze(script))
    print("\nAnalysis result:")
    print(f"  File: {result.file_path}")
    print(f"  Status: {result.status}")
    print(f"  Environment variables: {result.env_vars}")
    print(f"  Summary: {result.summary}")
    sys.exit(0)


def run_full_analysis(root_path: str, model: str, base_url: str, api_key: str, use_cache: bool, file: str | None) -> None:
    """Run full analysis on scripts."""
    print("=== Shell Script Parameter Analysis ===")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")

    if file:
        print(f"Analyzing single file: {file}")

    print()

    if file:
        # Analyze single file
        extractor = ShellScriptExtractor()
        analyzer = ShellScriptParameterAnalyzer(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        script = extractor.extract(file)
        if not script:
            print(f"Failed to extract script from {file}")
            sys.exit(1)

        print(f"Analyzing {file}...\n")
        results = asyncio.run(analyzer.analyze_batch([script]))
    else:
        orchestrator = ShellScriptAnalysisOrchestrator(
            model=model,
            base_url=base_url,
            api_key=api_key,
            use_cache=use_cache,
        )
        results, _ = orchestrator.run(root_path)

    # Write markdown report
    report_writer = MarkdownReportWriter()
    report_writer.write_report(results, model)

    # Output results summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    failed_results = [r for r in results if r.status == "FAIL"]
    passed_results = [r for r in results if r.status == "PASS"]

    print(f"✅ Passed: {len(passed_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    print()

    if failed_results:
        print("Failed Scripts:")
        print("-" * 60)
        for result in failed_results:
            print(f"\nFile: {result.file_path} FAIL 🛑")
            print(f"Vars: {', '.join(result.env_vars)}")
            print()
            print(f"Issue: {result.summary}")
            print()
        print("📄 Full report: reports/shell_env_var_violations.md")
        print()
        sys.exit(1)
    else:
        print("✅ All scripts passed - no environment variable violations")
        print()
        sys.exit(0)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for shell script parameter detector."""
    parser = argparse.ArgumentParser(
        description="Analyze shell scripts for parameter handling (env vars vs CLI args) using AI (Autogen v0.7.5)"
    )
    _ = parser.add_argument(
        "--root-path",
        default=".",
        help="Root path to search for scripts (default: current dir)",
    )
    _ = parser.add_argument(
        "--file",
        default=None,
        help="Analyze a specific file instead of all scripts",
    )
    _ = parser.add_argument(
        "--test",
        choices=["discover", "extract", "analyze", "full"],
        default="full",
        help="Run specific test phase (default: full)",
    )
    _ = parser.add_argument(
        "--model",
        default=os.getenv("LM_STUDIO_MODEL", "qwen2.5-7b-instruct-mlx"),
        help="Model name (default: qwen2.5-7b-instruct-mlx)",
    )
    _ = parser.add_argument(
        "--base-url",
        default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        help="LM Studio base URL (default: http://localhost:1234/v1)",
    )
    _ = parser.add_argument(
        "--api-key",
        default=os.getenv("LM_STUDIO_API_KEY", "local"),
        help="API key (default: local)",
    )
    _ = parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable hash-based caching (re-scan all files)",
    )

    args = parser.parse_args()

    # Check API availability for tests that require the model
    if args.test in ["analyze", "full"] and not check_api_available(args.base_url):
        print(f"\n⚠️  WARNING: Cannot connect to LLM API at {args.base_url}")
        print("Please ensure:")
        print("  1. LM Studio (or your LLM server) is running")
        print("  2. The server is accessible at the configured base URL")
        print(f"  3. The API endpoint {args.base_url}/models is reachable")
        print("\nCannot run the script without an available model API.")
        sys.exit(2)

    try:
        if args.test == "discover":
            test_discover(args.root_path)
        elif args.test == "extract":
            test_extract(args.root_path)
        elif args.test == "analyze":
            test_analyze(args.root_path, args.model, args.base_url, args.api_key)
        else:
            run_full_analysis(args.root_path, args.model, args.base_url, args.api_key, not args.no_cache, args.file)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
