#!/usr/bin/env python3
"""Fake Unit Test Detector using Autogen v0.7.5

This tool uses a local LLM via Autogen to detect fraudulent unit tests that don't
actually test the code properly. It analyzes test files using AST parsing and
submits each test case to an AI agent for review.

Usage:
    python tools/detect_fake_tests.py                    # Run full detection
    python tools/detect_fake_tests.py --test discover    # Test file discovery
    python tools/detect_fake_tests.py --test extract     # Test case extraction
    python tools/detect_fake_tests.py --test analyze     # Test agent analysis
"""

import argparse
import ast
import asyncio
from datetime import datetime
import glob
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass

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
class TestCase:
    """Represents a single test function extracted from a file."""

    file_path: str
    line_number: int
    function_name: str
    code: str  # Full function code
    imports: str  # All necessary imports
    full_context: str  # imports + code (sent to LLM)


@dataclass
class DetectionResult:
    """Result of analyzing a test case for fraudulence."""

    file_path: str
    line_number: int
    function_name: str
    is_fake: bool
    confidence: float  # 0.0-1.0
    reason: str  # Why it's fake


# =============================================================================
# Class 0: TestFileHashCache
# =============================================================================


class TestFileHashCache:
    """Manages hash-based caching to skip unchanged test files."""

    def __init__(self, cache_file: str = ".cache/test_file_hashes.json"):
        """Initialize the hash cache.

        Args:
            cache_file: Path to the JSON cache file
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load the hash cache from JSON file.

        Returns:
            Dictionary mapping file paths to hash info
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file: {e}. Starting with empty cache.")
                return {}
        return {}

    def _save_cache(self):
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

    def update_file_hash(self, file_path: str, fake_count: int = 0):
        """Update the hash for a file after scanning.

        Args:
            file_path: Path to the file
            fake_count: Number of fake tests detected in this file
        """
        file_hash = self._calculate_file_hash(file_path)
        self.cache[file_path] = {
            "hash": file_hash,
            "last_checked": datetime.now().isoformat(),
            "fake_count": fake_count,
        }
        self._save_cache()

    def get_cached_info(self, file_path: str) -> dict | None:
        """Get cached information for a file.

        Args:
            file_path: Path to the file

        Returns:
            Cached info dictionary or None if not cached
        """
        return self.cache.get(file_path)

    def remove_file(self, file_path: str):
        """Remove a file from the cache.

        Args:
            file_path: Path to the file to remove
        """
        if file_path in self.cache:
            del self.cache[file_path]
            self._save_cache()


# =============================================================================
# Class 1: TestFileFinder
# =============================================================================


class TestFileFinder:
    """Finds all unit test files in the project."""

    def find_test_files(self, root_path: str = ".") -> list[str]:
        """Find all test_*.py files in the tests/ directory.

        Args:
            root_path: Root path to search from

        Returns:
            Sorted list of relative paths to test files
        """
        # Find all test_*.py files
        pattern = os.path.join(root_path, "tests", "**", "test_*.py")
        test_files = glob.glob(pattern, recursive=True)

        # Filter out __pycache__ and __init__.py
        test_files = [f for f in test_files if "__pycache__" not in f and not f.endswith("__init__.py")]

        # Keep as relative paths and sort
        test_files.sort()

        return test_files


# =============================================================================
# Class 2: TestCaseExtractor
# =============================================================================


class TestCaseExtractor:
    """Extracts individual test cases from Python test files using AST."""

    def extract(self, file_path: str) -> list[TestCase]:
        """Extract all test cases from a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of TestCase objects
        """
        try:
            # Read file content
            source_code = self._read_file(file_path)

            # Parse AST
            tree = ast.parse(source_code, filename=file_path)

            # Extract imports
            imports = self._extract_imports(tree, source_code)

            # Extract test functions
            test_cases = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_case = self._create_test_case(file_path, node, source_code, imports)
                    test_cases.append(test_case)

            # Sort by line number
            test_cases.sort(key=lambda tc: tc.line_number)

            return test_cases

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error extracting from {file_path}: {e}")
            return []

    def _read_file(self, file_path: str) -> str:
        """Read file handling encoding issues."""
        try:
            with open(file_path, encoding="utf-8-sig") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fall back to latin-1
            with open(file_path, encoding="latin-1") as f:
                return f.read()

    def _extract_imports(self, tree: ast.AST, source_code: str) -> str:
        """Extract all import statements from the module."""
        imports_list = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_code = ast.get_source_segment(source_code, node)
                if import_code:
                    imports_list.append(import_code)

        return "\n".join(imports_list)

    def _create_test_case(
        self,
        file_path: str,
        node: ast.FunctionDef,
        source_code: str,
        imports: str,
    ) -> TestCase:
        """Create a TestCase object from an AST function node."""
        # Extract function code
        function_code = ast.get_source_segment(source_code, node) or ""

        # Create full context (imports + function)
        full_context = f"{imports}\n\n{function_code}" if imports else function_code

        return TestCase(
            file_path=file_path,
            line_number=node.lineno,
            function_name=node.name,
            code=function_code,
            imports=imports,
            full_context=full_context,
        )


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
    """Generates markdown reports for fake test detection results."""

    def __init__(self, report_path: str = "reports/agentic_code_evals.md"):
        """Initialize the report writer.

        Args:
            report_path: Path to write the report (default: reports/agentic_code_evals.md)
        """
        self.report_path = report_path

    def write_report(self, fake_results: list[DetectionResult], test_cases: list[TestCase], model: str):
        """Write a markdown report of fake test detection results.

        Args:
            fake_results: List of DetectionResult objects for fake tests
            test_cases: List of all TestCase objects analyzed
            model: Model name used for detection
        """
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        # Create a mapping from (file_path, line_number) to TestCase for quick lookup
        test_case_map = {(tc.file_path, tc.line_number): tc for tc in test_cases}

        with open(self.report_path, "w", encoding="utf-8") as f:
            # Header
            _ = f.write("# Agentic Code Evaluations Report\n\n")
            _ = f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            _ = f.write(f"**Model:** {model}\n\n")

            # Summary
            _ = f.write("## Summary\n\n")
            _ = f.write(f"- **Total Tests Analyzed:** {len(test_cases)}\n")
            _ = f.write(f"- **Fake Tests Detected:** {len(fake_results)}\n")
            _ = f.write(f"- **Legitimate Tests:** {len(test_cases) - len(fake_results)}\n")

            if fake_results:
                _ = f.write(f"- **Detection Rate:** {len(fake_results) / len(test_cases) * 100:.1f}%\n\n")
            else:
                _ = f.write(f"- **Detection Rate:** 0.0%\n\n")

            # Fake tests details
            if fake_results:
                _ = f.write("## Fake Tests Detected\n\n")

                # Group by file
                from collections import defaultdict
                grouped = defaultdict(list)
                for result in fake_results:
                    grouped[result.file_path].append(result)

                for file_path in sorted(grouped.keys()):
                    _ = f.write(f"### {file_path}\n\n")
                    for result in sorted(grouped[file_path], key=lambda r: r.line_number):
                        _ = f.write(f"#### `{result.function_name}` (line {result.line_number})\n\n")
                        _ = f.write(f"**Confidence:** {result.confidence:.2f}\n\n")
                        _ = f.write(f"**Reason:** {result.reason}\n\n")

                        # Get the actual code from the test case map
                        test_case = test_case_map.get((result.file_path, result.line_number))
                        if test_case:
                            _ = f.write("**Code:**\n\n")
                            _ = f.write("```python\n")
                            _ = f.write(test_case.code)
                            _ = f.write("\n```\n\n")
            else:
                _ = f.write("## ✅ All Tests Are Legitimate\n\n")
                _ = f.write("No fake tests were detected in the codebase.\n\n")

        logger.info(f"✓ Report written to {self.report_path}")


# =============================================================================
# Class 3: FakeTestDetector (Autogen v0.7.5 Agent)
# =============================================================================


class FakeTestDetector:
    """Uses Autogen v0.7.5 to detect fake/fraudulent unit tests."""

    # System prompt defining what constitutes a "fake" test
    SYSTEM_PROMPT = """You are a code review expert specializing in test quality analysis.

Your task is to analyze Python unit tests and identify "fake" or "fraudulent" tests
that don't actually test the code properly. Common patterns:
1. Empty test body or only 'pass' statement
2. Test contains no assertions
3. Trivial assertions that don't test real behavior (assert True, assert 1==1, etc.)
4. Exception swallowing (try/except: pass) that hides test failures
5. Commented-out assertions or test code
6. Tests that don't call the function/code under test
7. Mock all dependencies without testing real logic
8. Hardcoded return values in mocks that bypass testing
9. Test comments without test code below
10. Tests that check mock calls instead of actual behavior
11. Tests with setup that prevents actual testing
12. Assertions on mock calls but not on business logic

Respond with JSON only (no other text):
{
    "is_fake": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation if is_fake=true, otherwise empty string"
}"""

    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct-mlx",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "local",
    ):
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
                "structured_output": False,
            },
        )
        self.model = model

        logger.info(f"✓ FakeTestDetector initialized (model={model}, base_url={base_url})")

    async def analyze(self, test_case: TestCase) -> DetectionResult:
        """Analyze a test case to determine if it's fake.

        Args:
            test_case: The test case to analyze

        Returns:
            DetectionResult with analysis
        """
        try:
            # Build prompt
            prompt = f"""Analyze this test function:

{test_case.full_context}

Is this a fake/fraudulent test? Respond with JSON only."""

            # Create messages for the model
            messages = [
                UserMessage(content=self.SYSTEM_PROMPT, source="system"),
                UserMessage(content=prompt, source="user"),
            ]

            # Call model directly (no agent overhead)
            response = await self.model_client.create(
                messages=messages,
                extra_create_args={"max_tokens": 32768},
            )

            # Extract response text
            response_text = response.content

            # Parse JSON response
            return self._parse_response(response_text, test_case)

        except Exception as e:
            logger.warning(f"Error analyzing {test_case.function_name}: {e}")
            return DetectionResult(
                file_path=test_case.file_path,
                line_number=test_case.line_number,
                function_name=test_case.function_name,
                is_fake=False,
                confidence=0.0,
                reason=f"Analysis failed: {str(e)}",
            )

    def _parse_response(self, response_text: str, test_case: TestCase) -> DetectionResult:
        """Parse JSON response from LLM.

        Args:
            response_text: Raw response from LLM
            test_case: Original test case

        Returns:
            DetectionResult
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

            # Fix missing "reason": key - LLM sometimes outputs just "" instead of "reason": ""
            # Case 1: Single "" before closing brace: , "" }
            # Case 2: Double "" (one on each line): , "" , "" }
            json_str_before = json_str

            # First, remove any duplicate "" entries (LLM sometimes outputs two empty strings)
            # Pattern: , "" , "" } -> , "" }
            json_str = re.sub(r',\s*""\s*,\s*""', r', ""', json_str)

            # Now fix the remaining "" to be "reason": ""
            # Pattern: ,\s*"" before closing } should be ,\s*"reason": ""
            json_str = re.sub(r',\s*""\s*}', r', "reason": "" }', json_str)

            # Remove trailing commas before } or ]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Debug log if we fixed something
            if json_str != json_str_before:
                logger.debug(f"Fixed JSON: {json_str_before[:100]} -> {json_str[:100]}")

            data = json.loads(json_str)

            # Log parsed data to verify
            reason = data.get("reason", "")
            if not reason:
                logger.debug(f"Empty reason in parsed data: {data}")
                logger.debug(f"Original response: {response_text[:200]}")

            return DetectionResult(
                file_path=test_case.file_path,
                line_number=test_case.line_number,
                function_name=test_case.function_name,
                is_fake=data.get("is_fake", False),
                confidence=data.get("confidence", 0.0),
                reason=reason,
            )

        except json.JSONDecodeError as e:
            # Log the actual response for debugging
            logger.warning(f"JSON decode error: {e}")
            logger.warning(f"Raw response (first 500 chars): {response_text[:500]}")
            logger.warning(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'N/A'}")

            # Try regex extraction as fallback
            try:
                import re
                is_fake_match = re.search(r'"is_fake"\s*:\s*(true|false)', response_text, re.IGNORECASE)
                confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response_text)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', response_text)

                if is_fake_match:
                    is_fake = is_fake_match.group(1).lower() == "true"
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                    reason = reason_match.group(1) if reason_match else ""

                    return DetectionResult(
                        file_path=test_case.file_path,
                        line_number=test_case.line_number,
                        function_name=test_case.function_name,
                        is_fake=is_fake,
                        confidence=confidence,
                        reason=reason,
                    )
            except Exception:
                pass  # Fall through to default response

            # Default to not fake if parsing fails
            return DetectionResult(
                file_path=test_case.file_path,
                line_number=test_case.line_number,
                function_name=test_case.function_name,
                is_fake=False,
                confidence=0.0,
                reason="Failed to parse LLM response",
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.warning(f"Response was (first 500 chars): {response_text[:500]}")
            # Default to not fake if parsing fails
            return DetectionResult(
                file_path=test_case.file_path,
                line_number=test_case.line_number,
                function_name=test_case.function_name,
                is_fake=False,
                confidence=0.0,
                reason="Failed to parse LLM response",
            )

    async def analyze_batch(self, test_cases: list[TestCase]) -> list[DetectionResult]:
        """Analyze multiple test cases.

        Args:
            test_cases: List of test cases to analyze

        Returns:
            List of detection results
        """
        results = []
        current_file = None
        file_has_fake = False
        previous_was_fake = False

        for i, test_case in enumerate(test_cases):
            # Print filename header when we encounter a new file
            if test_case.file_path != current_file:
                # Print success message for previous file if it had no fakes
                if current_file is not None:
                    if not file_has_fake:
                        print(" ✅ ALL CHECKS PASSED")
                    else:
                        print()  # New line only if there were fake tests

                print(f"ANALYZING FILE: {test_case.file_path} ", end="", flush=True)
                current_file = test_case.file_path
                file_has_fake = False  # Reset for new file
                previous_was_fake = False  # Reset for new file

            result = await self.analyze(test_case)
            results.append(result)

            # Print dot for pass, show error for fail
            if result.is_fake:
                file_has_fake = True
                if not previous_was_fake:
                    print()  # New line to break from dots (only if previous wasn't also fake)
                print("-" * 80)
                print(f"❌ FAKE TEST DETECTED @ {test_case.file_path}:{test_case.function_name}:{test_case.line_number}")
                print(f"   Reason: {result.reason} (confidence: {result.confidence:.2f})")
                print(f"   Code:")
                for line in test_case.code.split('\n'):
                    print(f"     {line}")
                print("-" * 80)
                previous_was_fake = True
            else:
                print(".", end="", flush=True)
                previous_was_fake = False

        # Print success message for last file if it had no fakes
        if not file_has_fake:
            print(" ✅ ALL CHECKS PASSED")

        print()  # Final newline
        return results


# =============================================================================
# Class 4: FakeTestOrchestrator
# =============================================================================


class FakeTestOrchestrator:
    """Main orchestrator for fake test detection."""

    def __init__(
        self,
        model: str = "qwen2.5-7b-instruct-mlx",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "local",
        use_cache: bool = True,
    ):
        """Initialize orchestrator.

        Args:
            model: Model name in LM Studio
            base_url: LM Studio base URL
            api_key: API key
            use_cache: Whether to use hash-based caching to skip unchanged files
        """
        self.finder = TestFileFinder()
        self.extractor = TestCaseExtractor()
        self.detector = FakeTestDetector(model=model, base_url=base_url, api_key=api_key)
        self.use_cache = use_cache
        self.cache = TestFileHashCache() if use_cache else None

    def run(self, root_path: str = ".") -> tuple[list[str], list[DetectionResult], list[DetectionResult], list[TestCase]]:
        """Run complete fake test detection.

        Args:
            root_path: Root path to search for tests

        Returns:
            Tuple of (fake test locations, fake results, all results, all test cases)
        """
        # Step 1: Find test files
        logger.info("Step 1: Finding test files...")
        test_files = self.finder.find_test_files(root_path)
        logger.info(f"Found {len(test_files)} test files")

        # Step 1.5: Filter unchanged files if caching is enabled
        if self.use_cache:
            files_to_scan = []
            skipped_files = []
            for test_file in test_files:
                if self.cache.has_file_changed(test_file):
                    files_to_scan.append(test_file)
                    # Remove from cache before testing - will be re-added only if all tests pass
                    self.cache.remove_file(test_file)
                else:
                    skipped_files.append(test_file)
                    cached_info = self.cache.get_cached_info(test_file)
                    logger.info(f"Skipping unchanged file: {test_file} (last checked: {cached_info['last_checked']})")

            if skipped_files:
                logger.info(f"Skipped {len(skipped_files)} unchanged files (use --no-cache to force re-scan)")

            test_files = files_to_scan

        logger.info(f"Will scan {len(test_files)} files")

        # Step 2: Extract test cases
        logger.info("Step 2: Extracting test cases...")
        all_test_cases = []
        file_test_case_map = {}  # Track which test cases belong to which file
        for test_file in test_files:
            test_cases = self.extractor.extract(test_file)
            all_test_cases.extend(test_cases)
            file_test_case_map[test_file] = test_cases
        logger.info(f"Found {len(all_test_cases)} test cases")

        # Step 3: Analyze with AI agent
        logger.info("Step 3: Analyzing test cases with LLM...")
        results = asyncio.run(self.detector.analyze_batch(all_test_cases))

        # Step 4: Filter fake tests
        fake = [r for r in results if r.is_fake and r.confidence > 0.5]

        # Step 4.5: Update cache for scanned files (only cache files that passed)
        if self.use_cache:
            for test_file in test_files:
                # Count fake tests in this file
                fake_count = sum(1 for r in fake if r.file_path == test_file)
                # Only cache files with no fake tests - files with failures must be re-scanned
                if fake_count == 0:
                    self.cache.update_file_hash(test_file, fake_count)
                    logger.info(f"Cached clean file: {test_file}")

        # Step 5: Format output
        output_lines = []
        for r in fake:
            output_lines.append(f"{r.file_path}:{r.line_number}")

        return output_lines, fake, results, all_test_cases


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for fake test detector."""
    parser = argparse.ArgumentParser(description="Detect fraudulent unit tests using AI (Autogen v0.7.5)")
    _ = parser.add_argument(
        "--root-path",
        default=".",
        help="Root path to search for tests (default: current dir)",
    )
    _ = parser.add_argument(
        "--file",
        default=None,
        help="Test a specific file instead of all tests",
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
    if args.test in ["analyze", "full"]:
        if not check_api_available(args.base_url):
            print(f"\n⚠️  WARNING: Cannot connect to LLM API at {args.base_url}")
            print("Please ensure:")
            print("  1. LM Studio (or your LLM server) is running")
            print("  2. The server is accessible at the configured base URL")
            print(f"  3. The API endpoint {args.base_url}/models is reachable")
            print("\nCannot run the script without an available model API.")
            sys.exit(2)

    try:
        if args.test == "discover":
            # Test Phase 1: Discover test files
            print("=== Test Phase 1: Discover Test Files ===")
            finder = TestFileFinder()
            test_files = finder.find_test_files(args.root_path)
            print(f"\nFound {len(test_files)} test files:")
            for f in test_files:
                print(f"  - {f}")
            sys.exit(0)

        elif args.test == "extract":
            # Test Phase 2: Extract test cases from one file
            print("=== Test Phase 2: Extract Test Cases ===")
            finder = TestFileFinder()
            test_files = finder.find_test_files(args.root_path)
            if not test_files:
                print("No test files found")
                sys.exit(1)

            # Use first test file
            test_file = test_files[0]
            print(f"\nExtracting from: {test_file}")

            extractor = TestCaseExtractor()
            test_cases = extractor.extract(test_file)
            print(f"\nFound {len(test_cases)} test cases:")
            for tc in test_cases:
                print(f"\n  - {tc.function_name} (line {tc.line_number})")
                print(f"    Context length: {len(tc.full_context)} chars")
            sys.exit(0)

        elif args.test == "analyze":
            # Test Phase 3: Analyze one test case
            print("=== Test Phase 3: Analyze Test Case ===")
            finder = TestFileFinder()
            test_files = finder.find_test_files(args.root_path)
            if not test_files:
                print("No test files found")
                sys.exit(1)

            # Extract from first file
            extractor = TestCaseExtractor()
            test_cases = extractor.extract(test_files[0])
            if not test_cases:
                print("No test cases found")
                sys.exit(1)

            # Analyze first test case
            detector = FakeTestDetector(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            )
            result = asyncio.run(detector.analyze(test_cases[0]))
            print("\nAnalysis result:")
            print(f"  Test: {result.function_name}")
            print(f"  Is fake: {result.is_fake}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reason: {result.reason}")
            sys.exit(0)

        else:
            # Full detection
            print("=== Fake Unit Test Detector ===")
            print(f"Model: {args.model}")
            print(f"Base URL: {args.base_url}")

            if args.file:
                print(f"Testing single file: {args.file}")

            print()

            if args.file:
                # Test single file
                extractor = TestCaseExtractor()
                detector = FakeTestDetector(
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                )

                test_cases = extractor.extract(args.file)
                print(f"Found {len(test_cases)} test cases in {args.file}\n")

                results = asyncio.run(detector.analyze_batch(test_cases))
                fake = [r for r in results if r.is_fake and r.confidence > 0.5]

                fake_tests = [f"{r.file_path}:{r.line_number}" for r in fake]
            else:
                orchestrator = FakeTestOrchestrator(
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    use_cache=not args.no_cache,
                )
                fake_tests, fake, results, test_cases = orchestrator.run(args.root_path)

            # Write markdown report
            report_writer = MarkdownReportWriter()
            report_writer.write_report(fake, test_cases, args.model)

            # Output results summary
            print()
            if fake_tests:
                print("=" * 60)
                print(f"⚠️  SUMMARY: {len(fake_tests)} FAKE TEST(S) DETECTED")
                print("=" * 60)
                for result in fake:
                    print(f"  {result.file_path}:{result.function_name}:{result.line_number} - {result.reason}")
                print()
                sys.exit(1)  # Exit with code 1 if fake tests found
            else:
                print("=" * 60)
                print("✅ SUMMARY: All tests are legitimate")
                print("=" * 60)
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
