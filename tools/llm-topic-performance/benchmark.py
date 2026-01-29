#!/usr/bin/env python3
"""LLM Topic Extraction Performance Benchmark.

Queries LM Studio for available completion models, tests each on topic extraction
using the production prompt from src/topic_detection/agents/prompts.py,
and logs results with timing information.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import litellm  # noqa: E402

from src.topic_detection.agents.prompts import TopicDetectionPrompts  # noqa: E402

# Configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
REPORTS_DIR = PROJECT_ROOT / "reports" / "llm-topic-extraction-comparison"

# Model patterns to exclude
EXCLUDE_PATTERNS = [
    # Embedding models
    "embed",
    "bge-",
    "e5-",
    "gte-",
    "embedding",
    "Embedding",
    "BGE",
    "E5",
    "GTE",
    # Bulgarian models
    "BgGPT",
    "bggpt",
    # Dev/experimental models
    "Devstral",
    "devstral",
    # Translation models
    "translate",
    "Translate",
]

# Sample text for benchmarking (simulating a podcast transcript segment)
SAMPLE_TEXT = """
Artificial intelligence has transformed the healthcare industry in remarkable ways.
Machine learning algorithms can now detect diseases from medical imaging with accuracy
rivaling human experts. Natural language processing helps doctors analyze patient records
and medical literature more efficiently.

Climate change continues to impact global food security. Rising temperatures affect crop
yields, while extreme weather events disrupt supply chains. Scientists are developing
drought-resistant crop varieties and more sustainable farming practices to address these
challenges.

The global economy faces uncertainty as central banks adjust interest rates to combat
inflation. Supply chain disruptions from recent years continue to affect manufacturing
and consumer prices. Remote work trends have permanently altered commercial real estate
markets in major cities.

Space exploration has entered a new era with private companies leading ambitious missions.
Reusable rocket technology has dramatically reduced launch costs. Plans for lunar bases
and Mars colonies are progressing from science fiction toward reality.
"""


def get_available_models() -> list[str]:
    """Fetch available models from LM Studio API, excluding embedding models."""
    try:
        response = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        data = response.json()

        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            # Skip excluded models
            if any(pattern.lower() in model_id.lower() for pattern in EXCLUDE_PATTERNS):
                continue
            models.append(model_id)

        return sorted(models)
    except requests.RequestException as e:
        print(f"Error fetching models: {e}")
        sys.exit(1)


def parse_llm_response(response_text: str) -> dict:
    """Parse LLM response, handling thinking tags and markdown code blocks.

    Uses the same parsing logic as the production TopicExtractionAgent.
    """
    cleaned = response_text.strip()

    # Handle "thinking" models that wrap reasoning in <think> tags
    think_match = re.search(r"</think>\s*(.*)$", cleaned, re.DOTALL)
    if think_match:
        cleaned = think_match.group(1).strip()

    # Handle markdown code blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    if not cleaned:
        return {"parse_error": True, "error": "Empty response after cleaning", "raw": response_text[:500]}

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"parse_error": True, "error": str(e), "raw": response_text[:500]}


def extract_topics(model_id: str, text: str) -> tuple[dict[str, object], float]:
    """Extract topics from text using specified model and production prompts.

    Returns (response_data, elapsed_seconds).
    """
    messages = [
        {"role": "system", "content": TopicDetectionPrompts.getSystemPrompt()},
        {"role": "user", "content": TopicDetectionPrompts.getUserPrompt(text)},
    ]

    start_time = time.perf_counter()

    response = litellm.completion(
        model=f"openai/{model_id}",
        messages=messages,
        api_base=LM_STUDIO_BASE_URL,
        api_key="not-needed",
        temperature=0.1,
        max_tokens=1000,
    )

    elapsed = time.perf_counter() - start_time

    content = response.choices[0].message.content
    if content is None or content.strip() == "":
        return {"parse_error": True, "error": "Empty response from model"}, elapsed

    topics_data = parse_llm_response(content)
    return topics_data, elapsed


def wait_for_keypress(model_name: str) -> None:
    """Display next model and wait for user to press Enter."""
    print(f"\n{'=' * 60}")
    print(f"Next model to load: {model_name}")
    print("Press Enter to continue (or Ctrl+C to abort)...")
    print(f"{'=' * 60}")
    input()


def save_model_result(model_id: str, result: dict[str, object]) -> None:
    """Save individual model result to JSON file."""
    # Sanitize model name for filename
    safe_name = model_id.replace("/", "_").replace(":", "_")
    filepath = REPORTS_DIR / f"{safe_name}.json"

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved: {filepath}")


def save_performance_summary(results: list[dict[str, object]]) -> None:
    """Save performance summary to performance.json."""

    # Count topics for each result
    def count_topics(r: dict[str, object]) -> int:
        extracted = r.get("extracted_topics", {})
        if extracted.get("parse_error"):
            return 0
        return (
            len(extracted.get("high_level_topics", []))
            + len(extracted.get("mid_level_topics", []))
            + len(extracted.get("specific_topics", []))
        )

    summary = {
        "benchmark_date": datetime.now().isoformat(),
        "sample_text_length": len(SAMPLE_TEXT),
        "lm_studio_api": LM_STUDIO_BASE_URL,
        "models": [
            {
                "model": r["model"],
                "time_seconds": r["time_seconds"],
                "topics_extracted": count_topics(r),
                "should_index": r.get("extracted_topics", {}).get("should_index"),
                "success": not r.get("extracted_topics", {}).get("parse_error", False),
            }
            for r in results
        ],
    }

    # Sort by time (successful models first, then by speed)
    summary["models"].sort(key=lambda x: (not x["success"], x["time_seconds"] if x["time_seconds"] > 0 else 9999))

    filepath = REPORTS_DIR / "performance.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPerformance summary saved: {filepath}")


def main() -> None:
    """Run the LLM topic extraction benchmark."""
    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("LLM Topic Extraction Performance Benchmark")
    print("=" * 60)
    print(f"LM Studio API: {LM_STUDIO_BASE_URL}")
    print(f"Reports directory: {REPORTS_DIR}")
    print("Using production prompts from: src/topic_detection/agents/prompts.py")
    print()

    # Get available models
    print("Fetching available models...")
    models = get_available_models()

    if not models:
        print("No completion models found!")
        sys.exit(1)

    print(f"Found {len(models)} completion model(s):")
    for m in models:
        print(f"  - {m}")

    # Run benchmark
    results = []

    for i, model_id in enumerate(models):
        wait_for_keypress(model_id)

        print(f"\n[{i + 1}/{len(models)}] Testing: {model_id}")
        print("  Extracting topics...")

        try:
            topics_data, elapsed = extract_topics(model_id, SAMPLE_TEXT)

            result = {
                "model": model_id,
                "input_text": SAMPLE_TEXT.strip(),
                "extracted_topics": topics_data,
                "time_seconds": round(elapsed, 3),
                "timestamp": datetime.now().isoformat(),
            }

            results.append(result)
            save_model_result(model_id, result)

            print(f"  Time: {elapsed:.3f}s")

            if topics_data.get("parse_error"):
                print(f"  Warning: Parse error - {topics_data.get('error', 'unknown')}")
            else:
                print(f"  Should index: {topics_data.get('should_index', 'N/A')}")
                high = topics_data.get("high_level_topics", [])
                mid = topics_data.get("mid_level_topics", [])
                specific = topics_data.get("specific_topics", [])
                print(f"  Topics: {len(high)} high, {len(mid)} mid, {len(specific)} specific")
                if high:
                    print(f"    High-level: {', '.join(high[:3])}")

        except Exception as e:
            print(f"  Error: {e}")
            result = {
                "model": model_id,
                "input_text": SAMPLE_TEXT.strip(),
                "error": str(e),
                "time_seconds": -1,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)
            save_model_result(model_id, result)

    # Save performance summary
    save_performance_summary(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nResults by speed (fastest first):")

    sorted_results = sorted(
        [r for r in results if r["time_seconds"] > 0],
        key=lambda x: x["time_seconds"],
    )

    for i, r in enumerate(sorted_results, 1):
        status = "OK" if not r.get("extracted_topics", {}).get("parse_error") else "PARSE_ERR"
        print(f"  {i}. {r['model']}: {r['time_seconds']:.3f}s [{status}]")


if __name__ == "__main__":
    main()
