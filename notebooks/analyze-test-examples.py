#!/usr/bin/env python3
"""Extract test examples and analyze them for repetition metrics.

This script:
1. Extracts all text examples from unit tests
2. Runs RepetitionDetector on each example
3. Computes repetition count and sequence length (k)
4. Generates a CSV file with metrics
"""

import csv
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processing.repetition_detector import RepetitionDetector

# Test examples extracted from tests/test_repetition_detector.py
# Format: (text, is_hallucination, description)
TEST_EXAMPLES: list[tuple[str, int, str]] = [
    # POSITIVE EXAMPLES (is_hallucination=1)
    (
        " ".join(["you know, like,"] * 15),
        1,
        "Massive loop: 'you know, like,' × 15",
    ),
    (
        " ".join(["Yeah."] * 40),
        1,
        "Single word massive repetition: 'Yeah.' × 40",
    ),
    (
        " ".join(["and,"] * 40),
        1,
        "Single word massive repetition: 'and,' × 40",
    ),
    (
        " ".join(["We have to be able to do this with the X or Y."] * 10),
        1,
        "Long phrase repeated 10 times",
    ),
    (
        " ".join(["word"] * 11),
        1,
        "Single word × 11 (score=11, just above threshold)",
    ),
    (
        " ".join(["this is a long phrase"] * 5),
        1,
        "5-word phrase × 5 reps (score=25)",
    ),
    (
        " ".join(["customer data for performance monitoring"] * 10),
        1,
        "Technical phrase × 10 reps",
    ),
    (
        (
            "Because if there was another way to solve it, then what you would expect "
            "is that as soon as you get to the stage of prokaryotes that have other "
            "niches that they could colonize, if only they could drive towards "
            "complexity, this would somehow be solved. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah."
        ),
        1,
        "Real hallucination: Nick Lane 'Yeah.' pattern",
    ),
    (
        (
            "I don't know what you are saying. I would like to thank you for your "
            "support and for your notifications. I appreciate your support and your "
            "support. I would like to thank you for your support and your support "
            "for the data that you are sharing. I would like to thank you for your "
            "support and your support. I would like to thank you for your support "
            "and your support. I would like to thank you for your support. Thank you "
            "for your support. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your motivation."
        ),
        1,
        "Real hallucination: 'thank you for your support' pattern",
    ),
    (
        (
            "Thank you. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your support. "
            "I would like to thank you for your support. I would like to thank you for "
            "your support. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your support. "
            "I would like to thank you for your support. I would like to thank you for "
            "your support. Thank you. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support. Thank you. "
            "Thank you. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support."
        ),
        1,
        "Real hallucination: extensive 'I would like to thank you' pattern",
    ),
    # NEGATIVE EXAMPLES (is_hallucination=0)
    (
        "as like a, as like a, an object schema",
        0,
        "Natural stutter: 2x repetition (score=6 < 10)",
    ),
    (
        "for this work, for this work, for this work, the xi",
        0,
        "Ambiguous 3x repetition (score=9 < 10)",
    ),
    (
        " ".join(["word"] * 5),
        0,
        "Single word × 5 (score=5 < 10, though reps=5 >= min)",
    ),
    (
        " ".join(["word pair"] * 5),
        0,
        "Threshold boundary: k=2 × 5 = 10 (not > 10)",
    ),
    (
        " ".join(["test pattern here"] * 3),
        0,
        "3-word phrase × 3 (score=9 < 10, reps=3 < 5)",
    ),
    (
        " ".join(["word phrase pattern"] * 4),
        0,
        "3-word phrase × 4 (score=12 > 10, but reps=4 < 5)",
    ),
    (
        "because I have to I have to generate the chunk",
        0,
        "Natural stutter: 'I have to' × 2 (score=6)",
    ),
    (
        "that you can safely you can safely scale beta with N",
        0,
        "Natural repetition: 'you can safely' × 2 (score=6)",
    ),
    (
        "it. I don't think I don't think it's exactly like",
        0,
        "Natural stutter: 'I don't think' × 2 (score=6)",
    ),
    (
        "Thank you so much. Thank you so much.",
        0,
        "Polite closing: 'Thank you so much' × 2 (score=8)",
    ),
    (
        "showing up at Los Alamos? Who cares about you? Who cares about you?",
        0,
        "Rhetorical repetition: 'Who cares about you?' × 2 (score=8)",
    ),
    (
        "American, the sky is falling, the sky is falling, right?",
        0,
        "Idiom emphasis: 'the sky is falling' × 2 (score=8)",
    ),
    (
        (
            "Sver intend to serve, you know, act or behave, right, then you still "
            "need to be, you know, you need to be able to predict the consequences "
            "of your action. The more tightly linked your actions or your"
        ),
        0,
        "Natural speech with filler 'you know' pattern",
    ),
    (
        "I mean, he was like, it's weird that you don't use value functions, right?",
        0,
        "Casual conversation with varied phrasing",
    ),
    (
        (
            "ever intend to serve, you know, act or behave, right, then you still "
            "need to be, you know, you need to be able to predict the consequences "
            "of your action."
        ),
        0,
        "Natural speech: 'you need to be' in different contexts",
    ),
    (
        "expand your search. I would love to do that. I would love to do that. Yes",
        0,
        "Enthusiasm: 'I would love to do that' × 2 (borderline score=12)",
    ),
    (
        "maybe a percent of a percent of the speed of light",
        0,
        "Technical phrase: 'a percent of' × 2 (score=6)",
    ),
    (
        "I think this is like this is like very connected to",
        0,
        "Natural filler: 'this is like' × 2 (score=6)",
    ),
    (
        "We're very, we're very ready for it.",
        0,
        "Punctuation variation prevents detection",
    ),
    (
        "So we need to understand. so we need to understand.",
        0,
        "Capitalization variation prevents detection",
    ),
    (
        "word word word and phrase phrase phrase phrase and thing thing",
        0,
        "Multiple low-score patterns (3, 4, 2 all < 10)",
    ),
    (
        "apple banana apple cherry",
        0,
        "Non-consecutive repetition (not hallucination)",
    ),
    (
        "",
        0,
        "Empty text",
    ),
    (
        "hello",
        0,
        "Single word, no repetition",
    ),
    (
        "apple banana cherry apple banana cherry fig apple banana cherry",
        0,
        "Non-consecutive: 'apple banana cherry' scattered",
    ),
]


def analyze_text(text: str, detector: RepetitionDetector) -> tuple[int, int, int]:
    """Analyze text and return max repetition metrics.

    Args:
        text: Text to analyze.
        detector: RepetitionDetector instance.

    Returns:
        Tuple of (max_k, max_reps, max_score).
        If no patterns found, returns (0, 0, 0).
    """
    if not text.strip():
        return (0, 0, 0)

    # Use the public detect_hallucinations method which returns patterns with scores
    hallucinations = detector.detect_hallucinations(text)

    if not hallucinations:
        return (0, 0, 0)

    # Find the pattern with the highest score
    max_k = 0
    max_reps = 0
    max_score = 0

    for pattern in hallucinations:
        # pattern is a dict with keys: pattern, k, repetitions, score, start_idx, end_idx
        k = pattern["k"]
        reps = pattern["repetitions"]
        score = pattern["score"]

        if score > max_score:
            max_k = k
            max_reps = reps
            max_score = score

    return (max_k, max_reps, max_score)


def main() -> int:
    """Main entry point."""
    # Initialize detector with default parameters
    detector = RepetitionDetector(min_k=1, min_repetitions=5)

    # Prepare output file
    output_file = (
        Path(__file__).parent.parent / "data" / "analyze" / "hallucinations" / "test_examples_analysis.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(TEST_EXAMPLES)} test examples...")
    print(f"Output file: {output_file}")
    print()

    # Open CSV file for writing
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow([
            "repetitions",
            "sequence_length",
            "score",
            "is_hallucination",
            "text",
            "description",
        ])

        # Process each example
        positive_count = 0
        negative_count = 0

        for text, is_hallucination, description in TEST_EXAMPLES:
            k, reps, score = analyze_text(text, detector)

            # Write row
            writer.writerow([
                reps,
                k,
                score,
                is_hallucination,
                text[:200] if len(text) > 200 else text,  # Truncate long text
                description,
            ])

            if is_hallucination:
                positive_count += 1
            else:
                negative_count += 1

            # Print progress
            status = "✓ HALLUCINATION" if is_hallucination else "○ NORMAL"
            print(f"{status} | k={k:2d} reps={reps:2d} score={score:3d} | {description[:60]}")

    print()
    print("=" * 70)
    print("Analysis complete!")
    print(f"  Total examples: {len(TEST_EXAMPLES)}")
    print(f"  Positive (hallucination): {positive_count}")
    print(f"  Negative (normal speech): {negative_count}")
    print(f"  Output: {output_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
