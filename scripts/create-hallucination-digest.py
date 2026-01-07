#!/usr/bin/env python3
"""Create a digest of hallucination detection results grouped by file."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    """Generate hallucination digest grouped by file."""
    # Find all analysis JSON files
    transcripts_dir = Path(__file__).parent.parent / "data" / "downloads" / "transcripts"

    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}", file=sys.stderr)
        return 1

    # Collect all hallucinations grouped by file
    hallucinations_by_file = defaultdict(list)

    # Find all transcript-analysis directories
    analysis_dirs = list(transcripts_dir.glob("*/transcript-analysis"))

    for analysis_dir in analysis_dirs:
        channel_name = analysis_dir.parent.name

        # Process each JSON file in the analysis directory
        for json_file in analysis_dir.glob("*.json"):
            # Skip macOS resource fork files
            if json_file.name.startswith("._"):
                continue

            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract video name from source file
                source_file = Path(data["source_file"]).stem

                # Skip files with no hallucinations
                if not data["hallucinations_detected"]:
                    continue

                # Create file key
                file_key = f"{channel_name}/{source_file}.srt"

                # Process each hallucination
                for h in data["hallucinations_detected"]:
                    hallucinations_by_file[file_key].append({
                        "start_timestamp": h["start_timestamp"],
                        "end_timestamp": h["end_timestamp"],
                        "reps": h["repetition_count"],
                        "k": h["repetition_length"],
                        "window_word_count": h["window_word_count"],
                        "pattern": h["cleaned_text"],
                        "window": h["window_text"]
                    })

            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}", file=sys.stderr)
                continue

    if not hallucinations_by_file:
        print("No hallucinations found to digest.", file=sys.stderr)
        return 1

    # Generate markdown digest
    output_file = Path(__file__).parent.parent / "data" / "output" / "hallucination_digest.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_hallucinations = sum(len(v) for v in hallucinations_by_file.values())

    with output_file.open("w", encoding="utf-8") as f:
        # Header
        f.write("# Hallucination Detection Digest\n\n")
        f.write(f"Total Files with Hallucinations: {len(hallucinations_by_file)}\n\n")
        f.write(f"Total Hallucinations: {total_hallucinations}\n\n")
        f.write("---\n\n")

        # Process each file alphabetically
        for file_path in sorted(hallucinations_by_file.keys()):
            detections = hallucinations_by_file[file_path]

            f.write(f"## File: {file_path}\n\n")

            # Write each detection
            for h in detections:
                f.write(f"Start Timestamp: {h['start_timestamp']}\n")
                f.write(f"End Timestamp: {h['end_timestamp']}\n")
                f.write(f"Repetitions: {h['reps']}\n")
                f.write(f"Sequence Length: {h['k']}\n")
                f.write(f"Window Word Count: {h['window_word_count']}\n")
                f.write(f"Pattern: {h['pattern']}\n")
                f.write("Window Text:\n")
                f.write(f"{h['window']}\n\n")
                f.write("---\n\n")

    print(f"✅ Digest created: {output_file}")
    print(f"   Total files: {len(hallucinations_by_file)}")
    print(f"   Total hallucinations: {total_hallucinations}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
