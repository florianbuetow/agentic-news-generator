#!/usr/bin/env python3
"""Create a digest of hallucination detection results grouped by file."""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config


def main() -> int:  # noqa: C901
    """Generate hallucination digest grouped by file."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Load Config class
    config = Config(config_path)

    # Get hallucination detection output directory from Config
    transcripts_hallucinations_dir = config.getDataDownloadsTranscriptsHallucinationsDir()

    if not transcripts_hallucinations_dir.exists():
        print(f"Error: Hallucination output directory not found: {transcripts_hallucinations_dir}", file=sys.stderr)
        return 1

    # Collect all hallucinations grouped by file
    hallucinations_by_file = defaultdict(list)

    # Find all channel directories (each contains hallucination JSON files)
    analysis_dirs = [d for d in transcripts_hallucinations_dir.glob("*") if d.is_dir()]

    for analysis_dir in analysis_dirs:
        channel_name = analysis_dir.name

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
                    hallucinations_by_file[file_key].append(
                        {
                            "start_timestamp": h["start_timestamp"],
                            "end_timestamp": h["end_timestamp"],
                            "reps": h["repetition_count"],
                            "k": h["repetition_length"],
                            "window_word_count": h["window_word_count"],
                            "pattern": h["cleaned_text"],
                            "window": h["window_text"],
                        }
                    )

            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}", file=sys.stderr)
                continue

    if not hallucinations_by_file:
        print("No hallucinations found to digest.", file=sys.stderr)
        return 1

    # Generate markdown digest
    output_file = config.getDataOutputDir() / "hallucination_digest.md"
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
