#!/usr/bin/env python3
"""Transcribe audio files with language-specific translation prompts.

This script orchestrates the transcription of audio files from configured YouTube
channels using MLX Whisper. It supports multi-language transcription with automatic
translation to English for non-English content.

Features:
- Language-specific prompt templates
- Automatic translation for non-English videos
- Language grouping to minimize model switching
- Idempotent operation (skips existing transcripts)
- Comprehensive error handling and progress tracking
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config  # noqa: E402
from src.transcription.argument_helper import TranscriptionArgumentHelper  # noqa: E402
from src.util.fs_util import FSUtil  # noqa: E402
from src.util.whisper_languages import WhisperLanguages  # noqa: E402


def sanitize_channel_name(name: str) -> str:
    """Sanitize channel name for filesystem use.

    Args:
        name: The channel name to sanitize.

    Returns:
        A sanitized version of the name safe for use in filesystem paths.
    """
    # Replace spaces and special characters with underscores
    # Keep alphanumeric characters, hyphens, and underscores
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def transcribe_single_file(
    wav_file: Path,
    model_repo: str,
    output_dir: Path,
    task: str,
    language: str,
    hallucination_silence_threshold: float,
    compression_ratio_threshold: float,
    initial_prompt: str,
    verbose: bool,
) -> bool:
    """Call shell script to transcribe a single file.

    Args:
        wav_file: Path to WAV audio file
        model_repo: Hugging Face model repository
        output_dir: Temporary output directory
        task: Either "transcribe" or "translate"
        language: ISO language code
        hallucination_silence_threshold: Silence threshold for hallucination detection
        compression_ratio_threshold: Compression ratio threshold
        initial_prompt: Prompt text for Whisper
        verbose: Whether to print detailed output

    Returns:
        True on success, False on failure
    """
    script_path = Path(__file__).parent / "transcribe_single.sh"

    cmd = [
        "bash",
        str(script_path),
        str(wav_file),
        model_repo,
        str(output_dir),
        task,
        language,
        str(hallucination_silence_threshold),
        str(compression_ratio_threshold),
        initial_prompt,
    ]

    # Don't capture output so user can see real-time transcription progress
    result = subprocess.run(cmd)

    if result.returncode != 0:
        return False

    return True


def process_single_wav_file(  # noqa: C901
    wav_file: Path,
    channel_name: str,
    channel_transcripts_dir: Path,
    metadata_dir: Path,
    lang: str,
    language_name: str,
    model_repo: str,
    task: str,
    use_youtube_metadata: bool,
    description_max_length: int,
    metadata_video_subdir: str,
    hallucination_silence_threshold: float,
    compression_ratio_threshold: float,
    sleep_between_files: int,
    verbose: bool,
) -> tuple[bool, int]:
    """Process a single WAV file for transcription.

    Returns:
        Tuple of (success, files_moved_count)
    """
    base_name = wav_file.stem

    # Skip if transcript already exists (idempotent)
    if (channel_transcripts_dir / f"{base_name}.txt").exists():
        if verbose:
            print(f"    ⏭️  Skipping: {base_name}.wav (transcript already exists)")
        return (True, 0)

    print(f"    🎙️  Transcribing: {base_name}.wav")

    # Build initial prompt from metadata
    initial_prompt = ""
    if use_youtube_metadata:
        metadata_file = metadata_dir / channel_name / metadata_video_subdir / f"{base_name}.info.json"

        if verbose:
            print(f"      🔍 Looking for metadata: {metadata_file}")

        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)

                title = metadata.get("title", "")
                description = metadata.get("description", "")[:description_max_length]

                if verbose:
                    print(f"      ✅ Metadata found - Title: {title[:50]}...")

                if title:
                    initial_prompt = TranscriptionArgumentHelper.build_prompt(
                        language=lang,
                        language_name=language_name,
                        title=title,
                        description=description,
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"      ⚠️  Warning: Failed to parse metadata: {e}", file=sys.stderr)
        else:
            print(f"      🚨 ERROR: Metadata file not found: {metadata_file}")
            print("      💡 Run 'bash scripts/move-metadata.sh' to move .info.json files")
            return (False, 0)

    # Create temp output directory
    temp_output_dir = Path(tempfile.mkdtemp(prefix="transcribe_"))

    try:
        # Transcribe
        success = transcribe_single_file(
            wav_file,
            model_repo,
            temp_output_dir,
            task,
            lang,
            hallucination_silence_threshold,
            compression_ratio_threshold,
            initial_prompt,
            verbose,
        )

        if success:
            # Move generated files from temp to transcripts directory
            moved = 0
            for ext in ["txt", "srt", "vtt", "tsv", "json"]:
                for generated_file in temp_output_dir.glob(f"*.{ext}"):
                    target_path = channel_transcripts_dir / f"{base_name}.{ext}"
                    shutil.move(str(generated_file), str(target_path))
                    moved += 1

            if moved > 0:
                print(f"      ✅ Success: {base_name}")
                time.sleep(sleep_between_files)
                return (True, moved)
            else:
                print(f"      ❌ Failed: No transcript generated for {base_name}.wav")
                return (False, 0)
        else:
            print(f"      ❌ Failed to transcribe {base_name}.wav")
            return (False, 0)

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def main() -> int:  # noqa: C901
    """Main orchestration logic.

    Returns:
        0 if all files succeeded, 1 if any failed
    """
    # Load configuration
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    try:
        config = Config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Get paths from config
    audio_dir = project_root / config.getDataDownloadsAudioDir()
    transcripts_dir = project_root / config.getDataDownloadsTranscriptsDir()
    metadata_dir = project_root / config.getDataDownloadsMetadataDir()

    # Get transcription settings from config
    model_en_repo = config.getTranscriptionModelEnRepo()
    model_multi_repo = config.getTranscriptionModelMultiRepo()
    hallucination_silence_threshold = config.getTranscriptionHallucinationSilenceThreshold()
    compression_ratio_threshold = config.getTranscriptionCompressionRatioThreshold()
    use_youtube_metadata = config.getTranscriptionUseYoutubeMetadata()
    description_max_length = config.getTranscriptionDescriptionMaxLength()
    sleep_between_files = config.getTranscriptionSleepBetweenFiles()
    metadata_video_subdir = config.getTranscriptionMetadataVideoSubdir()
    verbose = config.getTranscriptionVerbose()

    # Get language names from WhisperLanguages
    supported_languages = WhisperLanguages.get_supported_languages()

    # Group channels by language
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for channel in config.get_channels():
        groups[channel.language].append(
            {
                "name": channel.name,
                "sanitized_name": sanitize_channel_name(channel.name),
                "language": channel.language,
            }
        )

    # Print header
    print("Starting batch transcription with language grouping")
    print(f"English model: {config.getTranscriptionModelEnName()} ({model_en_repo})")
    print(f"Multilingual model: {config.getTranscriptionModelMultiName()} ({model_multi_repo})")
    print(f"Hallucination silence threshold: {hallucination_silence_threshold}s")
    print(f"Compression ratio threshold: {compression_ratio_threshold}")
    print(f"Use YouTube metadata: {use_youtube_metadata}")
    print("=" * 56)
    print()

    # Counters for final summary
    total_processed = 0
    total_failed = 0

    # Process each language group
    for lang in sorted(groups.keys()):
        print()
        print("=" * 56)
        print(f"  Processing Language Group: {lang}")
        print("=" * 56)

        # Determine model and task for this language
        if lang == "en":
            model_repo = model_en_repo
            task = "transcribe"
        else:
            model_repo = model_multi_repo
            task = "translate"

        language_name = supported_languages.get(lang, lang)

        print(f"  Model: {config.getTranscriptionModelEnName() if lang == 'en' else config.getTranscriptionModelMultiName()}")
        print(f"  Task: {task}")
        print(f"  Language: {lang}")
        print()

        # Process each channel in this language group
        for channel_info in groups[lang]:
            channel_name = channel_info["sanitized_name"]
            channel_audio_dir = audio_dir / channel_name

            if not channel_audio_dir.exists():
                print(f"⚠️  Channel directory not found: {channel_audio_dir} (skipping)")
                continue

            print(f"  📁 Channel: {channel_name}")
            print("  ---")

            # Create transcripts directory for this channel
            channel_transcripts_dir = transcripts_dir / channel_name
            FSUtil.ensure_directory_exists(channel_transcripts_dir)

            # Counter for skipped files in this channel
            skipped_count = 0

            # Find WAV files (filter out macOS ._ files)
            wav_files = FSUtil.find_files_by_extension(channel_audio_dir, ".wav", recursive=False)
            wav_files = [f for f in wav_files if not f.name.startswith("._")]

            for wav_file in wav_files:
                success, moved_count = process_single_wav_file(
                    wav_file,
                    channel_name,
                    channel_transcripts_dir,
                    metadata_dir,
                    lang,
                    language_name,
                    model_repo,
                    task,
                    use_youtube_metadata,
                    description_max_length,
                    metadata_video_subdir,
                    hallucination_silence_threshold,
                    compression_ratio_threshold,
                    sleep_between_files,
                    verbose,
                )

                if moved_count == 0 and success:
                    # File was skipped (already exists)
                    skipped_count += 1
                elif success:
                    # File was processed successfully
                    total_processed += 1
                else:
                    # File processing failed
                    total_failed += 1

            # Print skip summary if any files were skipped
            if skipped_count > 0:
                print(f"  ⏭️  Skipped {skipped_count} file(s) (transcript already exists)")

            print()

    # Final summary
    print()
    print("=" * 56)
    print("  Transcription Complete")
    print("=" * 56)
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print()

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
