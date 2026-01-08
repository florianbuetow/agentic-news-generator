"""Tests for main.py functions."""

from pathlib import Path
from unittest.mock import MagicMock

from src.agents.topic_segmentation.models import (
    AgentSegmentationResponse,
    CriticRating,
    SegmentationAttempt,
    SegmentationResult,
    TopicBlock,
)
from src.main import is_transcript_already_processed, process_transcript_files


def test_is_transcript_already_processed_returns_false_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """Test that function returns False when output file does not exist."""
    output_dir = tmp_path / "output"
    video_id = "test_video"
    channel_name = "test_channel"

    result = is_transcript_already_processed(video_id, channel_name, output_dir)

    assert result is False


def test_is_transcript_already_processed_returns_true_when_file_exists(
    tmp_path: Path,
) -> None:
    """Test that function returns True when output file exists."""
    output_dir = tmp_path / "output"
    video_id = "test_video"
    channel_name = "test_channel"

    # Create the output file
    channel_dir = output_dir / channel_name
    channel_dir.mkdir(parents=True)
    output_file = channel_dir / f"{video_id}.json"
    output_file.write_text('{"test": "data"}')

    result = is_transcript_already_processed(video_id, channel_name, output_dir)

    assert result is True


def test_is_transcript_already_processed_with_different_channel(
    tmp_path: Path,
) -> None:
    """Test that function checks the correct channel directory."""
    output_dir = tmp_path / "output"
    video_id = "test_video"
    channel_name_1 = "channel_1"
    channel_name_2 = "channel_2"

    # Create file for channel_1
    channel_dir_1 = output_dir / channel_name_1
    channel_dir_1.mkdir(parents=True)
    output_file_1 = channel_dir_1 / f"{video_id}.json"
    output_file_1.write_text('{"test": "data"}')

    # Check channel_1 (should exist)
    result_1 = is_transcript_already_processed(video_id, channel_name_1, output_dir)
    assert result_1 is True

    # Check channel_2 (should not exist)
    result_2 = is_transcript_already_processed(video_id, channel_name_2, output_dir)
    assert result_2 is False


def test_is_transcript_already_processed_with_different_video_id(
    tmp_path: Path,
) -> None:
    """Test that function checks the correct video ID."""
    output_dir = tmp_path / "output"
    video_id_1 = "video_1"
    video_id_2 = "video_2"
    channel_name = "test_channel"

    # Create file for video_1
    channel_dir = output_dir / channel_name
    channel_dir.mkdir(parents=True)
    output_file_1 = channel_dir / f"{video_id_1}.json"
    output_file_1.write_text('{"test": "data"}')

    # Check video_1 (should exist)
    result_1 = is_transcript_already_processed(video_id_1, channel_name, output_dir)
    assert result_1 is True

    # Check video_2 (should not exist)
    result_2 = is_transcript_already_processed(video_id_2, channel_name, output_dir)
    assert result_2 is False


def test_process_transcript_files_skips_already_processed_files(
    tmp_path: Path,
) -> None:
    """Test that already-processed files are skipped."""
    output_dir = tmp_path / "output"
    channel_name = "test_channel"

    # Create transcript files
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    file1 = transcripts_dir / "video1.txt"
    file2 = transcripts_dir / "video2.txt"
    file3 = transcripts_dir / "video3.txt"
    file1.write_text("transcript 1")
    file2.write_text("transcript 2")
    file3.write_text("transcript 3")

    # Mark file1 and file2 as already processed
    channel_dir = output_dir / channel_name
    channel_dir.mkdir(parents=True)
    (channel_dir / "video1.json").write_text('{"processed": true}')
    (channel_dir / "video2.json").write_text('{"processed": true}')

    # Mock orchestrator with valid response
    mock_orchestrator = MagicMock()
    mock_segment = TopicBlock(
        id=1,
        start="00:00:00,000",
        end="00:00:10,000",
        topic="test-topic",
        summary="Test summary",
    )
    mock_response = AgentSegmentationResponse(segments=[mock_segment])
    mock_critic = CriticRating.model_validate(
        {"rating": "great", "pass": True, "reasoning": "Good test", "improvement_suggestions": "None"}
    )
    mock_attempt = SegmentationAttempt(attempt_number=1, response=mock_response, critic_rating=mock_critic)
    mock_result = SegmentationResult(success=True, attempts=[mock_attempt], best_attempt=mock_attempt, failure_reason=None)
    mock_orchestrator.segment_transcript.return_value = mock_result

    # Process files
    success_count, failure_count, failures = process_transcript_files(
        [file1, file2, file3],
        channel_name,
        mock_orchestrator,
        output_dir,
    )

    # All 3 files return success (2 skipped + 1 would-be-processed with mock)
    assert success_count == 3
    assert failure_count == 0
    assert failures == []


def test_process_transcript_files_processes_all_new_files(
    tmp_path: Path,
) -> None:
    """Test that all new files are processed when none exist."""
    output_dir = tmp_path / "output"
    channel_name = "test_channel"

    # Create transcript files
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    file1 = transcripts_dir / "video1.txt"
    file2 = transcripts_dir / "video2.txt"
    file1.write_text("transcript 1")
    file2.write_text("transcript 2")

    # Mock orchestrator with valid response
    mock_orchestrator = MagicMock()
    mock_segment = TopicBlock(
        id=1,
        start="00:00:00,000",
        end="00:00:10,000",
        topic="test-topic",
        summary="Test summary",
    )
    mock_response = AgentSegmentationResponse(segments=[mock_segment])
    mock_critic = CriticRating.model_validate(
        {"rating": "great", "pass": True, "reasoning": "Good test", "improvement_suggestions": "None"}
    )
    mock_attempt = SegmentationAttempt(attempt_number=1, response=mock_response, critic_rating=mock_critic)
    mock_result = SegmentationResult(success=True, attempts=[mock_attempt], best_attempt=mock_attempt, failure_reason=None)
    mock_orchestrator.segment_transcript.return_value = mock_result

    # Process files
    success_count, failure_count, failures = process_transcript_files(
        [file1, file2],
        channel_name,
        mock_orchestrator,
        output_dir,
    )

    # Both files should be processed
    assert success_count == 2
    assert failure_count == 0
    assert failures == []


def test_process_transcript_files_handles_empty_list(
    tmp_path: Path,
) -> None:
    """Test that empty file list is handled correctly."""
    output_dir = tmp_path / "output"
    channel_name = "test_channel"

    # Mock orchestrator
    mock_orchestrator = MagicMock()

    # Process empty list
    success_count, failure_count, failures = process_transcript_files(
        [],
        channel_name,
        mock_orchestrator,
        output_dir,
    )

    assert success_count == 0
    assert failure_count == 0
    assert failures == []
