from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from src.config import Config
from src.pipeline_integrity import (
    find_filtered_file_offenders,
    load_filtered_video_ids,
    remove_filtered_artifacts,
    run_pipeline_integrity,
)


class _Config:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.videos_dir = data_dir / "downloads" / "videos"
        self.audio_dir = data_dir / "downloads" / "audio"
        self.transcripts_dir = data_dir / "downloads" / "transcripts"
        self.hallucinations_dir = data_dir / "downloads" / "transcripts-hallucinations"
        self.cleaned_dir = data_dir / "downloads" / "transcripts_cleaned"
        self.summaries_dir = data_dir / "downloads" / "transcripts_summaries"
        self.metadata_dir = data_dir / "downloads" / "metadata"
        self.archive_videos_dir = data_dir / "archive" / "videos"
        self.reports_dir = data_dir.parent / "reports"

        for directory in (
            self.videos_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.hallucinations_dir,
            self.cleaned_dir,
            self.summaries_dir,
            self.metadata_dir,
            self.archive_videos_dir,
        ):
            directory.mkdir(parents=True)

    def get_data_dir(self) -> Path:
        return self.data_dir

    def get_data_downloads_videos_dir(self) -> Path:
        return self.videos_dir

    def get_data_downloads_audio_dir(self) -> Path:
        return self.audio_dir

    def get_data_downloads_transcripts_dir(self) -> Path:
        return self.transcripts_dir

    def get_data_downloads_transcripts_hallucinations_dir(self) -> Path:
        return self.hallucinations_dir

    def get_data_downloads_transcripts_cleaned_dir(self) -> Path:
        return self.cleaned_dir

    def get_data_downloads_transcripts_summaries_dir(self) -> Path:
        return self.summaries_dir

    def get_data_downloads_metadata_dir(self) -> Path:
        return self.metadata_dir

    def get_data_archive_videos_dir(self) -> Path:
        return self.archive_videos_dir

    def get_reports_dir(self) -> Path:
        return self.reports_dir


def _write_filter(path: Path, entries: list[str]) -> None:
    path.write_text(json.dumps({"data_downloads_audio_dir": entries}), encoding="utf-8")


def _write_download_archive(config: _Config, channel: str, video_ids: list[str]) -> None:
    channel_dir = config.videos_dir / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"youtube {video_id}\n" for video_id in video_ids)
    (channel_dir / "downloaded.txt").write_text(lines, encoding="utf-8")


def test_stage_one_lists_all_filtered_files_and_aborts_before_stage_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    first_id = "aaaAAA111_-"
    second_id = "bbbBBB222_-"
    stranded_id = "cccCCC333_-"
    _write_filter(filter_path, [f"Channel/{first_id}", f"Channel/{second_id}"])

    metadata_channel = config.metadata_dir / "Channel" / "video"
    metadata_channel.mkdir(parents=True)
    (metadata_channel / f"First [{first_id}].info.json").write_text("{}", encoding="utf-8")
    (metadata_channel / f"First [{first_id}].webp").write_text("thumbnail", encoding="utf-8")
    archive_channel = config.archive_videos_dir / "Channel"
    archive_channel.mkdir(parents=True)
    (archive_channel / f"Second [{second_id}].mp4").write_text("video", encoding="utf-8")
    _write_download_archive(config, "Channel", [stranded_id])

    result = run_pipeline_integrity(cast(Config, config), filter_path)
    output = capsys.readouterr().out

    assert result == 1
    assert first_id not in output
    assert second_id not in output
    assert "Stage 1 failed: 2 videos (3 artifact files)." in output
    assert str((config.reports_dir / "files-whose-video-id-is-listed-in-filefilter.txt").resolve()) in output
    assert "Stage 2/2" not in output
    assert stranded_id not in output
    assert (config.reports_dir / "files-whose-video-id-is-listed-in-filefilter.txt").read_text(encoding="utf-8").splitlines() == [
        str((metadata_channel / f"First [{first_id}].info.json").resolve()),
        str((metadata_channel / f"First [{first_id}].webp").resolve()),
        str((archive_channel / f"Second [{second_id}].mp4").resolve()),
    ]
    assert not (config.reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt").exists()


def test_stage_two_accepts_any_video_audio_or_transcript_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    _write_filter(filter_path, [])

    active_video_id = "vidVID111_-"
    archived_video_id = "arcARC222_-"
    audio_id = "audAUD333_-"
    transcript_id = "txtTXT444_-"
    stranded_id = "badBAD555_-"
    channel = "Channel"
    _write_download_archive(
        config,
        channel,
        [active_video_id, archived_video_id, audio_id, transcript_id, stranded_id],
    )

    (config.videos_dir / channel / f"Active [{active_video_id}].mp4").write_text("video", encoding="utf-8")
    archived_channel = config.archive_videos_dir / channel
    archived_channel.mkdir(parents=True)
    (archived_channel / f"Archived [{archived_video_id}].webm").write_text("video", encoding="utf-8")
    audio_channel = config.audio_dir / channel
    audio_channel.mkdir(parents=True)
    (audio_channel / f"Audio [{audio_id}].wav").write_text("audio", encoding="utf-8")
    transcript_channel = config.transcripts_dir / channel
    transcript_channel.mkdir(parents=True)
    (transcript_channel / f"Transcript [{transcript_id}].srt").write_text("transcript", encoding="utf-8")

    result = run_pipeline_integrity(cast(Config, config), filter_path)
    output = capsys.readouterr().out

    assert result == 1
    assert "Stage 1 passed" in output
    assert "Stage 2 failed: 1 downloaded.txt entry has no video, audio, or transcript file." in output
    assert stranded_id not in output
    assert (config.reports_dir / "files-whose-video-id-is-listed-in-filefilter.txt").read_text(encoding="utf-8") == ""
    assert (config.reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt").read_text(
        encoding="utf-8"
    ) == f"{(config.videos_dir / channel / 'downloaded.txt').resolve()}\t{stranded_id}\n"


def test_stage_two_matches_artifacts_within_the_same_channel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    _write_filter(filter_path, [])
    video_id = "sameID123_-"
    _write_download_archive(config, "Channel_A", [video_id])

    other_channel = config.transcripts_dir / "Channel_B"
    other_channel.mkdir(parents=True)
    (other_channel / f"Transcript [{video_id}].txt").write_text("transcript", encoding="utf-8")

    result = run_pipeline_integrity(cast(Config, config), filter_path)
    output = capsys.readouterr().out

    assert result == 1
    assert video_id not in output
    assert (config.reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt").read_text(
        encoding="utf-8"
    ) == f"{(config.videos_dir / 'Channel_A' / 'downloaded.txt').resolve()}\t{video_id}\n"


def test_stage_two_ignores_downloaded_video_ids_that_are_filtered(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    filtered_id = "skipID123_-"
    _write_filter(filter_path, [f"Channel/{filtered_id}"])
    _write_download_archive(config, "Channel", [filtered_id])

    result = run_pipeline_integrity(cast(Config, config), filter_path)
    output = capsys.readouterr().out

    assert result == 0
    assert "Stage 1 passed" in output
    assert "Stage 2 passed" in output
    assert filtered_id not in output
    assert (config.reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt").read_text(encoding="utf-8") == ""


def test_pipeline_integrity_passes_when_every_archive_entry_has_an_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    _write_filter(filter_path, [])
    video_id = "goodID123_-"
    channel = "Channel"
    _write_download_archive(config, channel, [video_id, video_id])
    transcript_channel = config.transcripts_dir / channel
    transcript_channel.mkdir(parents=True)
    (transcript_channel / f"Transcript [{video_id}].txt").write_text("transcript", encoding="utf-8")

    result = run_pipeline_integrity(cast(Config, config), filter_path)
    output = capsys.readouterr().out

    assert result == 0
    assert "Stage 1 passed" in output
    assert "Stage 2 passed" in output
    assert (config.reports_dir / "files-whose-video-id-is-listed-in-filefilter.txt").read_text(encoding="utf-8") == ""
    assert (config.reports_dir / "downloaded-video-ids-without-video-audio-or-transcript.txt").read_text(encoding="utf-8") == ""


def test_remove_filtered_artifacts_deletes_recomputed_files_but_preserves_download_archive(tmp_path: Path) -> None:
    config = _Config(tmp_path / "data")
    filter_path = tmp_path / "filefilter.json"
    filtered_id = "cleanID12_-"
    _write_filter(filter_path, [f"Channel/{filtered_id}"])

    metadata_channel = config.metadata_dir / "Channel" / "video"
    metadata_channel.mkdir(parents=True)
    metadata_file = metadata_channel / f"Metadata [{filtered_id}].info.json"
    metadata_file.write_text("{}", encoding="utf-8")
    apple_double_file = metadata_channel / f"._Thumbnail [{filtered_id}].webp"
    apple_double_file.write_text("sidecar", encoding="utf-8")

    cleaned_channel = config.cleaned_dir / "Channel"
    cleaned_channel.mkdir(parents=True)
    cleaned_file = cleaned_channel / f"Transcript [{filtered_id}].txt"
    cleaned_file.write_text("transcript", encoding="utf-8")

    _write_download_archive(config, "Channel", [filtered_id])
    download_archive = config.videos_dir / "Channel" / "downloaded.txt"
    unfiltered_file = metadata_channel / "Other [otherID1_-].info.json"
    unfiltered_file.write_text("{}", encoding="utf-8")

    filtered_ids = load_filtered_video_ids(filter_path)
    artifacts = find_filtered_file_offenders(cast(Config, config), filtered_ids, None)
    deleted, failures = remove_filtered_artifacts(cast(Config, config), filtered_ids, artifacts)

    assert deleted == 3
    assert failures == []
    assert not metadata_file.exists()
    assert not apple_double_file.exists()
    assert not cleaned_file.exists()
    assert download_archive.read_text(encoding="utf-8") == f"youtube {filtered_id}\n"
    assert unfiltered_file.is_file()


def test_remove_filtered_artifacts_refuses_symbolic_links(tmp_path: Path) -> None:
    config = _Config(tmp_path / "data")
    filtered_id = "linkID123_-"
    external_file = tmp_path / "external.txt"
    external_file.write_text("keep", encoding="utf-8")
    metadata_channel = config.metadata_dir / "Channel" / "video"
    metadata_channel.mkdir(parents=True)
    linked_artifact = metadata_channel / f"Linked [{filtered_id}].txt"
    linked_artifact.symlink_to(external_file)

    deleted, failures = remove_filtered_artifacts(cast(Config, config), {filtered_id}, [linked_artifact])

    assert deleted == 0
    assert failures == [(linked_artifact, "refusing to delete a symbolic link")]
    assert linked_artifact.is_symlink()
    assert external_file.read_text(encoding="utf-8") == "keep"


def test_find_filtered_file_offenders_announces_each_directory_before_scanning(tmp_path: Path) -> None:
    config = _Config(tmp_path / "data")
    announced: list[Path] = []

    offenders = find_filtered_file_offenders(cast(Config, config), set(), announced.append)

    assert offenders == []
    assert announced == [
        config.videos_dir,
        config.audio_dir,
        config.transcripts_dir,
        config.hallucinations_dir,
        config.cleaned_dir,
        config.summaries_dir,
        config.metadata_dir,
        config.archive_videos_dir,
    ]
