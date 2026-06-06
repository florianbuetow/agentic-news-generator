from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast


def _load_script_module() -> Any:
    script_path = Path(__file__).parent.parent / "scripts" / "find-files-without-youtube-id.py"
    spec = importlib.util.spec_from_file_location("find_files_without_youtube_id", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Any, module)


class _PathsConfig:
    def __init__(self, paths: dict[str, str]) -> None:
        self._paths = paths

    def model_dump(self) -> dict[str, str]:
        return self._paths


class _Config:
    def __init__(self, data_dir: Path, category_dirs: dict[str, Path], reports_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self._reports_dir = reports_dir if reports_dir is not None else data_dir.parent / "reports"
        paths = {"data_dir": str(data_dir)}
        for key, path in category_dirs.items():
            paths[key] = str(path)
        self._paths_config = _PathsConfig(paths)

    def get_data_dir(self) -> Path:
        return self._data_dir

    def get_reports_dir(self) -> Path:
        return self._reports_dir

    def get_paths_config(self) -> _PathsConfig:
        return self._paths_config


def test_extract_youtube_id_requires_brackets_immediately_before_extension() -> None:
    module = _load_script_module()

    assert module.extract_youtube_id("Title [abcDEF123_-].srt") == "abcDEF123_-"
    assert module.extract_youtube_id("Title [abcDEF123_-].info.json") == "abcDEF123_-"
    assert module.extract_youtube_id("Title [abcDEF123_-] extra.srt") is None
    assert module.extract_youtube_id("Title abcDEF123_-.srt") is None


def test_filename_without_youtube_id_removes_preceding_space() -> None:
    module = _load_script_module()

    assert module.filename_without_youtube_id("blablabla [2189ddasd].txt") == "blablabla.txt"
    assert module.filename_without_youtube_id("blablabla[2189ddasd].txt") == "blablabla.txt"
    assert module.filename_without_youtube_id("blablabla.txt") is None


def test_iter_orphan_files_reports_only_plain_files_without_id_sibling(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "videos"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)

    paired_with_id = channel_dir / "Paired [abcDEF123_-].mp4"
    paired_plain = channel_dir / "Paired.mp4"
    orphan_plain = channel_dir / "Orphan.mp4"
    id_only = channel_dir / "IdOnly [XYZ987abc_-].mp4"
    ignored_metadata = channel_dir / "._Orphan.mp4"
    for file_path in (paired_with_id, paired_plain, orphan_plain, id_only, ignored_metadata):
        file_path.write_text("content")

    orphans = list(module.iter_orphan_files(base_dir))

    assert orphans == [orphan_plain]


def test_iter_orphan_files_uses_exact_match_so_extra_suffix_is_orphan(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "transcripts"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)

    id_srt = channel_dir / "Title [abcDEF123_-].srt"
    exact_plain = channel_dir / "Title.srt"
    extra_suffix_plain = channel_dir / "Title.en.srt"
    for file_path in (id_srt, exact_plain, extra_suffix_plain):
        file_path.write_text("content")

    orphans = list(module.iter_orphan_files(base_dir))

    # Title.srt is covered by Title [id].srt; Title.en.srt is NOT (strip yields Title.srt).
    assert orphans == [extra_suffix_plain]


def test_iter_orphan_files_matches_within_same_directory_only(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "videos"
    channel_a = base_dir / "Channel A"
    channel_b = base_dir / "Channel B"
    channel_a.mkdir(parents=True)
    channel_b.mkdir(parents=True)

    (channel_a / "Title [abcDEF123_-].mp4").write_text("content")
    plain_in_other_dir = channel_b / "Title.mp4"
    plain_in_other_dir.write_text("content")

    orphans = list(module.iter_orphan_files(base_dir))

    # The ID-bearing file in Channel A does not cover the plain file in Channel B.
    assert orphans == [plain_in_other_dir]


def test_iter_orphan_files_excludes_download_archives(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "videos"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)

    (channel_dir / "downloaded.txt").write_text("yt-dlp archive")
    orphan = channel_dir / "Orphan.mp4"
    orphan.write_text("content")

    orphans = list(module.iter_orphan_files(base_dir))

    # The per-channel download archive is bookkeeping, not a missing-ID content finding.
    assert orphans == [orphan]


def test_iter_orphan_files_ignores_shell_scripts(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "metadata"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)

    (channel_dir / "rsync-to-nas.sh").write_text("#!/bin/sh\n")
    orphan = channel_dir / "Orphan.json"
    orphan.write_text("{}")

    orphans = list(module.iter_orphan_files(base_dir))

    # Shell scripts are never downloadable content and are ignored.
    assert orphans == [orphan]


def test_iter_orphan_files_ignores_dot_directories(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "videos"
    channel_dir = base_dir / "Example Channel"
    dot_dir = base_dir / ".claude"
    channel_dir.mkdir(parents=True)
    dot_dir.mkdir(parents=True)

    (dot_dir / "settings.local.json").write_text("{}")
    orphan = channel_dir / "Orphan.mp4"
    orphan.write_text("content")

    orphans = list(module.iter_orphan_files(base_dir))

    # Files under dot-directories (e.g. .claude) are ignored.
    assert orphans == [orphan]


def test_iter_orphan_files_returns_nothing_for_missing_directory(tmp_path: Path) -> None:
    module = _load_script_module()

    assert list(module.iter_orphan_files(tmp_path / "does-not-exist")) == []


def test_render_box_table_is_aligned_with_total() -> None:
    module = _load_script_module()

    table = module.render_box_table([("downloads/videos", 1), ("downloads/transcripts", 110)], 111)
    lines = table.splitlines()

    # Every rendered line shares the same width -> columns are aligned.
    assert len({len(line) for line in lines}) == 1
    assert lines[0].startswith("┌") and lines[0].endswith("┐")
    assert lines[-1].startswith("└") and lines[-1].endswith("┘")
    assert "Category" in table and "Orphans" in table
    assert "TOTAL" in table and "111" in table


def test_collect_results_and_render_report(tmp_path: Path) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "data"
    videos_dir = data_dir / "downloads" / "videos" / "Example Channel"
    audio_dir = data_dir / "downloads" / "audio" / "Example Channel"
    videos_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    (videos_dir / "Orphan.mp4").write_text("content")
    (audio_dir / "Paired [abcDEF123_-].wav").write_text("content")
    (audio_dir / "Paired.wav").write_text("content")

    config = _Config(
        data_dir,
        {
            "data_downloads_videos_dir": data_dir / "downloads" / "videos",
            "data_downloads_audio_dir": data_dir / "downloads" / "audio",
        },
    )

    results = module.collect_results(config)
    by_key = {result.config_key: result for result in results}

    assert [str(orphan) for orphan in by_key["data_downloads_videos_dir"].orphans] == ["Example Channel/Orphan.mp4"]
    assert by_key["data_downloads_audio_dir"].orphans == []

    report = module.render_report(results)
    assert report.startswith("```")
    assert "# downloads/videos" in report
    assert "Example Channel/Orphan.mp4" in report
    # The empty audio folder gets no H1 section.
    assert "# downloads/audio" not in report
    # Second processing step: the deduplicated channel/video list at the bottom.
    assert "# Channel/Video list" in report
    assert "Example Channel — Orphan" in report


def test_format_channel_video_line() -> None:
    module = _load_script_module()

    assert module.format_channel_video_line("AI_Engineer", "Some Title") == "AI_Engineer — Some Title"
    assert module.format_channel_video_line("", "Loose Title") == "Loose Title"


def test_deduplicated_channel_video_pairs_collapses_across_folders() -> None:
    module = _load_script_module()

    results = [
        module.FolderResult(
            "data_downloads_transcripts_dir",
            Path("downloads/transcripts"),
            [Path("AI_Engineer/Some Title.srt"), Path("AI_Engineer/Some Title.vtt"), Path("Anthropic/Other.srt")],
        ),
        module.FolderResult(
            "data_downloads_transcripts_summaries_dir",
            Path("downloads/transcripts_summaries"),
            [Path("AI_Engineer/Some Title.md")],
        ),
    ]

    pairs = module.deduplicated_channel_video_pairs(results)

    # "Some Title" appears as .srt/.vtt/.md across two folders -> one pair; sorted by channel then title.
    assert pairs == [("AI_Engineer", "Some Title"), ("Anthropic", "Other")]


def test_write_report_writes_to_reports_dir(tmp_path: Path) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"
    videos_dir = data_dir / "downloads" / "videos" / "Example Channel"
    videos_dir.mkdir(parents=True)
    (videos_dir / "Orphan.mp4").write_text("content")

    config = _Config(
        data_dir,
        {"data_downloads_videos_dir": data_dir / "downloads" / "videos"},
        reports_dir=reports_dir,
    )

    results = module.collect_results(config)
    report_path = module.write_report(config, module.render_report(results))

    assert report_path == reports_dir / "files-without-youtube-id.md"
    assert report_path.is_file()
    assert "# downloads/videos" in report_path.read_text(encoding="utf-8")
