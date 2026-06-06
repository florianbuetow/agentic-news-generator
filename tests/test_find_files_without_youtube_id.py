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
    def __init__(self, data_dir: Path, category_dirs: dict[str, Path]) -> None:
        self._data_dir = data_dir
        paths = {"data_dir": str(data_dir)}
        for key, path in category_dirs.items():
            paths[key] = str(path)
        self._paths_config = _PathsConfig(paths)

    def get_data_dir(self) -> Path:
        return self._data_dir

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


def test_iter_orphan_files_returns_nothing_for_missing_directory(tmp_path: Path) -> None:
    module = _load_script_module()

    assert list(module.iter_orphan_files(tmp_path / "does-not-exist")) == []


def test_print_scan_groups_by_category_and_counts(tmp_path: Path, capsys: Any) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "data"
    videos_dir = data_dir / "downloads" / "videos" / "Example Channel"
    audio_dir = data_dir / "downloads" / "audio" / "Example Channel"
    videos_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    orphan_video = videos_dir / "Orphan.mp4"
    orphan_video.write_text("content")
    (audio_dir / "Paired [abcDEF123_-].wav").write_text("content")
    (audio_dir / "Paired.wav").write_text("content")

    config = _Config(
        data_dir,
        {
            "data_downloads_videos_dir": data_dir / "downloads" / "videos",
            "data_downloads_audio_dir": data_dir / "downloads" / "audio",
        },
    )

    total = module.print_scan(config)
    captured = capsys.readouterr()

    assert total == 1
    assert "=== Scanning data_downloads_videos_dir:" in captured.out
    assert "=== Scanning data_downloads_audio_dir:" in captured.out
    assert "downloads/videos/Example Channel/Orphan.mp4" in captured.out
    assert "No files without a YouTube ID found in this folder." in captured.out
