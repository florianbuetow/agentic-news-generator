from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast


def _load_script_module() -> Any:
    script_path = Path(__file__).parent.parent / "scripts" / "cleanup-plain-filename-duplicates.py"
    spec = importlib.util.spec_from_file_location("cleanup_plain_filename_duplicates", script_path)
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
    def __init__(self, data_dir: Path, transcript_dir: Path) -> None:
        self._data_dir = data_dir
        self._paths_config = _PathsConfig(
            {
                "data_dir": str(data_dir),
                "data_downloads_transcripts_dir": str(transcript_dir),
            }
        )

    def get_data_dir(self) -> Path:
        return self._data_dir

    def get_paths_config(self) -> _PathsConfig:
        return self._paths_config


def test_extract_youtube_id_requires_brackets_immediately_before_extension() -> None:
    module = _load_script_module()

    assert module.extract_youtube_id("Title [abcDEF123_-].srt") == "abcDEF123_-"
    assert module.extract_youtube_id("blablabla [2189ddasd].txt") == "2189ddasd"
    assert module.extract_youtube_id("Title [abcDEF123_-].info.json") == "abcDEF123_-"
    assert module.extract_youtube_id("Title [abcDEF123_-] extra.srt") is None
    assert module.extract_youtube_id("Title abcDEF123_-.srt") is None


def test_filename_without_youtube_id_removes_preceding_space() -> None:
    module = _load_script_module()

    assert module.filename_without_youtube_id("blablabla [2189ddasd].txt") == "blablabla.txt"
    assert module.filename_without_youtube_id("blablabla[2189ddasd].txt") == "blablabla.txt"
    assert module.filename_without_youtube_id("blablabla [2189ddasd] extra.txt") is None


def test_iter_youtube_id_file_pairs_scans_channel_tree(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "transcripts"
    channel_dir = base_dir / "Example Channel"
    nested_dir = channel_dir / "video"
    nested_dir.mkdir(parents=True)

    matching_file = channel_dir / "First [abcDEF123_-].srt"
    sibling_file = channel_dir / "First.srt"
    nested_matching_file = nested_dir / "Second [XYZ987abc_-].info.json"
    nested_sibling_file = nested_dir / "Second.info.json"
    ignored_file = channel_dir / "No id.srt"
    ignored_metadata_file = channel_dir / "._First [abcDEF123_-].srt"
    matching_file.write_text("content")
    sibling_file.write_text("content")
    nested_matching_file.write_text("{}")
    nested_sibling_file.write_text("{}")
    ignored_file.write_text("content")
    ignored_metadata_file.write_text("content")

    matches = list(module.iter_youtube_id_file_pairs("data_downloads_transcripts_dir", tmp_path, base_dir, channel_dir))

    assert [(match.channel_name, match.video_id, match.file_with_id, match.file_without_id) for match in matches] == [
        ("Example Channel", "abcDEF123_-", matching_file, sibling_file),
        ("Example Channel", "XYZ987abc_-", nested_matching_file, nested_sibling_file),
    ]


def test_iter_youtube_id_file_pairs_ignores_files_without_non_id_sibling(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "transcripts"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)
    (channel_dir / "Only [abcDEF123_-].srt").write_text("content")

    matches = list(module.iter_youtube_id_file_pairs("data_downloads_transcripts_dir", tmp_path, base_dir, channel_dir))

    assert matches == []


def test_format_file_lines_are_descriptive_and_single_file_per_line(tmp_path: Path) -> None:
    module = _load_script_module()
    base_dir = tmp_path / "transcripts"
    channel_dir = base_dir / "Example Channel"
    channel_dir.mkdir(parents=True)
    file_with_id = channel_dir / "blablabla [2189ddasd].txt"
    file_without_id = channel_dir / "blablabla.txt"
    file_with_id.write_text("content")
    file_without_id.write_text("content")
    match = next(module.iter_youtube_id_file_pairs("data_downloads_transcripts_dir", tmp_path, base_dir, channel_dir))

    lines = module.format_file_lines(match)

    assert lines == (
        "[2189ddasd] id_filename     transcripts/Example Channel/blablabla [2189ddasd].txt",
        "[2189ddasd] plain_filename  transcripts/Example Channel/blablabla.txt",
    )
    assert all("\n" not in line for line in lines)
    assert all("channel=" not in line for line in lines)
    assert all("folder=" not in line for line in lines)


def test_copy_plain_files_continues_after_failed_copy_verification(tmp_path: Path, capsys: Any) -> None:
    module = _load_script_module()
    data_dir = tmp_path / "data"
    transcript_dir = data_dir / "transcripts"
    channel_dir = transcript_dir / "Example Channel"
    backup_data_dir = tmp_path / "backup" / "data"
    channel_dir.mkdir(parents=True)
    first_plain_file = channel_dir / "First.srt"
    second_plain_file = channel_dir / "Second.srt"
    (channel_dir / "First [first_id].srt").write_text("with id")
    first_plain_file.write_text("plain")
    (channel_dir / "Second [second_id].srt").write_text("with id")
    second_plain_file.write_text("plain")
    config = _Config(data_dir, transcript_dir)
    original_copy2 = module.shutil.copy2

    def copy2_with_first_failure(src: Path, dst: Path) -> None:
        if Path(src).name == first_plain_file.name:
            return
        original_copy2(src, dst)

    module.shutil.copy2 = copy2_with_first_failure
    try:
        exit_code = module.copy_plain_files(config, backup_data_dir)
    finally:
        module.shutil.copy2 = original_copy2

    captured = capsys.readouterr()

    assert exit_code == 1
    assert first_plain_file.is_file()
    assert not second_plain_file.exists()
    assert not (backup_data_dir / "transcripts" / "Example Channel" / "First.srt").exists()
    assert (backup_data_dir / "transcripts" / "Example Channel" / "Second.srt").is_file()
    assert "[first_id] copy verification failed" in captured.err
    assert "Failed to move 1 plain filename file(s)." in captured.err
    assert "[second_id] moved_plain_filename" in captured.out
