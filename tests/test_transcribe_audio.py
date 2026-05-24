import json
from pathlib import Path

from scripts.transcribe_audio import find_metadata_file, transcript_outputs_complete
from src.file_processing_filter import add_no_speech_to_filefilter


def _write_filefilter(path: Path, audio_entries: list[str]) -> None:
    path.write_text(json.dumps({"data_downloads_audio_dir": audio_entries}, indent=4) + "\n")


def _read_audio_entries(path: Path) -> list[str]:
    return json.loads(path.read_text())["data_downloads_audio_dir"]


def test_adds_new_entry(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    _write_filefilter(ff, [])
    result = add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [abc123XYZ]")
    assert result is True
    assert _read_audio_entries(ff) == ["OpenAI/abc123XYZ"]


def test_no_duplicate(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    _write_filefilter(ff, ["OpenAI/abc123XYZ"])
    result = add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [abc123XYZ]")
    assert result is False
    assert _read_audio_entries(ff) == ["OpenAI/abc123XYZ"]


def test_entries_sorted(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    _write_filefilter(ff, ["OpenAI/zzz999"])
    add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [aaa111]")
    assert _read_audio_entries(ff) == ["OpenAI/aaa111", "OpenAI/zzz999"]


def test_no_video_id_bracket(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    _write_filefilter(ff, [])
    result = add_no_speech_to_filefilter(ff, "OpenAI", "Video Without ID Suffix")
    assert result is False
    assert _read_audio_entries(ff) == []


def test_missing_filefilter(tmp_path: Path) -> None:
    ff = tmp_path / "does_not_exist.json"
    result = add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [abc123XYZ]")
    assert result is False


def test_missing_audio_key(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    ff.write_text(json.dumps({"data_downloads_transcripts_dir": []}, indent=4) + "\n")
    result = add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [abc123XYZ]")
    assert result is False


def test_preserves_other_keys(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    data: dict[str, list[str]] = {
        "data_downloads_audio_dir": [],
        "data_downloads_transcripts_dir": ["Chan/existing"],
    }
    ff.write_text(json.dumps(data, indent=4) + "\n")
    add_no_speech_to_filefilter(ff, "OpenAI", "Some Video [abc123XYZ]")
    written: dict[str, list[str]] = json.loads(ff.read_text())
    assert written["data_downloads_transcripts_dir"] == ["Chan/existing"]
    assert written["data_downloads_audio_dir"] == ["OpenAI/abc123XYZ"]


def test_video_id_with_hyphens_and_underscores(tmp_path: Path) -> None:
    ff = tmp_path / "filefilter.json"
    _write_filefilter(ff, [])
    result = add_no_speech_to_filefilter(ff, "SomeChannel", "Title [kVmp0uGtShk]")
    assert result is True
    assert _read_audio_entries(ff) == ["SomeChannel/kVmp0uGtShk"]


def test_transcript_outputs_complete_requires_txt_and_srt(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    base_name = "Some Video [abc123XYZ]"

    (transcripts_dir / f"{base_name}.txt").write_text("partial transcript")

    assert transcript_outputs_complete(transcripts_dir, base_name) is False

    (transcripts_dir / f"{base_name}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n")

    assert transcript_outputs_complete(transcripts_dir, base_name) is True


def test_find_metadata_file_matches_format_suffixed_stem_by_video_id(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_file = metadata_dir / "Some Video [abc123XYZ].info.json"
    metadata_file.write_text("{}")

    result = find_metadata_file(metadata_dir, "Some Video [abc123XYZ].f251")

    assert result == metadata_file
