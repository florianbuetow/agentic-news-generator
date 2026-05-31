"""Integration tests for the URL inbox queue reader."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import Config
from src.url_ingestion.classifier import UrlClassifier
from src.url_ingestion.normalizer import UrlNormalizer
from src.url_ingestion.queue_reader import UrlInboxQueueReader


def get_valid_paths_config(data_dir: Path) -> dict[str, str]:
    """Return a valid paths configuration dictionary for a temp data root."""
    return {
        "data_dir": str(data_dir),
        "data_models_dir": str(data_dir / "models"),
        "data_downloads_dir": str(data_dir / "downloads"),
        "data_downloads_videos_dir": str(data_dir / "downloads" / "videos"),
        "data_downloads_transcripts_dir": str(data_dir / "downloads" / "transcripts"),
        "data_downloads_transcripts_hallucinations_dir": str(data_dir / "downloads" / "transcripts-hallucinations"),
        "data_downloads_transcripts_cleaned_dir": str(data_dir / "downloads" / "transcripts_cleaned"),
        "data_downloads_transcripts_summaries_dir": str(data_dir / "downloads" / "transcripts_summaries"),
        "data_downloads_audio_dir": str(data_dir / "downloads" / "audio"),
        "data_downloads_metadata_dir": str(data_dir / "downloads" / "metadata"),
        "data_output_dir": str(data_dir / "output"),
        "data_input_dir": str(data_dir / "input"),
        "data_temp_dir": str(data_dir / "temp"),
        "data_archive_dir": str(data_dir / "archive"),
        "data_archive_videos_dir": str(data_dir / "archive" / "videos"),
        "data_logs_dir": str(data_dir / "logs"),
        "reports_dir": "reports",
    }


def get_test_url_processing_config() -> dict[str, str]:
    """Return URL processing config pointed at the configured test URL folder."""
    return {
        "base_dir": ".test/urls",
        "inbox_dir": "inbox_urls",
        "raw_dir": "data_raw",
        "cleaned_dir": "data_cleaned",
        "test_base_dir": ".test/urls",
    }


def write_config(config_path: Path, data_dir: Path) -> Config:
    """Write a minimal config file and return its Config object."""
    config_data: dict[str, Any] = {
        "paths": get_valid_paths_config(data_dir),
        "channels": [],
        "url_processing": get_test_url_processing_config(),
    }
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return Config(config_path)


def make_reader(config: Config) -> UrlInboxQueueReader:
    """Create a URL inbox queue reader with explicit processing components."""
    return UrlInboxQueueReader(config, UrlNormalizer(), UrlClassifier())


def test_queue_reader_merges_deduplicates_sorts_and_counts_urls(tmp_path: Path) -> None:
    """Read configured test inbox files, exclude archive folders, and return deterministic counts."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "done").mkdir()
    (inbox_dir / "unprocessed").mkdir()

    (inbox_dir / "list_b.txt").write_text(
        "\n".join(
            [
                "https://example.com/b.html",
                "https://example.com/a",
                "",
                "   ",
                "HTTPS://EXAMPLE.com/a",
            ]
        ),
        encoding="utf-8",
    )
    (inbox_dir / "list_a.txt").write_text(
        "\n".join(
            [
                "example.com/c.pdf",
                " https://example.com/b.html ",
                "https://example.com/search?q=ignored",
            ]
        ),
        encoding="utf-8",
    )
    (inbox_dir / "done" / "ignored.txt").write_text("https://example.com/done\n", encoding="utf-8")
    (inbox_dir / "unprocessed" / "ignored.txt").write_text("https://example.com/unprocessed\n", encoding="utf-8")

    summary = make_reader(config).read_queue()

    assert summary.total_url_lines_read == 6
    assert summary.unique_normalized_url_count == 2
    assert summary.duplicate_url_entry_count == 2
    assert summary.unique_urls == (
        "https://example.com/a",
        "https://example.com/b.html",
    )
    assert summary.type_counts == {
        "pdf": 0,
        "html": 1,
        "markdown": 0,
        "text": 0,
        "unknown": 1,
    }
    assert [queued_url.sanitized_url_stem for queued_url in summary.queued_urls] == [
        "https_example_com_a",
        "https_example_com_b_html",
    ]
    assert [(unprocessable.original_url, unprocessable.reason) for unprocessable in summary.unprocessable_urls] == [
        ("example.com/c.pdf", "line does not contain an http URL"),
        ("https://example.com/search?q=ignored", "URL contains a query string"),
    ]
    assert not config.get_url_raw_dir().exists()
    assert not config.get_url_cleaned_dir().exists()


def test_queue_reader_extracts_url_from_category_prefixed_lines(tmp_path: Path) -> None:
    """Strip a 'Category->Subcategory:' prefix and ingest the embedded URL."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "raindrop.txt").write_text(
        "\n".join(
            [
                "AI Engineering->LLM Evals:https://example.com/a.html",
                "Unsorted:http://example.com/b.pdf",
                "https://example.com/c",
            ]
        ),
        encoding="utf-8",
    )

    summary = make_reader(config).read_queue()

    assert summary.unprocessable_urls == ()
    assert summary.unique_urls == (
        "http://example.com/b.pdf",
        "https://example.com/a.html",
        "https://example.com/c",
    )


def test_queue_reader_rejects_lines_without_an_http_url(tmp_path: Path) -> None:
    """Reject inbox lines that contain no http URL and record them as errors."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "links.txt").write_text(
        "\n".join(
            [
                "AI Engineering->LLM Evals:https://example.com/a.html",
                "example.com/no-scheme",
                "Some Category:not-a-url",
            ]
        ),
        encoding="utf-8",
    )

    summary = make_reader(config).read_queue()

    assert summary.unique_urls == ("https://example.com/a.html",)
    assert [(unprocessable.original_url, unprocessable.reason) for unprocessable in summary.unprocessable_urls] == [
        ("example.com/no-scheme", "line does not contain an http URL"),
        ("Some Category:not-a-url", "line does not contain an http URL"),
    ]


def test_queue_reader_requires_configured_url_base_folder(tmp_path: Path) -> None:
    """Fail clearly when the configured URL base folder is missing."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        make_reader(config).read_queue()

    assert "Configured URL base folder does not exist" in str(exc_info.value)


def test_queue_reader_requires_configured_inbox_folder(tmp_path: Path) -> None:
    """Fail clearly when the configured URL inbox folder is missing."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)
    config.get_url_base_dir().mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc_info:
        make_reader(config).read_queue()

    assert "Configured URL inbox folder does not exist" in str(exc_info.value)


def test_queue_reader_records_sanitized_stem_collisions(tmp_path: Path) -> None:
    """Treat different normalized URLs with the same sanitized stem as unprocessable."""
    data_dir = tmp_path / "data"
    config = write_config(tmp_path / "config.yaml", data_dir)
    inbox_dir = config.get_url_inbox_dir()
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "links.txt").write_text(
        "\n".join(
            [
                "https://example.com/a.b",
                "https://example.com/a_b",
            ]
        ),
        encoding="utf-8",
    )

    summary = make_reader(config).read_queue()

    assert summary.unique_normalized_url_count == 2
    assert summary.queued_urls == ()
    assert [unprocessable.reason for unprocessable in summary.unprocessable_urls] == [
        "sanitized URL stem collision: https_example_com_a_b",
        "sanitized URL stem collision: https_example_com_a_b",
    ]
