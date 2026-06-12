"""Unit tests for YouTube ID extraction in the analytics package."""

import pytest

from src.analytics.ids import extract_youtube_id


class TestExtractYoutubeId:
    """Behavior of extract_youtube_id against the Title [id].ext convention."""

    def test_extracts_id_before_txt_extension(self) -> None:
        """The bracketed token right before the extension is the video ID."""
        assert extract_youtube_id("Building RAG Systems [abc123XYZ09].txt") == "abc123XYZ09"

    def test_extracts_id_before_srt_extension(self) -> None:
        """SRT files follow the same convention."""
        assert extract_youtube_id("Building RAG Systems [abc123XYZ09].srt") == "abc123XYZ09"

    def test_extracts_id_before_double_extension(self) -> None:
        """yt-dlp metadata files use a double extension (.info.json)."""
        assert extract_youtube_id("Building RAG Systems [abc123XYZ09].info.json") == "abc123XYZ09"

    def test_id_with_underscore_and_hyphen(self) -> None:
        """IDs may contain underscores and hyphens."""
        assert extract_youtube_id("Video [a_b-C_d-E_f].txt") == "a_b-C_d-E_f"

    def test_no_bracket_returns_none(self) -> None:
        """Files without a bracketed token have no ID."""
        assert extract_youtube_id("Plain filename.txt") is None

    def test_mid_name_bracket_not_matched(self) -> None:
        """A bracket not directly before the extension is title text, not an ID."""
        assert extract_youtube_id("We [talk] about things.txt") is None

    def test_last_bracket_before_extension_wins(self) -> None:
        """When the title itself contains brackets, the final token is the ID."""
        assert extract_youtube_id("Review [2024] of tools [abc123XYZ09].txt") == "abc123XYZ09"

    def test_invalid_characters_in_token_rejected(self) -> None:
        """Tokens with characters outside the ID charset are not IDs."""
        assert extract_youtube_id("Video [abc!12345678].txt") is None

    def test_works_on_full_paths(self) -> None:
        """Paths with directories still resolve the trailing ID token."""
        assert extract_youtube_id("Some_Channel/Video Title [abc123XYZ09].txt") == "abc123XYZ09"

    @pytest.mark.parametrize("filename", ["", "[].txt", "Video [].txt"])
    def test_empty_or_blank_tokens_return_none(self, filename: str) -> None:
        """Empty names and empty bracket tokens yield no ID."""
        assert extract_youtube_id(filename) is None
