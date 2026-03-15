"""Tests for LLMTopicLabeler."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import LLMConfig, TopicDetectionLLMLabelConfig
from src.topic_detection.hierarchy.labeling import LLMTopicLabeler
from src.topic_detection.hierarchy.schemas import LLMTopicLabelData


def make_config(*, enabled: bool = True) -> TopicDetectionLLMLabelConfig:
    """Create a test config."""
    return TopicDetectionLLMLabelConfig(
        enabled=enabled,
        llm=LLMConfig(
            model="openai/test-model",
            api_base="http://127.0.0.1:1234/v1",
            api_key="test-key",
            context_window=8192,
            max_tokens=512,
            temperature=0.3,
            context_window_threshold=90,
            max_retries=2,
            retry_delay=0.01,
            timeout_seconds=30,
        ),
    )


def mock_completion_response(content: str) -> MagicMock:
    """Build a mock litellm completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


VALID_JSON = '{"summary": "Discusses AI chips.", "about": "New chip architectures for AI.", "topic_labels": ["chips", "hardware", "ai"]}'


class TestLLMTopicLabelerDisabled:
    """Tests for disabled labeler."""

    def test_returns_none_when_disabled(self) -> None:
        labeler = LLMTopicLabeler(config=make_config(enabled=False))
        result = labeler.label(text="Some text about AI")
        assert result is None


class TestLLMTopicLabelerEmptyText:
    """Tests for empty/whitespace text."""

    def test_returns_none_for_empty_string(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        result = labeler.label(text="")
        assert result is None

    def test_returns_none_for_whitespace(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        result = labeler.label(text="   \n  ")
        assert result is None


class TestLLMTopicLabelerParseResponse:
    """Tests for response parsing logic."""

    def test_parses_plain_json(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        result = labeler.parse_response(VALID_JSON)
        assert isinstance(result, LLMTopicLabelData)
        assert result.summary == "Discusses AI chips."
        assert result.about == "New chip architectures for AI."
        assert result.topic_labels == ["chips", "hardware", "ai"]

    def test_parses_json_in_markdown_fence(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        wrapped = f"```json\n{VALID_JSON}\n```"
        result = labeler.parse_response(wrapped)
        assert result.topic_labels == ["chips", "hardware", "ai"]

    def test_parses_json_in_plain_fence(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        wrapped = f"```\n{VALID_JSON}\n```"
        result = labeler.parse_response(wrapped)
        assert result.summary == "Discusses AI chips."

    def test_parses_json_after_think_tags(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        wrapped = f"<think>Some reasoning here about the text.</think>\n{VALID_JSON}"
        result = labeler.parse_response(wrapped)
        assert result.topic_labels == ["chips", "hardware", "ai"]

    def test_parses_json_after_think_tags_with_markdown(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        wrapped = f"<think>Reasoning.</think>\n```json\n{VALID_JSON}\n```"
        result = labeler.parse_response(wrapped)
        assert result.about == "New chip architectures for AI."

    def test_raises_on_invalid_json(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
            labeler.parse_response("not json at all")

    def test_raises_on_empty_after_cleaning(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        with pytest.raises(ValueError, match="No JSON content after cleaning"):
            labeler.parse_response("<think>only thinking</think>")

    def test_raises_on_schema_mismatch(self) -> None:
        labeler = LLMTopicLabeler(config=make_config())
        with pytest.raises(ValueError, match="does not match expected schema"):
            labeler.parse_response('{"wrong_field": "value"}')


class TestLLMTopicLabelerCallLLM:
    """Tests for the LLM call integration."""

    @patch("src.topic_detection.hierarchy.labeling.litellm.completion")
    def test_successful_label(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = mock_completion_response(VALID_JSON)
        labeler = LLMTopicLabeler(config=make_config())
        result = labeler.label(text="Discussion about new AI chip designs")

        assert result is not None
        assert result.summary == "Discusses AI chips."
        assert result.topic_labels == ["chips", "hardware", "ai"]
        mock_completion.assert_called_once()

    @patch("src.topic_detection.hierarchy.labeling.litellm.completion")
    def test_retries_on_parse_failure(self, mock_completion: MagicMock) -> None:
        mock_completion.side_effect = [
            mock_completion_response("garbage response"),
            mock_completion_response(VALID_JSON),
        ]
        labeler = LLMTopicLabeler(config=make_config())
        result = labeler.label(text="Some text")

        assert result is not None
        assert result.summary == "Discusses AI chips."
        assert mock_completion.call_count == 2

    @patch("src.topic_detection.hierarchy.labeling.litellm.completion")
    def test_raises_after_all_retries_exhausted(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = mock_completion_response("not json")
        labeler = LLMTopicLabeler(config=make_config())

        with pytest.raises(ValueError, match="failed after 2 attempts"):
            labeler.label(text="Some text")

        assert mock_completion.call_count == 2

    @patch("src.topic_detection.hierarchy.labeling.litellm.completion")
    def test_raises_on_empty_response(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = mock_completion_response("")
        labeler = LLMTopicLabeler(config=make_config())

        with pytest.raises(ValueError, match="failed after 2 attempts"):
            labeler.label(text="Some text")

    @patch("src.topic_detection.hierarchy.labeling.litellm.completion")
    def test_passes_correct_model_params(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = mock_completion_response(VALID_JSON)
        labeler = LLMTopicLabeler(config=make_config())
        labeler.label(text="Test text")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/test-model"
        assert call_kwargs.kwargs["api_base"] == "http://127.0.0.1:1234/v1"
        assert call_kwargs.kwargs["api_key"] == "test-key"
        assert call_kwargs.kwargs["max_tokens"] == 512
        assert call_kwargs.kwargs["temperature"] == 0.3
