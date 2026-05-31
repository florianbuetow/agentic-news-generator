"""Unit tests for URL content formatting."""

import pytest
import tiktoken

from src.config import LLMConfig
from src.url_ingestion.formatting import FormattingAgent, MarkdownValidationError, OversizedDocumentError, validate_cleaned_markdown


class FakeLlmClient:
    """Fake LLM client for formatting tests."""

    def __init__(self, response: str) -> None:
        """Initialize the fake client."""
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Return the configured fake response."""
        self.prompts.append(prompt)
        return self.response


def make_llm_config(*, context_window: int = 1000, max_retries: int = 1) -> LLMConfig:
    """Build a valid LLM config for tests."""
    return LLMConfig(
        model="openai/test-model",
        api_base="http://127.0.0.1:1234/v1",
        api_key="test-key",
        context_window=context_window,
        max_tokens=100,
        temperature=0.0,
        context_window_threshold=90,
        max_retries=max_retries,
        retry_delay=0.01,
    )


def test_formatting_agent_renders_prompt_and_returns_valid_markdown() -> None:
    """Render prompt with source text and return validated Markdown."""
    fake_client = FakeLlmClient("# Heading\n\nSource text")
    agent = FormattingAgent(
        llm=make_llm_config(),
        prompt_template="Format this:\n{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=fake_client,
    )

    result = agent.format_markdown("Source text")

    assert result == "# Heading\n\nSource text"
    assert fake_client.prompts == ["Format this:\nSource text"]


def test_formatting_agent_skips_oversized_documents_before_llm_call() -> None:
    """Raise before calling the LLM when prompt token count exceeds the threshold."""
    fake_client = FakeLlmClient("unused")
    agent = FormattingAgent(
        llm=make_llm_config(context_window=10),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=10,
        llm_client=fake_client,
    )

    with pytest.raises(OversizedDocumentError):
        agent.format_markdown("word " * 100)

    assert fake_client.prompts == []


def test_cleaned_markdown_validation_rejects_inline_html() -> None:
    """Reject cleaned Markdown containing inline HTML tags."""
    with pytest.raises(MarkdownValidationError) as exc_info:
        validate_cleaned_markdown("Source text", "Cleaned <span>bad</span>")

    assert "inline HTML" in str(exc_info.value)


def test_cleaned_markdown_validation_rejects_too_short_large_output() -> None:
    """Reject cleaned Markdown shorter than 80% of large extracted input."""
    source_text = "a" * 500

    with pytest.raises(MarkdownValidationError) as exc_info:
        validate_cleaned_markdown(source_text, "too short")

    assert "too short" in str(exc_info.value)


def test_cleaned_markdown_validation_skips_length_check_for_small_input() -> None:
    """Require only non-empty and no inline HTML for small extracted inputs."""
    validate_cleaned_markdown("short source", "small")
