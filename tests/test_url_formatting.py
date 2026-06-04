"""Unit tests for URL content formatting."""

from types import SimpleNamespace

import pytest
import tiktoken

from src.config import LLMConfig
from src.url_ingestion.formatting import (
    FormattingAgent,
    LiteLlmClient,
    MarkdownValidationError,
    OversizedDocumentError,
    validate_cleaned_markdown,
)


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


class EchoLlmClient:
    """Fake LLM client that returns each prompt as the cleaned content."""

    def __init__(self) -> None:
        """Initialize captured prompts."""
        self.prompts: list[str] = []

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Echo the prompt."""
        self.prompts.append(prompt)
        return prompt


class FailOnceLlmClient:
    """Fake LLM client that fails once before returning valid Markdown."""

    def __init__(self) -> None:
        """Initialize call count."""
        self.call_count = 0

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Fail on the first call and succeed on the second."""
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("temporary local model failure")
        return "# Recovered\n\nSource text"


def make_llm_config(*, context_window: int = 1000, max_tokens: int = 100, max_retries: int = 1) -> LLMConfig:
    """Build a valid LLM config for tests."""
    return LLMConfig(
        model="openai/test-model",
        api_base="http://127.0.0.1:1234/v1",
        api_key="test-key",
        context_window=context_window,
        max_tokens=max_tokens,
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


def test_litellm_client_passes_configured_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward configured request timeouts to litellm."""
    captured_kwargs: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="# Done"))])

    monkeypatch.setattr("src.url_ingestion.formatting.litellm.completion", fake_completion)
    llm = make_llm_config().model_copy(update={"request_timeout_seconds": 12.5})

    result = LiteLlmClient().complete("Format this", llm)

    assert result == "# Done"
    assert captured_kwargs["timeout"] == 12.5


def test_formatting_agent_skips_oversized_documents() -> None:
    """Skip large extracted documents that exceed the configured context threshold."""
    fake_client = EchoLlmClient()
    progress_messages: list[str] = []
    agent = FormattingAgent(
        llm=make_llm_config(context_window=50),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=fake_client,
        progress_callback=progress_messages.append,
    )
    source_text = " ".join(f"word{i}" for i in range(200))

    with pytest.raises(OversizedDocumentError) as exc_info:
        agent.format_markdown_with_stats(source_text)

    assert "document prompt contains" in str(exc_info.value)
    assert fake_client.prompts == []
    assert progress_messages == []


def test_formatting_agent_reports_retry_failures() -> None:
    """Emit failed-attempt progress before retrying LLM formatting."""
    fake_client = FailOnceLlmClient()
    progress_messages: list[str] = []
    agent = FormattingAgent(
        llm=make_llm_config(max_retries=2),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=fake_client,
        progress_callback=progress_messages.append,
    )

    result = agent.format_markdown_with_stats("Source text")

    assert result.cleaned_markdown == "# Recovered\n\nSource text"
    assert result.attempts == 2
    assert fake_client.call_count == 2
    assert any(
        "formatting_failed: attempt=1/2 error=RuntimeError: temporary local model failure" in message for message in progress_messages
    )
    assert any(message.startswith("formatting_done: attempt=2/2 elapsed_seconds=") for message in progress_messages)


def test_formatting_agent_estimates_prompt_tokens_without_llm_call() -> None:
    """Report single-call prompt tokens without sending prompts to the LLM."""
    fake_client = FakeLlmClient("unused")
    agent = FormattingAgent(
        llm=make_llm_config(context_window=50),
        prompt_template="{source_text}",
        encoder=tiktoken.get_encoding("o200k_base"),
        skip_threshold_pct=80,
        llm_client=fake_client,
    )
    source_text = " ".join(f"word{i}" for i in range(200))

    estimate = agent.estimate_work(source_text)

    assert estimate.prompt_tokens > 0
    assert fake_client.prompts == []


def test_cleaned_markdown_validation_rejects_inline_html() -> None:
    """Reject cleaned Markdown containing inline HTML tags."""
    with pytest.raises(MarkdownValidationError) as exc_info:
        validate_cleaned_markdown("Source text", "Cleaned <span>bad</span>")

    assert "inline HTML" in str(exc_info.value)


def test_cleaned_markdown_validation_allows_python_angle_bracket_reprs() -> None:
    """Allow Python/IPython repr text that looks like an angle-bracket tag."""
    validate_cleaned_markdown("Source text", "Inspect output: <no docstring>\n\nCell: <ipython-input-23-b5adf20be596>")
    validate_cleaned_markdown("Source text", "Python type repr: <class 'numpy.ndarray'>")


def test_cleaned_markdown_validation_ignores_html_inside_code() -> None:
    """Allow HTML syntax when it appears inside Markdown code."""
    validate_cleaned_markdown("Source text", "Inline code: `<span>example</span>`")
    validate_cleaned_markdown("Source text", "```html\n<div>example</div>\n```")


def test_cleaned_markdown_validation_allows_shorter_large_output() -> None:
    """Allow shorter cleaned Markdown because extraction can include boilerplate."""
    source_text = "a" * 500

    validate_cleaned_markdown(source_text, "short cleaned content")


def test_cleaned_markdown_validation_skips_length_check_for_small_input() -> None:
    """Require only non-empty and no inline HTML for small extracted inputs."""
    validate_cleaned_markdown("short source", "small")
