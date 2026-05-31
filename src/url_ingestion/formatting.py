"""Formatting agent and Markdown validation for URL raw content."""

import re
import time
from dataclasses import dataclass
from typing import Protocol

import litellm
import tiktoken
from litellm.exceptions import BadRequestError

from src.config import LLMConfig


class OversizedDocumentError(ValueError):
    """Raised when a raw document exceeds the configured context threshold."""


class MarkdownValidationError(ValueError):
    """Raised when cleaned Markdown fails validation."""


class LlmClient(Protocol):
    """LLM client protocol for formatting tests and production calls."""

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Return a formatted response for a prompt."""
        ...


class LiteLlmClient:
    """LiteLLM-backed formatting client."""

    def complete(self, prompt: str, llm: LLMConfig) -> str:
        """Call the configured LLM and return response text."""
        try:
            response = litellm.completion(
                model=llm.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=llm.api_base,
                api_key=llm.api_key,
                max_tokens=llm.max_tokens,
                temperature=llm.temperature,
            )
        except BadRequestError as exc:
            if "No models loaded" in str(exc):
                raise RuntimeError(f"No models loaded in LM Studio. Expected model: {llm.model}") from exc
            raise

        response_text = response.choices[0].message.content
        if response_text is None or not response_text.strip():
            raise ValueError("LLM returned empty response")
        return strip_think_tags(response_text.strip())


@dataclass(frozen=True)
class FormattingAgent:
    """Format extracted URL content into clean Markdown."""

    llm: LLMConfig
    prompt_template: str
    encoder: tiktoken.Encoding
    skip_threshold_pct: int
    llm_client: LlmClient

    def render_prompt(self, source_text: str) -> str:
        """Render the formatting prompt for extracted source text."""
        return self.prompt_template.replace("{source_text}", source_text)

    def format_markdown(self, source_text: str) -> str:
        """Format extracted source text as Markdown with retries and validation."""
        prompt = self.render_prompt(source_text)
        prompt_tokens = len(self.encoder.encode(prompt, disallowed_special=()))
        token_limit = int(self.llm.context_window * self.skip_threshold_pct / 100)
        if prompt_tokens > token_limit:
            usage_pct = (prompt_tokens / self.llm.context_window * 100.0) if self.llm.context_window > 0 else 0.0
            raise OversizedDocumentError(
                f"document prompt contains {prompt_tokens:,} tokens ({usage_pct:.1f}% of context window "
                f"{self.llm.context_window:,}; threshold: {self.skip_threshold_pct}%/{token_limit:,} tokens)"
            )

        attempt = 1
        while attempt <= self.llm.max_retries:
            try:
                cleaned_markdown = self.llm_client.complete(prompt, self.llm)
                validate_cleaned_markdown(source_text, cleaned_markdown)
                return cleaned_markdown
            except Exception:
                if attempt == self.llm.max_retries:
                    raise
                time.sleep(self.llm.retry_delay)
            attempt += 1

        raise RuntimeError("unreachable")


def strip_think_tags(text: str) -> str:
    """Strip model reasoning tags from a response."""
    think_match = re.search(r"</think>\s*(.*)$", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


def validate_cleaned_markdown(source_text: str, cleaned_markdown: str) -> None:
    """Validate cleaned Markdown against Milestone 4 output rules."""
    if not cleaned_markdown.strip():
        raise MarkdownValidationError("cleaned Markdown is empty")
    inline_html_re = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>")
    if inline_html_re.search(cleaned_markdown):
        raise MarkdownValidationError("cleaned Markdown contains inline HTML")

    source_length = whitespace_normalized_length(source_text)
    cleaned_length = whitespace_normalized_length(cleaned_markdown)
    if source_length >= 500 and cleaned_length < int(source_length * 0.8):
        raise MarkdownValidationError(
            f"cleaned Markdown is too short: {cleaned_length} normalized chars; expected at least {int(source_length * 0.8)}"
        )


def whitespace_normalized_length(text: str) -> int:
    """Return whitespace-normalized character length."""
    return len(" ".join(text.split()))
