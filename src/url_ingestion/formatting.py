"""Formatting agent and Markdown validation for URL raw content."""

import re
import time
from collections.abc import Callable
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


@dataclass(frozen=True)
class FormattingResult:
    """Cleaned Markdown plus useful formatting metrics."""

    cleaned_markdown: str
    prompt_tokens: int
    output_chars: int
    attempts: int
    elapsed_seconds: float


@dataclass(frozen=True)
class FormattingWorkEstimate:
    """Estimated LLM work for extracted source text."""

    prompt_tokens: int


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
                timeout=llm.request_timeout_seconds,
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
    progress_callback: Callable[[str], None] | None = None

    def render_prompt(self, source_text: str) -> str:
        """Render the formatting prompt for extracted source text."""
        return self.prompt_template.replace("{source_text}", source_text)

    def format_markdown(self, source_text: str) -> str:
        """Format extracted source text as Markdown with retries and validation."""
        return self.format_markdown_with_stats(source_text).cleaned_markdown

    def estimate_work(self, source_text: str) -> FormattingWorkEstimate:
        """Estimate formatting work without calling the LLM."""
        return FormattingWorkEstimate(prompt_tokens=self._prompt_token_count(source_text))

    def format_markdown_with_stats(self, source_text: str) -> FormattingResult:
        """Format extracted source text as Markdown and return progress metrics."""
        cleaned_markdown, prompt_tokens, attempts, elapsed_seconds = self._format_single_document(source_text)
        return FormattingResult(
            cleaned_markdown=cleaned_markdown,
            prompt_tokens=prompt_tokens,
            output_chars=len(cleaned_markdown),
            attempts=attempts,
            elapsed_seconds=elapsed_seconds,
        )

    def _format_single_document(self, source_text: str) -> tuple[str, int, int, float]:
        """Format one source document and return Markdown, prompt tokens, and attempts."""
        prompt = self.render_prompt(source_text)
        prompt_tokens = len(self.encoder.encode(prompt, disallowed_special=()))
        self._raise_if_oversized(prompt_tokens)

        attempt = 1
        while attempt <= self.llm.max_retries:
            try:
                self._emit_progress(f"formatting: attempt={attempt}/{self.llm.max_retries} prompt_tokens={prompt_tokens}")
                started_at = time.monotonic()
                cleaned_markdown = self.llm_client.complete(prompt, self.llm)
                validate_cleaned_markdown(source_text, cleaned_markdown)
                elapsed_seconds = time.monotonic() - started_at
                self._emit_progress(f"formatting_done: attempt={attempt}/{self.llm.max_retries} elapsed_seconds={elapsed_seconds:.2f}")
                return cleaned_markdown, prompt_tokens, attempt, elapsed_seconds
            except Exception:
                self._emit_progress(f"formatting_failed: attempt={attempt}/{self.llm.max_retries} error={format_exception_summary()}")
                if attempt == self.llm.max_retries:
                    raise
                time.sleep(self.llm.retry_delay)
            attempt += 1

        raise RuntimeError("unreachable")

    def _emit_progress(self, message: str) -> None:
        """Send formatting progress to the configured observer."""
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _prompt_token_count(self, source_text: str) -> int:
        """Return rendered prompt token count for source text."""
        return len(self.encoder.encode(self.render_prompt(source_text), disallowed_special=()))

    def _token_limit(self) -> int:
        """Return configured prompt token limit."""
        return int(self.llm.context_window * self.skip_threshold_pct / 100)

    def _raise_if_oversized(self, prompt_tokens: int) -> None:
        """Raise when the rendered prompt exceeds the configured context threshold."""
        token_limit = self._token_limit()
        if prompt_tokens <= token_limit:
            return
        usage_pct = (prompt_tokens / self.llm.context_window * 100.0) if self.llm.context_window > 0 else 0.0
        raise OversizedDocumentError(
            f"document prompt contains {prompt_tokens:,} tokens ({usage_pct:.1f}% of context window "
            f"{self.llm.context_window:,}; threshold: {self.skip_threshold_pct}%/{token_limit:,} tokens)"
        )


def strip_think_tags(text: str) -> str:
    """Strip model reasoning tags from a response."""
    think_match = re.search(r"</think>\s*(.*)$", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return text


def format_exception_summary() -> str:
    """Return the active exception as a compact single-line progress summary."""
    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return "unknown"
    summary = f"{type(exc).__name__}: {' '.join(str(exc).split())}"
    if len(summary) > 240:
        return f"{summary[:237]}..."
    return summary


def validate_cleaned_markdown(source_text: str, cleaned_markdown: str) -> None:
    """Validate cleaned Markdown against basic output safety rules."""
    if not cleaned_markdown.strip():
        raise MarkdownValidationError("cleaned Markdown is empty")
    if contains_inline_html(cleaned_markdown):
        raise MarkdownValidationError("cleaned Markdown contains inline HTML")


def contains_inline_html(markdown: str) -> bool:
    """Return whether Markdown contains likely rendered HTML outside code."""
    searchable_markdown = strip_markdown_code(markdown)
    html_tag_names = {
        "a",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "button",
        "canvas",
        "code",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "em",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "iframe",
        "img",
        "input",
        "label",
        "li",
        "main",
        "nav",
        "ol",
        "option",
        "p",
        "pre",
        "script",
        "section",
        "select",
        "span",
        "strong",
        "style",
        "summary",
        "svg",
        "table",
        "tbody",
        "td",
        "textarea",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    for match in re.finditer(r"</?\s*([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>", searchable_markdown):
        if match.group(1).lower() in html_tag_names:
            return True
    return False


def strip_markdown_code(markdown: str) -> str:
    """Remove fenced and inline code before validating prose Markdown."""
    without_fences = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", "", without_fences)
