"""Base classes and protocols for article-generation agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel

from src.config import Config, LLMConfig
from src.util.token_validator import validate_token_usage

if TYPE_CHECKING:
    from src.agents.article_generation.models import ArticleResponse, Concern, Verdict

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM interactions."""

    def complete(self, *, llm_config: LLMConfig, messages: list[dict[str, str]]) -> str:
        """Send a completion request and return text response."""
        ...


class KnowledgeBaseRetriever(Protocol):
    """Protocol for KB retrieval."""

    def search(self, *, query: str, top_k: int, timeout_seconds: int) -> list[dict[str, str]]:
        """Search the knowledge base."""
        ...


class PerplexityClient(Protocol):
    """Protocol for Perplexity search calls."""

    def search(self, *, query: str, model: str, timeout_seconds: int) -> dict[str, object]:
        """Execute a search request."""
        ...


T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    """Base class for all article-generation agents."""

    def __init__(self, *, llm_config: LLMConfig, config: Config, llm_client: LLMClient) -> None:
        self._llm_config = llm_config
        self._config = config
        self._llm_client = llm_client
        self._agent_name = self.__class__.__name__
        self._logger = logging.getLogger(f"{__name__}.{self._agent_name}")

    def _validate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Validate token usage before an LLM call."""
        token_count = validate_token_usage(
            messages=messages,
            context_window=self._llm_config.context_window,
            threshold=self._llm_config.context_window_threshold,
            encoding_name=self._config.getEncodingName(),
        )
        context_pct = (token_count / self._llm_config.context_window) * 100
        self._logger.info(
            "Token validation: %d tokens (%.1f%% of %d context window, threshold=%d%%)",
            token_count,
            context_pct,
            self._llm_config.context_window,
            self._llm_config.context_window_threshold,
        )
        return token_count

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the configured LLM client."""
        token_count = self._validate_tokens(messages)
        self._logger.info(
            "Calling LLM: model=%s messages=%d validated_tokens=%d",
            self._llm_config.model,
            len(messages),
            token_count,
        )
        started_at = time.perf_counter()
        response = self._llm_client.complete(llm_config=self._llm_config, messages=messages)
        elapsed_seconds = time.perf_counter() - started_at
        self._logger.info("LLM call completed in %.1fs, response_chars=%d", elapsed_seconds, len(response))
        return response

    def _parse_json_response(self, response: str, model_class: type[T]) -> T:
        """Parse and validate JSON output, stripping markdown fences when present."""
        text = response.strip()
        if text.startswith("```"):
            text = text[7:] if text.startswith("```json") else text[3:]
            text = text.rsplit("```", 1)[0].strip()
        self._logger.debug("Parsing JSON response into %s (chars=%d)", model_class.__name__, len(text))
        result = model_class.model_validate_json(text)
        self._logger.info("Parsed response into %s successfully", model_class.__name__)
        return result


class BaseSpecialistAgent(BaseAgent, ABC):
    """Base class for specialist agents returning verdicts."""

    @abstractmethod
    def evaluate(
        self,
        *,
        concern: Concern,
        article: ArticleResponse,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_requirements: str,
    ) -> Verdict:
        """Evaluate a concern and return a verdict."""
