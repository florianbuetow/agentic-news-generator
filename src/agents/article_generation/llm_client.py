"""LLM client adapter for article-generation agents."""

from __future__ import annotations

import logging
import time
import urllib.request

from litellm import completion

from src.config import LLMConfig

logger = logging.getLogger(__name__)


class LiteLLMClient:
    """LiteLLM-backed implementation of the LLM client protocol."""

    def check_connectivity(self, *, api_base: str, timeout_seconds: int) -> None:
        """Verify that the LLM API endpoint is reachable.

        Raises:
            ConnectionError: If the endpoint is unreachable.
        """
        models_url = f"{api_base.rstrip('/')}/models"
        logger.info("Pre-flight connectivity check: GET %s (timeout=%ds)", models_url, timeout_seconds)
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # nosec B310
                status = resp.status
            logger.info("Pre-flight check passed: status=%d", status)
        except Exception as exc:
            raise ConnectionError(f"LLM API endpoint unreachable at {api_base} — {type(exc).__name__}: {exc}") from exc

    def complete(self, *, llm_config: LLMConfig, messages: list[dict[str, str]]) -> str:
        """Call LiteLLM with configured retry behavior."""
        attempts = 0
        total_attempts = llm_config.max_retries + 1
        while True:
            attempts += 1
            logger.info(
                "LLM request: attempt %d/%d model=%s api_base=%s timeout=%ds max_tokens=%d temperature=%.2f",
                attempts,
                total_attempts,
                llm_config.model,
                llm_config.api_base,
                llm_config.timeout_seconds,
                llm_config.max_tokens,
                llm_config.temperature,
            )
            started_at = time.perf_counter()
            try:
                response = completion(
                    model=llm_config.model,
                    api_base=llm_config.api_base,
                    api_key=llm_config.api_key,
                    messages=messages,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens,
                    timeout=llm_config.timeout_seconds,
                    stream=False,
                )
                elapsed = time.perf_counter() - started_at
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")
                response_chars = len(content)
                logger.info(
                    "LLM response: attempt %d/%d succeeded in %.1fs, response_chars=%d",
                    attempts,
                    total_attempts,
                    elapsed,
                    response_chars,
                )
                return content
            except Exception as exc:
                elapsed = time.perf_counter() - started_at
                logger.error(
                    "LLM request failed: attempt %d/%d after %.1fs — %s: %s",
                    attempts,
                    total_attempts,
                    elapsed,
                    type(exc).__name__,
                    exc,
                )

                if attempts >= total_attempts:
                    logger.error("LLM retries exhausted after %d attempts", total_attempts)
                    raise

                logger.info("LLM retrying in %.1fs...", llm_config.retry_delay)
                time.sleep(llm_config.retry_delay)
