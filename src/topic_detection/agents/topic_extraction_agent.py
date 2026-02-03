"""Topic extraction agent using LLM for topic detection."""

import json
import re
import time

import litellm
from litellm.exceptions import BadRequestError
from pydantic import ValidationError

from src.config import LLMConfig
from src.topic_detection.agents.models import TopicDetectionResult
from src.topic_detection.agents.prompts import TopicDetectionPrompts


class TopicExtractionAgent:
    """Agent that extracts topics from text segments using an LLM.

    Uses LiteLLM to call an LLM (via LM Studio or other OpenAI-compatible API)
    and extracts topics at multiple granularity levels along with a description.

    Args:
        llm_config: LLM configuration with model, API settings, etc.
        max_retries: Number of retries on LLM parsing errors.
        retry_delay: Delay in seconds between retries.

    Example:
        >>> agent = TopicExtractionAgent(
        ...     config.get_topic_detection_config().topic_detection_llm,
        ...     max_retries=3,
        ...     retry_delay=2.0,
        ... )
        >>> result = agent.detect("This segment discusses the latest advances in AI...")
        >>> print(result.topics)
        ['AI', 'Machine Learning', 'GPT-4 capabilities']
        >>> print(result.description)
        'Discussion of recent AI advances including GPT-4.'
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        max_retries: int,
        retry_delay: float,
    ) -> None:
        self._config = llm_config
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def detect(self, segment_text: str) -> TopicDetectionResult:
        """Extract topics and description from a text segment.

        Args:
            segment_text: The transcript segment text to analyze.

        Returns:
            TopicDetectionResult with topics and description.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
                       or doesn't match the expected schema after all retries.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return self._call_llm(segment_text)
            except ValueError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    print(f"      Retry {attempt + 1}/{self._max_retries - 1} after error: {e}")
                    time.sleep(self._retry_delay)

        raise ValueError(f"Failed after {self._max_retries} attempts. Last error: {last_error}")

    def _call_llm(self, segment_text: str) -> TopicDetectionResult:
        """Make a single LLM call and parse the response.

        Args:
            segment_text: The transcript segment text to analyze.

        Returns:
            TopicDetectionResult with topics and description.

        Raises:
            ValueError: If the LLM response is empty or cannot be parsed.
        """
        # Build the messages for the completion call
        messages = [
            {"role": "system", "content": TopicDetectionPrompts.getSystemPrompt()},
            {"role": "user", "content": TopicDetectionPrompts.getUserPrompt(segment_text)},
        ]

        # Call the LLM
        try:
            response = litellm.completion(
                model=self._config.model,
                messages=messages,
                api_base=self._config.api_base,
                api_key=self._config.api_key,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )
        except BadRequestError as e:
            error_msg = str(e)
            if "No models loaded" in error_msg:
                raise BadRequestError(
                    message=f"No models loaded in LM Studio. Expected model: {self._config.model}. "
                    f"Load it with: lms load {self._config.model.split('/')[-1]}",
                    model=self._config.model,
                    llm_provider="openai",
                ) from e
            raise

        # Extract the response text
        response_text = response.choices[0].message.content
        if response_text is None or response_text.strip() == "":
            raise ValueError("LLM returned empty response (model may not be loaded or API unreachable)")

        # Parse and validate the JSON response
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> TopicDetectionResult:
        """Parse the LLM response text into a TopicDetectionResult.

        Args:
            response_text: Raw text response from the LLM.

        Returns:
            Validated TopicDetectionResult.

        Raises:
            ValueError: If parsing or validation fails.
        """
        # Try to extract JSON from the response
        cleaned = response_text.strip()

        # Handle "thinking" models that wrap reasoning in <think> tags
        # Extract content after </think> if present
        think_match = re.search(r"</think>\s*(.*)$", cleaned, re.DOTALL)
        if think_match:
            cleaned = think_match.group(1).strip()

        # Handle cases where the model might wrap JSON in markdown code blocks
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        # If still empty after cleaning, the response had no usable content
        if not cleaned:
            raise ValueError(f"No JSON content found after cleaning response. Raw response (first 500 chars): {response_text[:500]}")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:500]}") from e

        try:
            return TopicDetectionResult.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"LLM response does not match expected schema: {e}. Data: {data}") from e
