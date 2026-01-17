"""Topic extraction agent using LLM for topic detection."""

import json
import os

import litellm
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

    Example:
        >>> agent = TopicExtractionAgent(config.get_topic_detection_config().topic_detection_llm)
        >>> result = agent.detect("This segment discusses the latest advances in AI...")
        >>> print(result.topics)
        ['AI', 'Machine Learning', 'GPT-4 capabilities']
        >>> print(result.description)
        'Discussion of recent AI advances including GPT-4.'
    """

    def __init__(self, llm_config: LLMConfig) -> None:
        self._config = llm_config

    def detect(self, segment_text: str) -> TopicDetectionResult:
        """Extract topics and description from a text segment.

        Args:
            segment_text: The transcript segment text to analyze.

        Returns:
            TopicDetectionResult with topics and description.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
                       or doesn't match the expected schema.
        """
        # Get API key from environment (empty string for local LLM without auth)
        api_key = os.environ.get(self._config.api_key_env)
        if api_key is None:
            api_key = ""

        # Build the messages for the completion call
        messages = [
            {"role": "system", "content": TopicDetectionPrompts.getSystemPrompt()},
            {"role": "user", "content": TopicDetectionPrompts.getUserPrompt(segment_text)},
        ]

        # Call the LLM
        response = litellm.completion(
            model=self._config.model,
            messages=messages,
            api_base=self._config.api_base,
            api_key=api_key,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        # Extract the response text
        response_text = response.choices[0].message.content
        if response_text is None:
            raise ValueError("LLM returned empty response")

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
        # Handle cases where the model might wrap JSON in markdown code blocks
        cleaned = response_text.strip()

        # Remove markdown code block if present
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:500]}") from e

        try:
            return TopicDetectionResult.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"LLM response does not match expected schema: {e}. Data: {data}") from e
