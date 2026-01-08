"""Topic segmentation agent using litellm."""

import json
import os

from litellm import completion
from pydantic import ValidationError

from src.agents.topic_segmentation.agent_prompts import RETRY_PROMPT_TEMPLATE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.agents.topic_segmentation.models import AgentSegmentationResponse
from src.agents.topic_segmentation.token_validator import validate_token_usage
from src.config import Config, LLMConfig


class TopicSegmentationAgent:
    """Agent that segments video transcripts by topics."""

    def __init__(self, llm_config: LLMConfig, config: Config) -> None:
        """Initialize the topic segmentation agent.

        Args:
            llm_config: LLM configuration for this agent.
            config: Full application configuration.

        Raises:
            KeyError: If required environment variables are missing.
        """
        self._llm_config = llm_config
        self._config = config
        self._system_prompt = SYSTEM_PROMPT
        self._user_prompt_template = USER_PROMPT_TEMPLATE
        self._retry_prompt_template = RETRY_PROMPT_TEMPLATE

        # Get API key from environment
        api_key = os.environ.get(llm_config.api_key_env)
        if api_key is None:
            raise KeyError(f"Environment variable '{llm_config.api_key_env}' not found")

        self._api_key = api_key

    def segment(
        self,
        simplified_transcript: str,
        retry_feedback: dict[str, str | bool] | None,
    ) -> AgentSegmentationResponse:
        """Segment a transcript by topics.

        Args:
            simplified_transcript: Simplified format transcript.
            retry_feedback: Critic feedback for retry (optional).

        Returns:
            Validated segmentation response.

        Raises:
            ValueError: If agent response is invalid.
        """
        # Build user message
        if retry_feedback is None:
            user_message = self._user_prompt_template.format(
                simplified_transcript=simplified_transcript,
            )
        else:
            user_message = self._retry_prompt_template.format(
                rating=retry_feedback["rating"],
                pass_status=retry_feedback["pass"],
                reasoning=retry_feedback["reasoning"],
                improvement_suggestions=retry_feedback["improvement_suggestions"],
                simplified_transcript=simplified_transcript,
            )

        # Validate token usage before calling LLM
        messages = [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": user_message}]

        try:
            token_count = validate_token_usage(
                messages=messages,
                context_window=self._llm_config.context_window,
                threshold=self._llm_config.context_window_threshold,
                encoding_name=self._config.getEncodingName(),
            )
            print(
                f"        [Agent] Token count: {token_count:,} tokens "
                f"({token_count / self._llm_config.context_window * 100:.1f}% of context window)"
            )
        except ValueError as e:
            print(f"        [Agent] ✗ Token validation failed: {e}")
            raise

        # Call litellm (use the same 'messages' variable)
        print("        [Agent] Calling LLM API...")
        response = completion(
            model=self._llm_config.model,
            api_base=self._llm_config.api_base,
            api_key=self._api_key,
            messages=messages,  # Reuse the messages we validated
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            timeout=600,
            stream=False,
        )

        print("        [Agent] Received LLM response, extracting content...")
        response_text = response.choices[0].message.content
        if response_text is None:
            raise ValueError("Agent produced empty response")

        print(f"        [Agent] Response length: {len(response_text)} chars")

        # Parse and validate
        try:
            print("        [Agent] Parsing JSON...")
            response_data = json.loads(response_text)
            print("        [Agent] Validating with Pydantic...")
            validated = AgentSegmentationResponse.model_validate(response_data)
            print(f"        [Agent] ✓ Validation successful, {len(validated.segments)} segments")
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"        [Agent] ✗ Validation failed: {e}")
            raise ValueError(f"Agent produced invalid response: {e}\nResponse: {response_text}") from e
