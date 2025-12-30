"""Topic segmentation agent using litellm."""

import json
import os
from pathlib import Path

from litellm import completion
from pydantic import ValidationError

from src.agents.topic_segmentation.models import AgentSegmentationResponse
from src.config import LLMConfig


class TopicSegmentationAgent:
    """Agent that segments video transcripts by topics."""

    def __init__(self, llm_config: LLMConfig, prompts_dir: Path) -> None:
        """Initialize the topic segmentation agent.

        Args:
            llm_config: LLM configuration for this agent.
            prompts_dir: Directory containing prompt files.

        Raises:
            FileNotFoundError: If prompt files are missing.
            KeyError: If required environment variables are missing.
        """
        self._llm_config = llm_config
        self._prompts_dir = prompts_dir

        # Load prompts
        self._system_prompt = self._load_prompt("agent_prompt_system.txt")
        self._user_prompt_template = self._load_prompt("agent_prompt_user.txt")
        self._retry_prompt_template = self._load_prompt("agent_prompt_user_retry.txt")

        # Get API key from environment
        api_key = os.environ.get(llm_config.api_key_env)
        if api_key is None:
            raise KeyError(f"Environment variable '{llm_config.api_key_env}' not found")

        self._api_key = api_key

    def segment(
        self,
        video_id: str,
        video_title: str,
        channel_name: str,
        simplified_transcript: str,
        retry_feedback: dict[str, str | bool] | None,
    ) -> AgentSegmentationResponse:
        """Segment a transcript by topics.

        Args:
            video_id: Unique video identifier.
            video_title: Video title.
            channel_name: YouTube channel name.
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
                video_id=video_id,
                video_title=video_title,
                channel_name=channel_name,
                simplified_transcript=simplified_transcript,
            )
        else:
            user_message = self._retry_prompt_template.format(
                rating=retry_feedback["rating"],
                pass_status=retry_feedback["pass"],
                reasoning=retry_feedback["reasoning"],
                improvement_suggestions=retry_feedback["improvement_suggestions"],
                video_id=video_id,
                video_title=video_title,
                channel_name=channel_name,
                simplified_transcript=simplified_transcript,
            )

        # Call litellm
        response = completion(
            model=self._llm_config.model,
            api_base=self._llm_config.api_base,
            api_key=self._api_key,
            messages=[{"role": "system", "content": self._system_prompt}, {"role": "user", "content": user_message}],
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
        )

        response_text = response.choices[0].message.content
        if response_text is None:
            raise ValueError("Agent produced empty response")

        # Parse and validate
        try:
            response_data = json.loads(response_text)
            return AgentSegmentationResponse.model_validate(response_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Agent produced invalid response: {e}\nResponse: {response_text}") from e

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt file.

        Args:
            filename: Name of the prompt file.

        Returns:
            Prompt text content.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        prompt_path = self._prompts_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
