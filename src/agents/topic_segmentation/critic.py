"""Quality critic agent for topic segmentation."""

import json
import os
from pathlib import Path

from litellm import completion
from pydantic import ValidationError

from src.agents.topic_segmentation.models import AgentSegmentationResponse, CriticRating
from src.config import LLMConfig


class TopicSegmentationCritic:
    """Critic agent that evaluates segmentation quality."""

    def __init__(self, llm_config: LLMConfig, prompts_dir: Path) -> None:
        """Initialize the critic agent.

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
        self._system_prompt = self._load_prompt("critic_prompt_system.txt")
        self._user_prompt_template = self._load_prompt("critic_prompt_user.txt")

        # Get API key from environment
        api_key = os.environ.get(llm_config.api_key_env)
        if api_key is None:
            raise KeyError(f"Environment variable '{llm_config.api_key_env}' not found")

        self._api_key = api_key

    def evaluate(
        self,
        simplified_transcript: str,
        segmentation: AgentSegmentationResponse,
    ) -> CriticRating:
        """Evaluate a segmentation for quality.

        Args:
            simplified_transcript: Original simplified transcript.
            segmentation: Proposed segmentation to evaluate.

        Returns:
            Validated critic rating.

        Raises:
            ValueError: If critic response is invalid.
        """
        # Convert segmentation to JSON for prompt
        segmentation_json = json.dumps(segmentation.model_dump(), indent=2)

        # Build user message
        user_message = self._user_prompt_template.format(
            simplified_transcript=simplified_transcript,
            segmentation_json=segmentation_json,
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
            raise ValueError("Critic produced empty response")

        # Parse and validate
        try:
            response_data = json.loads(response_text)
            return CriticRating.model_validate(response_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Critic produced invalid response: {e}\nResponse: {response_text}") from e

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
