"""Article generation agent using litellm with automatic JSON validation."""

import json
from datetime import UTC, datetime
from typing import Literal, cast

from litellm import completion

from src.agents.article_generation import agent_prompts
from src.agents.article_generation.models import ArticleGenerationResult, ArticleMetadata, ArticleResponse
from src.config import Config, LLMConfig
from src.util.token_validator import validate_token_usage


class ArticleWriterAgent:
    """Agent that generates science journalism articles from topic transcripts."""

    # Class-level prompt templates (avoids module-level constants)
    _SYSTEM_PROMPT_TEMPLATE = agent_prompts.SYSTEM_PROMPT
    _USER_PROMPT_TEMPLATE = agent_prompts.USER_PROMPT_TEMPLATE

    def __init__(self, llm_config: LLMConfig, config: Config) -> None:
        """Initialize the article generation agent.

        Args:
            llm_config: LLM configuration for this agent.
            config: Full application configuration.

        Raises:
            KeyError: If required environment variables are missing.
        """
        self._llm_config = llm_config
        self._config = config
        self._api_key = llm_config.api_key

    def _build_metadata(self, source_metadata: dict[str, str | None], style_mode: str, target_length_words: str) -> ArticleMetadata:
        """Build article metadata.

        Args:
            source_metadata: Source metadata dictionary.
            style_mode: Writing style mode.
            target_length_words: Target word count.

        Returns:
            ArticleMetadata instance.

        Raises:
            ValueError: If required metadata fields are missing.
        """
        topic_slug_value = source_metadata.get("topic_slug")
        if topic_slug_value is None:
            raise ValueError("Required metadata field 'topic_slug' is missing")

        topic_title_value = source_metadata.get("topic_title")
        if topic_title_value is None:
            raise ValueError("Required metadata field 'topic_title' is missing")

        channel_name_value = source_metadata.get("channel_name")
        if channel_name_value is None:
            raise ValueError("Required metadata field 'channel_name' is missing")

        video_id_value = source_metadata.get("video_id")
        if video_id_value is None:
            raise ValueError("Required metadata field 'video_id' is missing")

        video_title_value = source_metadata.get("video_title")
        if video_title_value is None:
            raise ValueError("Required metadata field 'video_title' is missing")

        source_file_value = source_metadata.get("source_file")
        if source_file_value is None:
            raise ValueError("Required metadata field 'source_file' is missing")

        return ArticleMetadata(
            topic_slug=topic_slug_value,
            topic_title=topic_title_value,
            style_mode=cast(Literal["NATURE_NEWS", "SCIAM_MAGAZINE"], style_mode),
            target_length_words=target_length_words,
            source_channel=channel_name_value,
            source_video_id=video_id_value,
            source_video_title=video_title_value,
            source_publish_date=source_metadata.get("publish_date"),
            source_file=source_file_value,
            generated_at=datetime.now(UTC).isoformat(),
        )

    def generate_article(
        self,
        source_text: str,
        source_metadata: dict[str, str | None],
        style_mode: str,
        target_length_words: str,
    ) -> ArticleGenerationResult:
        """Generate an article from source transcript.

        Args:
            source_text: Aggregated transcript text.
            source_metadata: Video metadata (channel_name, video_id, video_title, publish_date).
            style_mode: Writing style ("NATURE_NEWS" or "SCIAM_MAGAZINE").
            target_length_words: Target word count (e.g., "900-1200").

        Returns:
            ArticleGenerationResult with success/failure and article data.
        """
        # Validate style_mode against allowed styles
        allowed_styles = self._config.get_allowed_article_styles()
        if style_mode not in allowed_styles:
            return ArticleGenerationResult(
                success=False,
                article=None,
                metadata=None,
                error=f"Invalid style_mode '{style_mode}'. Allowed styles: {allowed_styles}",
            )

        try:
            # Build prompts
            system_prompt = self._SYSTEM_PROMPT_TEMPLATE.format(
                style_mode=style_mode,
                target_length_words=target_length_words,
            )

            # Extract metadata values
            channel_name_value = source_metadata.get("channel_name")
            if channel_name_value is None:
                raise ValueError("Required metadata field 'channel_name' is missing")

            video_title_value = source_metadata.get("video_title")
            if video_title_value is None:
                raise ValueError("Required metadata field 'video_title' is missing")

            video_id_value = source_metadata.get("video_id")
            if video_id_value is None:
                raise ValueError("Required metadata field 'video_id' is missing")

            publish_date_value = source_metadata.get("publish_date")
            if publish_date_value is None:
                raise ValueError("Required metadata field 'publish_date' is missing")

            user_prompt = self._USER_PROMPT_TEMPLATE.format(
                style_mode=style_mode,
                target_length_words=target_length_words,
                source_text=source_text,
                channel_name=channel_name_value,
                video_title=video_title_value,
                video_id=video_id_value,
                publish_date=publish_date_value,
            )

            # Validate token usage before calling LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

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

            # Call litellm (model will follow prompt instructions for JSON output)
            print("        [Agent] Calling LLM API...")

            response = completion(
                model=self._llm_config.model,
                api_base=self._llm_config.api_base,
                api_key=self._api_key,
                messages=messages,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
                timeout=self._config.get_article_timeout_seconds(),
                stream=False,
            )

            print("        [Agent] Received LLM response, extracting content...")
            response_text = response.choices[0].message.content
            if response_text is None:
                raise ValueError("Agent produced empty response")

            print(f"        [Agent] Response length: {len(response_text)} chars")

            # Strip markdown code fences if present
            response_text = response_text.strip()
            if (response_text.startswith("```json") or response_text.startswith("```")) and response_text.endswith("```"):
                # Remove opening fence
                response_text = response_text[7:] if response_text.startswith("```json") else response_text[3:]
                # Remove closing fence
                response_text = response_text[:-3].strip()
                print("        [Agent] Stripped markdown code fences from response")

            # Parse and validate
            print("        [Agent] Parsing JSON...")
            response_data = json.loads(response_text)
            print("        [Agent] Validating with Pydantic...")
            validated_article = ArticleResponse.model_validate(response_data)
            print("        [Agent] ✓ Validation successful")

            # Build metadata
            metadata = self._build_metadata(source_metadata, style_mode, target_length_words)

            return ArticleGenerationResult(
                success=True,
                article=validated_article,
                metadata=metadata,
                error=None,
            )

        except Exception as e:
            print(f"        [Agent] ✗ Unexpected error: {e}")
            return ArticleGenerationResult(
                success=False,
                article=None,
                metadata=None,
                error=f"Unexpected error: {e}",
            )
