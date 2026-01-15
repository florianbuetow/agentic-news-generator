"""Helper class for building MLX Whisper transcription arguments."""


class TranscriptionArgumentHelper:
    """Builds arguments and prompts for MLX Whisper transcription.

    Note: For translation tasks (non-English to English), we intentionally
    return an empty prompt because Whisper's translate task works better
    without any prompt guidance. Including foreign language text in prompts
    can confuse Whisper and cause it to output in the source language.
    """

    # Prompt template for English transcription only
    PROMPT_TEMPLATE_ENGLISH = "This is a YouTube video with the title: {title}. Description: {description}"

    @classmethod
    def build_prompt(
        cls,
        language: str,
        language_name: str,
        title: str | None,
        description: str | None,
    ) -> str:
        """Build language-specific prompt for transcription.

        Args:
            language: ISO language code (e.g., 'en', 'de')
            language_name: Full language name (e.g., 'english', 'german')
            title: Video title from metadata (None if unavailable)
            description: Video description from metadata (None if unavailable)

        Returns:
            Prompt string for English transcription, or empty string for translation tasks
        """
        # For non-English (translation tasks), use empty prompt
        # Whisper's translate task works better without prompts containing foreign language text
        if language != "en":
            return ""

        # For English transcription, use metadata if available
        if title is None:
            return ""

        # Use empty string if description is None (description is optional metadata)
        return cls.PROMPT_TEMPLATE_ENGLISH.format(
            title=title,
            description=description if description is not None else "",
        )
