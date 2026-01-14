"""Utility for Whisper language support."""


class WhisperLanguages:
    """Utility class for retrieving Whisper-supported languages."""

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Get dictionary of Whisper-supported languages.

        Returns:
            Dictionary mapping language codes (e.g., 'en', 'ja') to
            English language names (e.g., 'english', 'japanese').

        Note:
            Imports from mlx_whisper.tokenizer.LANGUAGES (verified available).
            Contains 100 Whisper-supported languages.

        Examples:
            >>> langs = WhisperLanguages.get_supported_languages()
            >>> langs['en']
            'english'
            >>> langs['ja']
            'japanese'
        """
        from mlx_whisper.tokenizer import LANGUAGES

        return dict(LANGUAGES)
