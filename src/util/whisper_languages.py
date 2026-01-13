"""Utility for Whisper language support."""


class WhisperLanguages:
    """Utility class for retrieving Whisper-supported languages."""

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Get dictionary of Whisper-supported languages.

        Returns:
            Dictionary mapping language codes (e.g., 'en', 'ja') to
            English language names (e.g., 'english', 'japanese').
            Includes special '??' code mapping to 'unknown'.

        Note:
            Imports from mlx_whisper.tokenizer.LANGUAGES (verified available).
            Contains 100 Whisper-supported languages plus '??' for unknown.

        Examples:
            >>> langs = WhisperLanguages.get_supported_languages()
            >>> langs['en']
            'english'
            >>> langs['ja']
            'japanese'
            >>> langs['??']
            'unknown'
            >>> len(langs)
            101
        """
        from mlx_whisper.tokenizer import LANGUAGES

        # Create new dict with Whisper languages + our special '??' code
        supported_languages = dict(LANGUAGES)  # Copy to avoid modifying original
        supported_languages["??"] = "unknown"

        return supported_languages
