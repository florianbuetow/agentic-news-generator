"""Unit tests for WhisperLanguages utility class."""

from src.util.whisper_languages import WhisperLanguages


class TestWhisperLanguages:
    """Test cases for the WhisperLanguages utility class."""

    def test_get_supported_languages_returns_dict(self) -> None:
        """Test that get_supported_languages returns a dictionary."""
        languages = WhisperLanguages.get_supported_languages()
        assert isinstance(languages, dict)

    def test_get_supported_languages_includes_en(self) -> None:
        """Test that English (en) is in supported languages."""
        languages = WhisperLanguages.get_supported_languages()
        assert "en" in languages

    def test_get_supported_languages_includes_unknown(self) -> None:
        """Test that '??' (unknown) is in supported languages."""
        languages = WhisperLanguages.get_supported_languages()
        assert "??" in languages

    def test_get_supported_languages_count(self) -> None:
        """Test that exactly 101 languages are supported (100 Whisper + '??')."""
        languages = WhisperLanguages.get_supported_languages()
        assert len(languages) == 101

    def test_get_supported_languages_english_name(self) -> None:
        """Test that 'en' maps to 'english'."""
        languages = WhisperLanguages.get_supported_languages()
        assert languages["en"] == "english"

    def test_get_supported_languages_unknown_name(self) -> None:
        """Test that '??' maps to 'unknown'."""
        languages = WhisperLanguages.get_supported_languages()
        assert languages["??"] == "unknown"

    def test_get_supported_languages_keys_are_lowercase(self) -> None:
        """Test that all language keys are lowercase strings."""
        languages = WhisperLanguages.get_supported_languages()
        for key in languages:
            assert isinstance(key, str)
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_get_supported_languages_values_are_strings(self) -> None:
        """Test that all language values are strings."""
        languages = WhisperLanguages.get_supported_languages()
        for value in languages.values():
            assert isinstance(value, str)

    def test_get_supported_languages_is_immutable(self) -> None:
        """Test that modifying returned dict doesn't affect future calls."""
        languages1 = WhisperLanguages.get_supported_languages()
        original_count = len(languages1)

        # Modify the returned dict
        languages1["test"] = "test_language"

        # Get a new dict
        languages2 = WhisperLanguages.get_supported_languages()

        # Verify the new dict is unaffected
        assert len(languages2) == original_count
        assert "test" not in languages2

    def test_get_supported_languages_has_common_languages(self) -> None:
        """Test that common languages are present: en, ja, es, fr, de, zh."""
        languages = WhisperLanguages.get_supported_languages()
        common_languages = ["en", "ja", "es", "fr", "de", "zh"]

        for lang_code in common_languages:
            assert lang_code in languages, f"Common language '{lang_code}' not found"
            assert isinstance(languages[lang_code], str)
            assert len(languages[lang_code]) > 0, f"Language name for '{lang_code}' is empty"
