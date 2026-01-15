"""Unit tests for LanguageDetector utility class."""

from pathlib import Path

import pytest

from src.config import Config
from src.nlp import DetectionResult, LanguageDetector


class TestDetectionResult:
    """Test cases for DetectionResult dataclass."""

    def test_detection_result_creation(self) -> None:
        """Test creating a DetectionResult instance."""
        result = DetectionResult(language="en", confidence=0.95)
        assert result.language == "en"
        assert result.confidence == 0.95

    def test_detection_result_fields(self) -> None:
        """Test that DetectionResult has correct fields."""
        result = DetectionResult(language="de", confidence=0.87)
        assert hasattr(result, "language")
        assert hasattr(result, "confidence")

    def test_detection_result_types(self) -> None:
        """Test that DetectionResult fields have correct types."""
        result = DetectionResult(language="fr", confidence=0.99)
        assert isinstance(result.language, str)
        assert isinstance(result.confidence, float)


class TestLanguageDetectorSupportedLanguages:
    """Test cases for LanguageDetector supported languages."""

    def test_supported_languages_count(self) -> None:
        """Test that exactly 176 languages are supported."""
        assert len(LanguageDetector.SUPPORTED_LANGUAGES) == 176

    def test_supported_languages_includes_common(self) -> None:
        """Test that common languages are present."""
        common_languages = ["en", "de", "fr", "es", "it", "ja", "zh", "ru", "ar", "pt"]
        for lang_code in common_languages:
            assert lang_code in LanguageDetector.SUPPORTED_LANGUAGES

    def test_supported_languages_keys_are_lowercase(self) -> None:
        """Test that all language codes are lowercase."""
        for key in LanguageDetector.SUPPORTED_LANGUAGES:
            assert key == key.lower(), f"Language code '{key}' is not lowercase"

    def test_supported_languages_values_are_strings(self) -> None:
        """Test that all language names are non-empty strings."""
        for value in LanguageDetector.SUPPORTED_LANGUAGES.values():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_supported_languages_specific_mappings(self) -> None:
        """Test specific language code to name mappings."""
        mappings = {
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        for code, name in mappings.items():
            assert LanguageDetector.SUPPORTED_LANGUAGES[code] == name


class TestLanguageDetectorInitialization:
    """Test cases for LanguageDetector initialization."""

    def test_init_missing_model_raises_error(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when model is missing."""
        model_path = tmp_path / "nonexistent_model.ftz"
        with pytest.raises(FileNotFoundError, match="Model not found"):
            LanguageDetector(model_path=model_path)

    def test_init_error_message_suggests_just_init(self, tmp_path: Path) -> None:
        """Test that error message suggests running 'just init'."""
        model_path = tmp_path / "nonexistent_model.ftz"
        with pytest.raises(FileNotFoundError, match="just init"):
            LanguageDetector(model_path=model_path)


class TestLanguageDetectorWithModel:
    """Integration tests for LanguageDetector with actual model.

    These tests require the FastText model to be downloaded via 'just init'.
    """

    @pytest.fixture(scope="class")
    def detector(self) -> LanguageDetector:
        """Create a LanguageDetector instance with downloaded model."""
        config = Config(Path("config/config.yaml"))
        model_path = config.getDataModelsDir() / "fasttext" / "lid.176.ftz"

        if not model_path.exists():
            pytest.skip("FastText model not downloaded. Run 'just init' first.")

        return LanguageDetector(model_path=model_path)

    def test_detect_english(self, detector: LanguageDetector) -> None:
        """Test detecting English text."""
        result = detector.detect("Hello, how are you?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "en"
        assert result.confidence > 0.5

    def test_detect_french(self, detector: LanguageDetector) -> None:
        """Test detecting French text."""
        result = detector.detect("Bonjour, comment allez-vous?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "fr"
        assert result.confidence > 0.5

    def test_detect_german(self, detector: LanguageDetector) -> None:
        """Test detecting German text."""
        result = detector.detect("Guten Tag, wie geht es Ihnen?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "de"
        assert result.confidence > 0.5

    def test_detect_spanish(self, detector: LanguageDetector) -> None:
        """Test detecting Spanish text."""
        result = detector.detect("Hola, ¿cómo estás?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "es"
        assert result.confidence > 0.3

    def test_detect_italian(self, detector: LanguageDetector) -> None:
        """Test detecting Italian text."""
        result = detector.detect("Ciao, come stai?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "it"
        assert result.confidence > 0.5

    def test_detect_russian(self, detector: LanguageDetector) -> None:
        """Test detecting Russian text."""
        result = detector.detect("Привет, как дела?", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "ru"
        assert result.confidence > 0.5

    def test_detect_empty_string(self, detector: LanguageDetector) -> None:
        """Test detecting empty string returns empty result."""
        result = detector.detect("", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == ""
        assert result.confidence == 0.0

    def test_detect_whitespace_only(self, detector: LanguageDetector) -> None:
        """Test detecting whitespace-only string returns empty result."""
        result = detector.detect("   \n\t  ", k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == ""
        assert result.confidence == 0.0

    def test_detect_top_k_returns_list(self, detector: LanguageDetector) -> None:
        """Test that detect with k>1 returns a list of results."""
        results = detector.detect("Hello world", k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DetectionResult)
            assert isinstance(result.language, str)
            assert isinstance(result.confidence, float)

    def test_detect_top_k_sorted_by_confidence(self, detector: LanguageDetector) -> None:
        """Test that top-k results are sorted by confidence (descending)."""
        results = detector.detect("Bonjour le monde", k=5)
        assert isinstance(results, list)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_detect_empty_list_for_empty_text_with_k_greater_1(self, detector: LanguageDetector) -> None:
        """Test that empty text with k>1 returns empty list."""
        results = detector.detect("", k=3)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_detect_language_returns_code(self, detector: LanguageDetector) -> None:
        """Test detect_language returns just the language code."""
        lang_code = detector.detect_language("Hello world")
        assert isinstance(lang_code, str)
        assert lang_code == "en"

    def test_detect_language_empty_string(self, detector: LanguageDetector) -> None:
        """Test detect_language returns empty string for empty input."""
        lang_code = detector.detect_language("")
        assert lang_code == ""

    def test_detect_batch_multiple_texts(self, detector: LanguageDetector) -> None:
        """Test batch detection with multiple texts."""
        texts = ["Hello", "Bonjour", "Hola"]
        results = detector.detect_batch(texts, k=1)
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_detect_batch_with_k_greater_1(self, detector: LanguageDetector) -> None:
        """Test batch detection with k>1 returns lists."""
        texts = ["Hello", "Bonjour"]
        results = detector.detect_batch(texts, k=3)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, list)
            assert len(result) == 3

    def test_detect_batch_empty_list(self, detector: LanguageDetector) -> None:
        """Test batch detection with empty list."""
        results = detector.detect_batch([], k=1)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_model_path_property(self, detector: LanguageDetector) -> None:
        """Test that model_path property returns correct path."""
        model_path = detector.model_path
        assert isinstance(model_path, Path)
        assert model_path.exists()
        assert model_path.name == "lid.176.ftz"

    def test_get_supported_languages_returns_copy(self, detector: LanguageDetector) -> None:
        """Test that get_supported_languages returns a copy of the dict."""
        langs1 = detector.get_supported_languages()
        langs2 = detector.get_supported_languages()

        # Modify first dict
        langs1["test"] = "Test Language"

        # Second dict should be unaffected
        assert "test" not in langs2
        assert len(langs2) == 176

    def test_get_supported_languages_returns_all_176(self, detector: LanguageDetector) -> None:
        """Test that get_supported_languages returns all 176 languages."""
        langs = detector.get_supported_languages()
        assert len(langs) == 176

    def test_is_supported_valid_code(self, detector: LanguageDetector) -> None:
        """Test is_supported returns True for valid language codes."""
        assert detector.is_supported("en") is True
        assert detector.is_supported("de") is True
        assert detector.is_supported("fr") is True

    def test_is_supported_invalid_code(self, detector: LanguageDetector) -> None:
        """Test is_supported returns False for invalid language codes."""
        assert detector.is_supported("xyz") is False
        assert detector.is_supported("invalid") is False
        assert detector.is_supported("") is False

    def test_is_supported_case_insensitive(self, detector: LanguageDetector) -> None:
        """Test is_supported is case-insensitive."""
        assert detector.is_supported("EN") is True
        assert detector.is_supported("En") is True
        assert detector.is_supported("eN") is True

    def test_get_language_name_valid_code(self, detector: LanguageDetector) -> None:
        """Test get_language_name returns correct names for valid codes."""
        assert detector.get_language_name("en") == "English"
        assert detector.get_language_name("de") == "German"
        assert detector.get_language_name("fr") == "French"

    def test_get_language_name_invalid_code(self, detector: LanguageDetector) -> None:
        """Test get_language_name returns None for invalid codes."""
        assert detector.get_language_name("xyz") is None
        assert detector.get_language_name("invalid") is None
        assert detector.get_language_name("") is None

    def test_get_language_name_case_insensitive(self, detector: LanguageDetector) -> None:
        """Test get_language_name is case-insensitive."""
        assert detector.get_language_name("EN") == "English"
        assert detector.get_language_name("De") == "German"
        assert detector.get_language_name("FR") == "French"

    def test_detect_filters_non_alpha_words(self, detector: LanguageDetector) -> None:
        """Test that non-alpha words are filtered before detection."""
        # Text with numbers and symbols mixed with English words
        text = "Hello 123 456 world !!! ### test"
        result = detector.detect(text, k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "en"

    def test_detect_only_non_alpha_returns_unknown_language(self, detector: LanguageDetector) -> None:
        """Test that text with only non-alpha characters returns '??' language code."""
        # Only numbers and symbols, no alphabetic characters
        text = "123 456 789 !!! ### $$$ %%% @@@"
        result = detector.detect(text, k=1)
        assert isinstance(result, DetectionResult)
        assert result.language == "??"
        assert result.confidence == 0.0

    def test_detect_only_non_alpha_with_k_greater_1(self, detector: LanguageDetector) -> None:
        """Test that non-alpha only text with k>1 returns list with '??' language."""
        text = "123 456 !!! ###"
        results = detector.detect(text, k=3)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].language == "??"
        assert results[0].confidence == 0.0

    def test_detect_language_only_non_alpha(self, detector: LanguageDetector) -> None:
        """Test detect_language returns '??' for non-alpha only text."""
        text = "123 456 !!! ###"
        lang_code = detector.detect_language(text)
        assert lang_code == "??"

    def test_filter_alpha_words_keeps_alpha(self) -> None:
        """Test that filter_alpha_words keeps words with alphabetic characters."""
        filtered = LanguageDetector.filter_alpha_words("Hello 123 world test")
        assert "Hello" in filtered
        assert "world" in filtered
        assert "test" in filtered

    def test_filter_alpha_words_removes_non_alpha(self) -> None:
        """Test that filter_alpha_words removes pure non-alpha words."""
        filtered = LanguageDetector.filter_alpha_words("Hello 123 456 world")
        assert "123" not in filtered
        assert "456" not in filtered

    def test_filter_alpha_words_keeps_mixed_words(self) -> None:
        """Test that filter_alpha_words keeps words with both alpha and non-alpha."""
        filtered = LanguageDetector.filter_alpha_words("test123 hello456world")
        assert "test123" in filtered
        assert "hello456world" in filtered

    def test_filter_alpha_words_empty_input(self) -> None:
        """Test that filter_alpha_words handles empty input."""
        filtered = LanguageDetector.filter_alpha_words("")
        assert filtered == ""

    def test_filter_alpha_words_only_non_alpha(self) -> None:
        """Test that filter_alpha_words returns empty for non-alpha only input."""
        filtered = LanguageDetector.filter_alpha_words("123 456 !!! ###")
        assert filtered == ""
