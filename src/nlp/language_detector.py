"""Language detection using FastText's pre-trained language identification models.

Model used:
- lid.176.ftz (917KB): Compressed model with good accuracy

Supports 176 languages. Models trained on Wikipedia, Tatoeba, and SETimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fasttext


@dataclass
class DetectionResult:
    """Result of language detection."""

    language: str  # ISO 639-1 code (e.g., "en", "de", "fr")
    confidence: float  # 0.0 to 1.0


class LanguageDetector:
    """FastText-based language detector supporting 176 languages.

    Example:
        >>> from pathlib import Path
        >>> from src.config import Config
        >>> config = Config(Path("config/config.yaml"))
        >>> model_path = config.getDataModelsDir() / "fasttext" / "lid.176.ftz"
        >>> detector = LanguageDetector(model_path=model_path)
        >>> detector.detect_language("Hello, how are you?")
        'en'
        >>> detector.detect("Guten Tag, wie geht es Ihnen?", k=1)
        DetectionResult(language='de', confidence=0.987...)
    """

    LABEL_PREFIX = "__label__"

    # All 176 languages supported by FastText lid.176 models
    # Keys: ISO 639-1 codes (or ISO 639-3 where no 2-letter code exists)
    # Values: Human-readable language names
    # codespell:ignore als,te
    SUPPORTED_LANGUAGES: dict[str, str] = {
        "af": "Afrikaans",
        "als": "Alemannic",  # codespell:ignore als
        "am": "Amharic",
        "an": "Aragonese",
        "ar": "Arabic",
        "arz": "Egyptian Arabic",
        "as": "Assamese",
        "ast": "Asturian",
        "av": "Avaric",
        "az": "Azerbaijani",
        "azb": "South Azerbaijani",
        "ba": "Bashkir",
        "bar": "Bavarian",
        "bcl": "Central Bikol",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bh": "Bihari",
        "bn": "Bengali",
        "bo": "Tibetan",
        "bpy": "Bishnupriya Manipuri",
        "br": "Breton",
        "bs": "Bosnian",
        "bxr": "Buryat",
        "ca": "Catalan",
        "cbk": "Chavacano",
        "ce": "Chechen",
        "ceb": "Cebuano",
        "ckb": "Central Kurdish",
        "co": "Corsican",
        "cs": "Czech",
        "cv": "Chuvash",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "diq": "Dimli",
        "dsb": "Lower Sorbian",
        "dty": "Dotyali",
        "dv": "Divehi",
        "el": "Greek",
        "eml": "Emilian-Romagnol",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "fr": "French",
        "frr": "Northern Frisian",
        "fy": "Western Frisian",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "gl": "Galician",
        "gn": "Guarani",
        "gom": "Goan Konkani",
        "gu": "Gujarati",
        "gv": "Manx",
        "he": "Hebrew",
        "hi": "Hindi",
        "hif": "Fiji Hindi",
        "hr": "Croatian",
        "hsb": "Upper Sorbian",
        "ht": "Haitian Creole",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ilo": "Iloko",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jbo": "Lojban",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "krc": "Karachay-Balkar",
        "ku": "Kurdish",
        "kv": "Komi",
        "kw": "Cornish",
        "ky": "Kyrgyz",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lez": "Lezghian",
        "li": "Limburgish",
        "lmo": "Lombard",
        "lo": "Lao",
        "lrc": "Northern Luri",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mai": "Maithili",
        "mg": "Malagasy",
        "mhr": "Eastern Mari",
        "min": "Minangkabau",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mr": "Marathi",
        "mrj": "Western Mari",
        "ms": "Malay",
        "mt": "Maltese",
        "mwl": "Mirandese",
        "my": "Burmese",
        "myv": "Erzya",
        "mzn": "Mazanderani",
        "nah": "Nahuatl",
        "nap": "Neapolitan",
        "nds": "Low German",
        "ne": "Nepali",
        "new": "Newari",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan",
        "or": "Odia",
        "os": "Ossetic",
        "pa": "Punjabi",
        "pam": "Pampanga",
        "pfl": "Palatine German",
        "pl": "Polish",
        "pms": "Piedmontese",
        "pnb": "Western Punjabi",
        "ps": "Pashto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "ro": "Romanian",
        "ru": "Russian",
        "rue": "Rusyn",
        "sa": "Sanskrit",
        "sah": "Sakha",
        "sc": "Sardinian",
        "scn": "Sicilian",
        "sco": "Scots",
        "sd": "Sindhi",
        "sh": "Serbo-Croatian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",  # codespell:ignore te
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "tyv": "Tuvan",
        "ug": "Uyghur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vec": "Venetian",
        "vep": "Veps",
        "vi": "Vietnamese",
        "vls": "West Flemish",
        "vo": "Volapük",
        "wa": "Walloon",
        "war": "Waray",
        "wuu": "Wu Chinese",
        "xal": "Kalmyk",
        "xmf": "Mingrelian",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "yue": "Cantonese",
        "zh": "Chinese",
    }

    def __init__(self, model_path: str | Path) -> None:
        """Initialize the language detector.

        Args:
            model_path: Path to the FastText model file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        self._model: fasttext.FastText._FastText | None = None
        self._model_path = Path(model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found at {self._model_path}. Run 'just init' to download the required models.")

        self._load_model()

    def _load_model(self) -> None:
        """Load the FastText model from disk."""
        # Suppress FastText warning about deprecated load_model
        fasttext.FastText.eprint = lambda x: None
        self._model = fasttext.load_model(str(self._model_path))

    @staticmethod
    def filter_alpha_words(text: str) -> str:
        """Filter text to keep only words containing alphabetic characters.

        Args:
            text: Input text to filter.

        Returns:
            Text with only words containing at least one alphabetic character.
            Returns empty string if no alphabetic words remain.
        """
        words = text.split()
        alpha_words = [word for word in words if any(c.isalpha() for c in word)]
        return " ".join(alpha_words)

    def detect(self, text: str, k: int) -> DetectionResult | list[DetectionResult]:
        """Detect the language of the given text.

        Filters out non-alphabetic words before detection. If no alphabetic words
        remain after filtering, returns '??' as the language code.

        Args:
            text: The text to analyze (should be UTF-8 encoded).
            k: Number of top predictions to return.

        Returns:
            DetectionResult if k=1, otherwise list of DetectionResult sorted by confidence.
            Returns '??' language code if no alphabetic words found.
        """
        if not text or text.isspace():
            if k == 1:
                return DetectionResult(language="", confidence=0.0)
            return []

        # Filter to keep only words containing alphabetic characters
        filtered_text = self.filter_alpha_words(text)

        # If no alphabetic words remain, return '??' as language code
        if not filtered_text or filtered_text.isspace():
            if k == 1:
                return DetectionResult(language="??", confidence=0.0)
            return [DetectionResult(language="??", confidence=0.0)]

        # FastText doesn't handle newlines well - replace with spaces
        clean_text = re.sub(r"\s+", " ", filtered_text.strip())

        if self._model is None:
            raise RuntimeError("Model must be loaded before calling detect()")
        labels, scores = self._model.predict(clean_text, k=k)

        results = [
            DetectionResult(
                language=label.replace(self.LABEL_PREFIX, ""),
                confidence=float(score),
            )
            for label, score in zip(labels, scores, strict=True)
        ]

        return results[0] if k == 1 else results

    def detect_language(self, text: str) -> str:
        """Detect the language and return just the language code.

        Args:
            text: The text to analyze.

        Returns:
            ISO 639-1 language code (e.g., "en", "de", "fr").
            Returns '??' if text contains no alphabetic words.
            Returns empty string for empty/whitespace input.
        """
        result = self.detect(text, k=1)
        return result.language if isinstance(result, DetectionResult) else ""

    def detect_batch(self, texts: list[str], k: int) -> list[DetectionResult | list[DetectionResult]]:
        """Detect languages for multiple texts.

        Args:
            texts: List of texts to analyze.
            k: Number of top predictions per text.

        Returns:
            List of results, one per input text.
        """
        return [self.detect(text, k=k) for text in texts]

    @property
    def model_path(self) -> Path:
        """Return the path to the loaded model file."""
        return self._model_path

    def get_supported_languages(self) -> dict[str, str]:
        """Get the dictionary of supported language codes and their names.

        Returns:
            Dict mapping language codes to human-readable names.
        """
        return self.SUPPORTED_LANGUAGES.copy()

    def is_supported(self, language_code: str) -> bool:
        """Check if a language code is supported."""
        return language_code.lower() in self.SUPPORTED_LANGUAGES

    def get_language_name(self, language_code: str) -> str | None:
        """Get the human-readable name for a language code.

        Args:
            language_code: ISO language code (e.g., "en", "de").

        Returns:
            Human-readable name if supported, None otherwise.
        """
        return self.SUPPORTED_LANGUAGES.get(language_code.lower())
