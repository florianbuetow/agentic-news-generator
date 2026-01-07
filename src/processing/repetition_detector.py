"""Repetition detection for hallucination analysis in transcripts."""

import re
import unicodedata

from src.config import Config
from src.models.hallucination_classifier import HallucinationClassifier


class RepetitionDetector:
    """Detect repetitive patterns in text using suffix array algorithm and ML classification."""

    _WS = re.compile(r"\s+")

    def __init__(self, min_k: int, min_repetitions: int, config: Config) -> None:
        """Initialize RepetitionDetector with ML classifier.

        Args:
            min_k: Minimum phrase length in words.
            min_repetitions: Minimum consecutive repetitions.
            config: Config object with hallucination detection settings.
        """
        self.min_k = min_k
        self.min_repetitions = min_repetitions
        self.classifier = HallucinationClassifier(config=config)

    def prepare(self, text: str) -> str:
        """Normalize whitespace and remove invisible Unicode characters.

        This minimal normalization helps distinguish real hallucinations (exact repetitions)
        from natural speech patterns (variations in capitalization/punctuation).
        Removes invisible formatting characters (RTL/LTR marks, zero-width spaces, etc.)
        """
        # Remove invisible Unicode formatting characters (category "C" = control/format)
        # but preserve common whitespace characters
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t ")

        # Normalize whitespace
        return self._WS.sub(" ", text).strip()

    def detect(self, text: str, min_k: int) -> list[list[int]]:
        """Detects repetitions and returns a list of [start, end, k].

        If multiple k exist for the same start point, only the largest k is kept.
        """
        cleaned_text = self.prepare(text)
        if not cleaned_text:
            return []

        words = cleaned_text.split()
        if len(words) < min_k * 2:
            return []

        # 1. Tokenize
        tokens, _ = self._tokenize(words)

        # 2. Build Suffix Array
        suffixes = list(range(len(tokens) - min_k + 1))
        suffixes.sort(key=lambda x: tokens[x:])

        # 3. Find all raw repetitions
        raw_results: list[list[int]] = []
        i = 0
        while i < len(suffixes) - 1:
            j = i + 1
            lcp_len = self._longest_common_prefix(tokens, suffixes[i], suffixes[j])

            if lcp_len >= min_k:
                match_indices = [suffixes[i], suffixes[j]]
                while j + 1 < len(suffixes):
                    if self._longest_common_prefix(tokens, suffixes[j], suffixes[j + 1]) >= lcp_len:
                        match_indices.append(suffixes[j + 1])
                        j += 1
                    else:
                        break

                # Store every occurrence of this specific phrase
                raw_results.extend([[start_idx, start_idx + lcp_len, lcp_len] for start_idx in match_indices])
                i = j
            else:
                i += 1

        # 4. Filter to keep only the largest k for overlapping starts
        # We sort by start index (ascending) and then by k (descending)
        raw_results.sort(key=lambda x: (x[0], -x[2]))

        final_results: list[list[int]] = []
        last_start = -1
        last_end = -1

        for start, end, k in raw_results:
            # If this starts at the same place as the last one,
            # we skip it because the previous one was longer (due to our sort)
            if start == last_start:
                continue

            # Optional: Overlap check. If this sequence is entirely contained
            # within the previous found sequence, we skip it.
            if start < last_end and end <= last_end:
                continue

            final_results.append([start, end, k])
            last_start = start
            last_end = end

        return final_results

    def is_consecutive_repetition(self, words: list[str], start: int, end: int) -> bool:
        """Check if a phrase repeats immediately after itself.

        Args:
            words: List of all words in the text.
            start: Start index of the first occurrence.
            end: End index of the first occurrence (exclusive).

        Returns:
            True if the phrase at [start:end] repeats at [end:end+(end-start)].
        """
        phrase_length = end - start

        # Check if there's enough room for a consecutive repetition
        if end + phrase_length > len(words):
            return False

        # Extract the phrase and the following words
        phrase = words[start:end]
        next_phrase = words[end : end + phrase_length]

        # Check if they're identical
        return phrase == next_phrase

    def count_consecutive_repetitions(self, words: list[str], start: int, end: int) -> int:
        """Count how many times a phrase repeats consecutively.

        Args:
            words: List of all words.
            start: Start index of first occurrence.
            end: End index of first occurrence (exclusive).

        Returns:
            Number of consecutive repetitions (minimum 2 for true repetitions).
        """
        phrase_length = end - start
        repetition_count = 1  # Count the first occurrence

        current_pos = end
        while current_pos + phrase_length <= len(words):
            next_phrase = words[current_pos : current_pos + phrase_length]
            original_phrase = words[start:end]

            if next_phrase == original_phrase:
                repetition_count += 1
                current_pos += phrase_length
            else:
                break

        return repetition_count

    def detect_hallucinations(self, text: str) -> list[list[int]]:
        """Detect hallucinations using ML classifier.

        Uses SVM classifier to determine if repetition pattern is a hallucination.

        Args:
            text: Text to analyze.

        Returns:
            List of [start, end, k, repetition_count] for qualifying patterns.
        """
        # First, get all repetitions using existing detect() method
        all_repetitions = self.detect(text, min_k=self.min_k)

        if not all_repetitions:
            return []

        # Prepare text to get word list
        cleaned_text = self.prepare(text)
        words = cleaned_text.split()

        # Filter using ML classifier
        hallucinations: list[list[int]] = []
        for start, end, k in all_repetitions:
            if self.is_consecutive_repetition(words, start, end):
                # Count how many times it repeats
                rep_count = self.count_consecutive_repetitions(words, start, end)

                # Use ML classifier to determine if this is a hallucination
                if rep_count >= self.min_repetitions:
                    is_hallucination = self.classifier.predict(rep_count, k)
                    if is_hallucination:
                        hallucinations.append([start, end, k, rep_count])
                        break  # Stop processing window after first detection

        return hallucinations

    def _tokenize(self, words: list[str]) -> tuple[list[int], dict[str, int]]:
        vocab: dict[str, int] = {}
        tokens: list[int] = []
        next_id = 0
        for word in words:
            if word not in vocab:
                vocab[word] = next_id
                next_id += 1
            tokens.append(vocab[word])
        return tokens, vocab

    def _longest_common_prefix(self, tokens: list[int], idx1: int, idx2: int) -> int:
        length = 0
        limit = min(len(tokens) - idx1, len(tokens) - idx2)
        while length < limit:
            if tokens[idx1 + length] == tokens[idx2 + length]:
                length += 1
            else:
                break
        return length
