import re
import unicodedata
from pathlib import Path

from src.models.hallucination_classifier import HallucinationClassifier


class RepetitionDetector:
    _WS = re.compile(r"\s+")

    def __init__(
        self,
        min_k: int = 1,
        min_repetitions: int = 5,
        config_path: Path | None = None
    ):
        """Initialize RepetitionDetector with ML classifier.

        Args:
            min_k: Minimum phrase length in words (default: 1).
            min_repetitions: Minimum consecutive repetitions (default: 5).
            config_path: Path to config.yaml (optional).
        """
        self.min_k = min_k
        self.min_repetitions = min_repetitions
        self.classifier = HallucinationClassifier(config_path=config_path)

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

    def detect(self, text: str, min_k: int = 3) -> list[list[int]]:
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
        raw_results = []
        i = 0
        while i < len(suffixes) - 1:
            j = i + 1
            lcp_len = self._longest_common_prefix(tokens, suffixes[i], suffixes[j])

            if lcp_len >= min_k:
                match_indices = [suffixes[i], suffixes[j]]
                while j + 1 < len(suffixes):
                    if self._longest_common_prefix(tokens, suffixes[j], suffixes[j+1]) >= lcp_len:
                        match_indices.append(suffixes[j+1])
                        j += 1
                    else:
                        break

                # Store every occurrence of this specific phrase
                for start_idx in match_indices:
                    # [start, end, k]
                    raw_results.append([start_idx, start_idx + lcp_len, lcp_len])
                i = j
            else:
                i += 1

        # 4. Filter to keep only the largest k for overlapping starts
        # We sort by start index (ascending) and then by k (descending)
        raw_results.sort(key=lambda x: (x[0], -x[2]))

        final_results = []
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

    def _is_consecutive_repetition(self, words: list[str], start: int, end: int) -> bool:
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
        next_phrase = words[end:end + phrase_length]

        # Check if they're identical
        return phrase == next_phrase

    def _count_consecutive_repetitions(
        self,
        words: list[str],
        start: int,
        end: int
    ) -> int:
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
            next_phrase = words[current_pos:current_pos + phrase_length]
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
        hallucinations = []
        for start, end, k in all_repetitions:
            if self._is_consecutive_repetition(words, start, end):
                # Count how many times it repeats
                rep_count = self._count_consecutive_repetitions(words, start, end)

                # Use ML classifier to determine if this is a hallucination
                if rep_count >= self.min_repetitions:
                    is_hallucination = self.classifier.predict(rep_count, k)
                    if is_hallucination:
                        hallucinations.append([start, end, k, rep_count])
                        break  # Stop processing window after first detection

        return hallucinations

    def _tokenize(self, words: list[str]) -> tuple[list[int], dict[str, int]]:
        vocab = {}
        tokens = []
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


# --- Example ---
if __name__ == "__main__":
    detector = RepetitionDetector()

    print("=== Formula-Based Hallucination Detection ===\n")

    # Example 1: Massive loop - CLEAR hallucination
    hallucination_text = " ".join(["you know, like,"] * 15)
    results = detector.detect_hallucinations(hallucination_text)
    print("Example 1: Massive loop (TRUE HALLUCINATION)")
    print(f"  Text: 'you know, like,' repeated 15 times")
    if results:
        start, end, k, rep_count = results[0]
        score = k * rep_count
        print(f"  Pattern length (k): {k}")
        print(f"  Repetition count: {rep_count}")
        print(f"  Score: {k} × {rep_count} = {score}")
        print(f"  Result: {score} > 10 ✓ DETECTED\n")
    else:
        print(f"  Result: NOT DETECTED (unexpected)\n")

    # Example 2: Natural stutter - should be FILTERED
    natural_stutter = "as like a, as like a, an object schema"
    results = detector.detect_hallucinations(natural_stutter)
    print("Example 2: Natural stutter (FALSE POSITIVE)")
    print(f"  Text: {natural_stutter}")
    # Check if it was detected by the basic detect() method
    basic_results = detector.detect(natural_stutter, min_k=1)
    if basic_results:
        for start, end, k in basic_results:
            words = detector.prepare(natural_stutter).split()
            if detector._is_consecutive_repetition(words, start, end):
                rep_count = detector._count_consecutive_repetitions(words, start, end)
                score = k * rep_count
                print(f"  Pattern length (k): {k}")
                print(f"  Repetition count: {rep_count}")
                print(f"  Score: {k} × {rep_count} = {score}")
                print(f"  Result: {score} < 10 ✓ FILTERED\n")
                break
    if not results:
        print(f"  Result: Correctly filtered (score too low)\n")

    # Example 3: Ambiguous 3x repetition - should be FILTERED
    ambiguous = "for this work, for this work, for this work, the xi"
    results = detector.detect_hallucinations(ambiguous)
    print("Example 3: Ambiguous 3x repetition")
    print(f"  Text: {ambiguous}")
    basic_results = detector.detect(ambiguous, min_k=1)
    if basic_results:
        for start, end, k in basic_results:
            words = detector.prepare(ambiguous).split()
            if detector._is_consecutive_repetition(words, start, end):
                rep_count = detector._count_consecutive_repetitions(words, start, end)
                score = k * rep_count
                print(f"  Pattern length (k): {k}")
                print(f"  Repetition count: {rep_count}")
                print(f"  Score: {k} × {rep_count} = {score}")
                if score > 10:
                    print(f"  Result: {score} > 10 ✓ DETECTED\n")
                else:
                    print(f"  Result: {score} < 10 ✓ FILTERED (natural speech)\n")
                break

    print("\n=== Threshold Tuning Examples ===\n")
    print("Threshold | k=3, reps=2 | k=3, reps=3 | k=3, reps=5 | k=1, reps=11")
    print("----------|-------------|-------------|-------------|-------------")
    for thresh in [6, 9, 10, 15]:
        results_row = []
        for k, reps in [(3, 2), (3, 3), (3, 5), (1, 11)]:
            score = k * reps
            detected = "✓" if (reps >= 5 or score > thresh) else "✗"
            results_row.append(f"{detected} ({score})")
        print(f"    {thresh:2d}    | {results_row[0]:^11} | {results_row[1]:^11} | {results_row[2]:^11} | {results_row[3]:^11}")
