import re


class RepetitionDetector:
    _NON_ALPHA = re.compile(r"[^a-z\s]+")
    _WS = re.compile(r"\s+")

    def prepare(self, text: str) -> str:
        text = text.lower()
        text = self._NON_ALPHA.sub(" ", text)
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

    def detect_hallucinations(self, text: str, min_k: int = 3) -> list[list[int]]:
        """Detect consecutive repetitions (hallucinations) in text.

        This method finds phrases that repeat immediately back-to-back,
        which is a characteristic of hallucinated text from speech models.

        Args:
            text: Text to analyze.
            min_k: Minimum repetition length in words.

        Returns:
            List of [start, end, k] for consecutive repetitions only.
        """
        # First, get all repetitions using existing detect() method
        all_repetitions = self.detect(text, min_k=min_k)

        if not all_repetitions:
            return []

        # Prepare text to get word list
        cleaned_text = self.prepare(text)
        words = cleaned_text.split()

        # Filter to keep only consecutive repetitions
        hallucinations = []
        for start, end, k in all_repetitions:
            if self._is_consecutive_repetition(words, start, end):
                hallucinations.append([start, end, k])

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
    sample = "apple banana cherry date apple banana cherry fig apple banana cherry date"

    # "apple banana cherry date" is k=4
    # "apple banana cherry" is k=3
    # The algorithm will prioritize the k=4 version.
    results = detector.detect(sample, min_k=3)

    print("Detected [start, end, k]:")
    for res in results:
        print(res)
