"""Tokenizer for the topic segmentation pipeline.

Handles conversion between raw text and token lists.
Uses simple whitespace/punctuation tokenization - no external dependencies.
"""

import re


class Tokenizer:
    """Simple word tokenizer that splits on whitespace and punctuation.

    Preserves punctuation as separate tokens to allow accurate reconstruction.
    No external dependencies (no NLTK, spaCy, etc.).

    Example:
        >>> t = Tokenizer()
        >>> t.tokenize("Hello, world!")
        ['Hello', ',', 'world', '!']
        >>> t.detokenize(['Hello', ',', 'world', '!'])
        'Hello, world!'
    """

    # Pattern matches either:
    # - One or more word characters (letters, digits, underscore)
    # - Or a single non-whitespace, non-word character (punctuation)
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    # Punctuation that should attach to the preceding token (no space before)
    ATTACH_LEFT: set[str] = {",", ".", "!", "?", ";", ":", "'", '"', ")", "]", "}", "%"}

    # Punctuation that should attach to the following token (no space after)
    ATTACH_RIGHT: set[str] = {"(", "[", "{", '"', "'", "$"}

    def tokenize(self, text: str) -> list[str]:
        """Convert text to a list of tokens.

        Args:
            text: Raw input text string.

        Returns:
            List of token strings.
        """
        if not text:
            return []

        tokens = self.TOKEN_PATTERN.findall(text)
        return tokens

    def detokenize(self, tokens: list[str]) -> str:
        """Reconstruct text from tokens.

        Attempts to produce readable text by handling punctuation spacing.
        Not guaranteed to be identical to original (whitespace may differ).

        Args:
            tokens: List of token strings.

        Returns:
            Reconstructed text string.
        """
        if not tokens:
            return ""

        result = [tokens[0]]

        for token in tokens[1:]:
            # Decide whether to add a space before this token
            if token in self.ATTACH_LEFT:
                # Punctuation that attaches to previous word
                result.append(token)
            elif result and result[-1] in self.ATTACH_RIGHT:
                # Previous token wants to attach to this one
                result.append(token)
            else:
                # Normal case: add space
                result.append(" ")
                result.append(token)

        return "".join(result)
