"""Token usage validation for LLM API calls."""

from typing import Any

import tiktoken


class ContextWindowExceededError(ValueError):
    """Raised when token usage exceeds context window threshold."""

    def __init__(self, token_count: int, context_window: int, threshold: int, threshold_tokens: int) -> None:
        """Initialize the exception with token usage details.

        Args:
            token_count: Actual token count from messages.
            context_window: Maximum context window size in tokens.
            threshold: Percentage threshold (0-100) that was configured.
            threshold_tokens: Calculated threshold in absolute tokens.
        """
        self.token_count = token_count
        self.context_window = context_window
        self.threshold = threshold
        self.threshold_tokens = threshold_tokens

        percentage = (token_count / context_window * 100) if context_window > 0 else 0

        super().__init__(
            f"Token usage ({token_count:,} tokens) exceeds {threshold}% threshold "
            f"({threshold_tokens:,} tokens) of context window ({context_window:,} tokens). "
            f"Current usage: {percentage:.1f}%"
        )


def count_tokens_from_messages(messages: list[dict[str, Any]], encoding_name: str = "o200k_base") -> int:
    """Count tokens in a list of messages using tiktoken.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        encoding_name: Tiktoken encoding to use (default: o200k_base for GPT-4o).

    Returns:
        Total token count for the messages.
    """
    encoding = tiktoken.get_encoding(encoding_name)

    # Count tokens for each message
    # Format: <|im_start|>{role}\n{content}<|im_end|>\n
    # Approximate overhead: 4 tokens per message + content tokens
    total_tokens = 0

    for message in messages:
        # Message formatting overhead (role tags, newlines, etc.)
        total_tokens += 4  # Approximate overhead per message

        # Count tokens in role
        role = message.get("role", "")
        total_tokens += len(encoding.encode(role))

        # Count tokens in content
        content = message.get("content", "")
        total_tokens += len(encoding.encode(content))

    # Add overhead for response priming
    total_tokens += 3  # Approximate overhead for assistant response priming

    return total_tokens


def validate_token_usage(
    messages: list[dict[str, Any]],
    context_window: int,
    threshold: int,
    encoding_name: str = "o200k_base",
) -> int:
    """Validate that token usage is within acceptable limits.

    Args:
        messages: List of message dicts to send to LLM.
        context_window: Maximum context window size in tokens.
        threshold: Percentage threshold (0-100) for acceptable usage.
        encoding_name: Tiktoken encoding to use (default: o200k_base).

    Returns:
        Total token count.

    Raises:
        ContextWindowExceededError: If token usage exceeds threshold.
    """
    # Count tokens
    token_count = count_tokens_from_messages(messages, encoding_name)

    # Calculate threshold in tokens
    threshold_tokens = int(context_window * (threshold / 100.0))

    # Check if exceeded
    if token_count > threshold_tokens:
        raise ContextWindowExceededError(
            token_count=token_count,
            context_window=context_window,
            threshold=threshold,
            threshold_tokens=threshold_tokens,
        )

    return token_count
