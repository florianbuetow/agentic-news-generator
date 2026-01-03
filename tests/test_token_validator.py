"""Tests for token validation utility."""

import pytest

from src.agents.topic_segmentation.token_validator import (
    ContextWindowExceededError,
    count_tokens_from_messages,
    validate_token_usage,
)


class TestCountTokensFromMessages:
    """Tests for count_tokens_from_messages function."""

    def test_count_empty_messages(self) -> None:
        """Test token counting with empty messages list."""
        messages: list[dict[str, str]] = []
        token_count = count_tokens_from_messages(messages)
        # Should only include response priming overhead
        assert token_count == 3

    def test_count_single_message(self) -> None:
        """Test token counting with a single message."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        token_count = count_tokens_from_messages(messages)
        # Should be > 0 (content + overhead)
        assert token_count > 0

    def test_count_multiple_messages(self) -> None:
        """Test token counting with multiple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
        token_count = count_tokens_from_messages(messages)
        # Should be greater than single message
        assert token_count > 20  # Reasonable minimum for this content

    def test_count_long_content(self) -> None:
        """Test token counting with long content."""
        long_text = "Hello " * 1000  # Repeat to create longer content
        messages = [{"role": "user", "content": long_text}]
        token_count = count_tokens_from_messages(messages)
        # Should be proportional to content length
        assert token_count > 1000  # Should have many tokens for long content

    def test_different_encoding(self) -> None:
        """Test that encoding parameter works."""
        messages = [{"role": "user", "content": "Test message"}]
        # Just verify it doesn't raise an error
        token_count = count_tokens_from_messages(messages, encoding_name="cl100k_base")
        assert token_count > 0


class TestValidateTokenUsage:
    """Tests for validate_token_usage function."""

    def test_within_threshold(self) -> None:
        """Test that validation passes when within threshold."""
        messages = [{"role": "user", "content": "Short message"}]
        context_window = 100000
        threshold = 90

        # Should not raise exception
        token_count = validate_token_usage(messages, context_window, threshold)
        assert token_count > 0
        assert token_count < (context_window * 0.9)

    def test_exceeds_threshold(self) -> None:
        """Test that validation raises error when exceeding threshold."""
        # Create a large message
        large_content = "test " * 50000
        messages = [{"role": "user", "content": large_content}]

        # First, get the actual token count
        actual_count = count_tokens_from_messages(messages)

        # Set context window and threshold such that we exceed it
        # For example, if actual count is 50000, set window to 50000 and threshold to 90%
        # This means threshold is 45000, but actual is 50000, so it should exceed
        context_window = actual_count
        threshold = 90  # 90% of actual_count < actual_count

        with pytest.raises(ContextWindowExceededError) as exc_info:
            validate_token_usage(messages, context_window, threshold)

        # Verify exception attributes
        assert exc_info.value.context_window == context_window
        assert exc_info.value.threshold == threshold
        assert exc_info.value.token_count > exc_info.value.threshold_tokens

    def test_exactly_at_threshold(self) -> None:
        """Test validation with token count exactly at threshold."""
        # This is hard to test precisely, but we can verify the boundary behavior
        messages = [{"role": "user", "content": "x" * 100}]

        # First, get actual token count
        actual_count = count_tokens_from_messages(messages)

        # Set context window such that threshold is exactly the token count
        context_window = actual_count
        threshold = 100

        # Should not raise (token_count == threshold_tokens)
        token_count = validate_token_usage(messages, context_window, threshold)
        assert token_count == actual_count

    def test_one_token_over_threshold(self) -> None:
        """Test that even one token over threshold raises error."""
        messages = [{"role": "user", "content": "x" * 100}]

        # Get actual token count
        actual_count = count_tokens_from_messages(messages)

        # Set context window such that threshold is one less than token count
        threshold_tokens = actual_count - 1
        context_window = threshold_tokens  # 100% threshold
        threshold = 100

        with pytest.raises(ContextWindowExceededError):
            validate_token_usage(messages, context_window, threshold)

    def test_zero_threshold(self) -> None:
        """Test validation with 0% threshold."""
        messages = [{"role": "user", "content": "Any message"}]
        context_window = 100000
        threshold = 0

        # Should raise since any tokens > 0% threshold
        with pytest.raises(ContextWindowExceededError):
            validate_token_usage(messages, context_window, threshold)

    def test_hundred_percent_threshold(self) -> None:
        """Test validation with 100% threshold."""
        messages = [{"role": "user", "content": "Test message"}]
        context_window = 100000
        threshold = 100

        # Should pass for reasonable message sizes
        token_count = validate_token_usage(messages, context_window, threshold)
        assert token_count > 0
        assert token_count <= context_window

    def test_small_context_window(self) -> None:
        """Test with very small context window."""
        messages = [{"role": "user", "content": "Hello world"}]
        context_window = 10  # Very small
        threshold = 90

        # Should raise since message likely exceeds small window
        with pytest.raises(ContextWindowExceededError):
            validate_token_usage(messages, context_window, threshold)


class TestContextWindowExceededError:
    """Tests for ContextWindowExceededError exception."""

    def test_error_message_format(self) -> None:
        """Test that error message is properly formatted."""
        error = ContextWindowExceededError(
            token_count=10000,
            context_window=100000,
            threshold=90,
            threshold_tokens=90000,
        )

        error_msg = str(error)

        # Check that all key information is in the message
        assert "10,000" in error_msg  # token_count with formatting
        assert "90%" in error_msg  # threshold percentage
        assert "90,000" in error_msg  # threshold_tokens with formatting
        assert "100,000" in error_msg  # context_window with formatting
        assert "10.0%" in error_msg  # current usage percentage

    def test_error_attributes(self) -> None:
        """Test that error stores all attributes correctly."""
        token_count = 95000
        context_window = 100000
        threshold = 90
        threshold_tokens = 90000

        error = ContextWindowExceededError(
            token_count=token_count,
            context_window=context_window,
            threshold=threshold,
            threshold_tokens=threshold_tokens,
        )

        assert error.token_count == token_count
        assert error.context_window == context_window
        assert error.threshold == threshold
        assert error.threshold_tokens == threshold_tokens

    def test_error_inheritance(self) -> None:
        """Test that error inherits from ValueError."""
        error = ContextWindowExceededError(
            token_count=100,
            context_window=90,
            threshold=90,
            threshold_tokens=81,
        )

        assert isinstance(error, ValueError)

    def test_zero_context_window_no_division_error(self) -> None:
        """Test that zero context window doesn't cause division by zero."""
        error = ContextWindowExceededError(
            token_count=100,
            context_window=0,
            threshold=90,
            threshold_tokens=0,
        )

        # Should not raise, percentage should be 0
        error_msg = str(error)
        assert "0.0%" in error_msg
