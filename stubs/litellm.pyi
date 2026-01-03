"""Type stubs for litellm library."""

class ModelResponse:
    """Response from LLM completion."""

    choices: list[Choice]

class Choice:
    """Individual choice in completion response."""

    message: Message

class Message:
    """Message in completion response."""

    content: str | None

def completion(
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int | None = None,
    stream: bool | None = None,
) -> ModelResponse: ...
