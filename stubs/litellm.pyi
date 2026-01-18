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
    response_format: dict[str, object] | None = None,
) -> ModelResponse: ...

class EmbeddingData:
    """Individual embedding data item."""

    embedding: list[float]

    def __getitem__(self, key: str) -> list[float]: ...

class EmbeddingResponse:
    """Response from embedding API."""

    data: list[EmbeddingData]

def embedding(
    model: str,
    input: list[str],
    api_base: str | None = None,
    api_key: str | None = None,
) -> EmbeddingResponse: ...
