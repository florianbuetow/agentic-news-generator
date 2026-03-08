"""Stub for keybert._model."""

class KeyBERT:
    def __init__(self, model: object = ...) -> None: ...
    def extract_keywords(
        self,
        docs: str,
        *,
        keyphrase_ngram_range: tuple[int, int] = ...,
        stop_words: str | None = ...,
        top_n: int = ...,
        use_mmr: bool = ...,
        diversity: float = ...,
    ) -> list[tuple[str, float]]: ...
