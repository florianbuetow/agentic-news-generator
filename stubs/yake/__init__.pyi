"""Stub package for yake."""

class KeywordExtractor:
    def __init__(
        self,
        *,
        lan: str = ...,
        n: int = ...,
        dedupLim: float = ...,
        dedupFunc: str = ...,
        windowsSize: int = ...,
        top: int = ...,
    ) -> None: ...
    def extract_keywords(self, text: str) -> list[tuple[str, float]]: ...
