"""Stub for sklearn.feature_extraction.text."""

from collections.abc import Iterable

class TfidfVectorizer:
    def __init__(
        self,
        *,
        input: str,
        encoding: str,
        decode_error: str,
        strip_accents: str | None,
        lowercase: bool,
        preprocessor: object,
        tokenizer: object,
        analyzer: str,
        stop_words: str | None,
        token_pattern: str,
        ngram_range: tuple[int, int],
        max_df: float,
        min_df: int,
        max_features: int,
        vocabulary: object,
        binary: bool,
        dtype: object,
        norm: str,
        use_idf: bool,
        smooth_idf: bool,
        sublinear_tf: bool,
    ) -> None: ...
    def fit(self, raw_documents: Iterable[str]) -> TfidfVectorizer: ...
    def transform(self, raw_documents: Iterable[str]) -> SparseMatrix: ...
    def get_feature_names_out(self) -> list[str]: ...

class SparseRow:
    nnz: int
    indices: list[int]
    data: list[float]

class SparseMatrix:
    def __getitem__(self, idx: int) -> SparseRow: ...
