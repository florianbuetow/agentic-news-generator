"""Type stubs for the onnxruntime APIs used by this project."""

from collections.abc import Sequence
from typing import Any

class SessionOptions:
    intra_op_num_threads: int
    def __init__(self) -> None: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def run(self, output_names: Sequence[str] | None, input_feed: dict[str, Any]) -> list[Any]: ...
