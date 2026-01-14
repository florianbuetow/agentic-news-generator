"""Type stubs for fasttext library.

FastText is a library for efficient text classification and representation learning.
This stub file provides minimal type information for the language_detector.py usage.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

class FastText:
    """FastText module namespace."""

    # Module-level attribute for suppressing warnings
    eprint: Callable[[str], None]

    class _FastText:
        """FastText model class (internal)."""

        def predict(
            self,
            text: str,
            k: int = 1,
            threshold: float = 0.0,
            on_unicode_error: str = "strict",
        ) -> tuple[tuple[str, ...], npt.NDArray[np.float32]]:
            """Predict language labels for input text.

            Args:
                text: Input text to classify
                k: Number of top predictions to return
                threshold: Minimum confidence threshold
                on_unicode_error: How to handle unicode errors

            Returns:
                Tuple of (labels, scores) where:
                - labels: tuple of label strings (e.g., ('__label__en',))
                - scores: numpy array of confidence scores
            """
            ...

def load_model(path: str) -> FastText._FastText:
    """Load a FastText model from disk.

    Args:
        path: Path to the .ftz or .bin model file

    Returns:
        FastText model instance
    """
    ...
