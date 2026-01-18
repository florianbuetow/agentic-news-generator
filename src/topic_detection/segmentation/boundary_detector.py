"""Boundary detector for the topic segmentation pipeline.

Identifies topic boundaries from similarity scores using
various threshold methods.
"""

import numpy as np
from numpy.typing import NDArray

from src.topic_detection.segmentation.data_types import BoundaryData, SimilarityData


class BoundaryDetector:
    """Detects topic boundaries from similarity scores.

    Supports three detection methods:
    - "relative": Depth scoring - finds valleys where similarity drops
                  significantly relative to surrounding peaks.
    - "absolute": Fixed threshold - marks boundaries where similarity
                  falls below a fixed value.
    - "percentile": Marks boundaries where similarity is in the bottom
                   N percentile of all scores.

    Args:
        method: Detection method ("relative", "absolute", or "percentile").
        threshold: Threshold parameter (interpretation depends on method).
                  - relative: minimum depth score (0.0-1.0, typical: 0.1-0.5)
                  - absolute: minimum similarity (0.0-1.0, typical: 0.3-0.7)
                  - percentile: bottom percentile (0-100, typical: 10-30)

    Example:
        >>> detector = BoundaryDetector(method="relative", threshold=0.4)
        >>> boundary_data = detector.detect(similarity_data)
        >>> print(boundary_data.boundary_indices)  # [150, 340, 520]
    """

    VALID_METHODS: set[str] = {"relative", "absolute", "percentile"}

    def __init__(self, method: str, threshold: float) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")

        self.method = method
        self.threshold = threshold

    def detect(self, similarity_data: SimilarityData) -> BoundaryData:
        """Detect topic boundaries from similarity scores.

        Args:
            similarity_data: SimilarityData from the SimilarityCalculator.

        Returns:
            BoundaryData with boundary positions and depth scores.
        """
        scores = similarity_data.scores
        positions = similarity_data.comparison_positions

        if len(scores) == 0:
            return BoundaryData(boundary_indices=[], depths=[])

        if self.method == "relative":
            boundaries, depths = self._detect_relative(scores, positions)
        elif self.method == "absolute":
            boundaries, depths = self._detect_absolute(scores, positions)
        else:  # percentile
            boundaries, depths = self._detect_percentile(scores, positions)

        return BoundaryData(boundary_indices=boundaries, depths=depths)

    def _detect_relative(self, scores: NDArray[np.float64], positions: list[int]) -> tuple[list[int], list[float]]:
        """Detect boundaries using depth scoring at local minima.

        A boundary is placed at local minima where the "depth" exceeds
        the threshold. Depth is calculated as the average drop from
        the nearest peaks on both sides.

        Returns:
            Tuple of (boundary_indices, depths)
        """
        boundaries: list[int] = []
        depths: list[float] = []

        # Find local minima
        minima_indices = self._find_local_minima(scores)

        for idx in minima_indices:
            depth = self._compute_depth(scores, idx)

            if depth > self.threshold:
                boundaries.append(positions[idx])
                depths.append(depth)

        return boundaries, depths

    def _detect_absolute(self, scores: NDArray[np.float64], positions: list[int]) -> tuple[list[int], list[float]]:
        """Detect boundaries where similarity falls below absolute threshold.

        Returns:
            Tuple of (boundary_indices, depths)
        """
        boundaries: list[int] = []
        depths: list[float] = []

        for i, score in enumerate(scores):
            if score < self.threshold:
                boundaries.append(positions[i])
                # Use (threshold - score) as a pseudo-depth measure
                depths.append(self.threshold - float(score))

        return boundaries, depths

    def _detect_percentile(self, scores: NDArray[np.float64], positions: list[int]) -> tuple[list[int], list[float]]:
        """Detect boundaries where similarity is in bottom N percentile.

        Returns:
            Tuple of (boundary_indices, depths)
        """
        boundaries: list[int] = []
        depths: list[float] = []

        cutoff = float(np.percentile(scores, self.threshold))

        for i, score in enumerate(scores):
            if score <= cutoff:
                boundaries.append(positions[i])
                depths.append(cutoff - float(score))

        return boundaries, depths

    def _find_local_minima(self, scores: NDArray[np.float64]) -> list[int]:
        """Find indices of local minima in the scores array.

        A local minimum is a point lower than both its neighbors.
        Endpoints are included if they're lower than their single neighbor.

        Returns:
            List of indices that are local minima.
        """
        if len(scores) == 0:
            return []

        if len(scores) == 1:
            return [0]

        minima: list[int] = []

        # Check first element
        if scores[0] < scores[1]:
            minima.append(0)

        # Check middle elements
        minima.extend(i for i in range(1, len(scores) - 1) if scores[i] < scores[i - 1] and scores[i] < scores[i + 1])

        # Check last element
        if scores[-1] < scores[-2]:
            minima.append(len(scores) - 1)

        return minima

    def _compute_depth(self, scores: NDArray[np.float64], idx: int) -> float:
        """Compute the depth score for a local minimum.

        Depth = average of (left_peak - valley) and (right_peak - valley)

        This measures how much the similarity "dips" at this point
        relative to the surrounding context.

        Args:
            scores: Array of similarity scores.
            idx: Index of the local minimum.

        Returns:
            Depth score (higher = more significant boundary).
        """
        valley = float(scores[idx])

        # Find left peak (maximum from start to idx)
        left_peak = float(np.max(scores[: idx + 1])) if idx > 0 else valley

        # Find right peak (maximum from idx to end)
        right_peak = float(np.max(scores[idx:])) if idx < len(scores) - 1 else valley

        # Depth is average drop from both sides
        left_depth = left_peak - valley
        right_depth = right_peak - valley

        return (left_depth + right_depth) / 2
