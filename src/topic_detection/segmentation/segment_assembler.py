"""Segment assembler for the topic segmentation pipeline.

Converts boundary positions back into text segments.
"""

from src.topic_detection.segmentation.data_types import BoundaryData, Segment
from src.topic_detection.segmentation.tokenizer import Tokenizer


class SegmentAssembler:
    """Assembles final text segments from words and boundary positions.

    Takes the detected boundaries and original words, splits the text
    at boundary positions, and reconstructs readable text segments.

    Args:
        tokenizer: Tokenizer instance for converting words back to text.

    Example:
        >>> assembler = SegmentAssembler(tokenizer)
        >>> segments = assembler.assemble(words, boundary_data)
        >>> for seg in segments:
        ...     print(f"Segment ({seg.word_count} words): {seg.text[:50]}...")
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def assemble(self, words: list[str], boundary_data: BoundaryData) -> list[Segment]:
        """Assemble segments from words and boundaries.

        Args:
            words: Original word list.
            boundary_data: BoundaryData from the BoundaryDetector.

        Returns:
            List of Segment objects in document order.
        """
        if not words:
            return []

        # Get valid boundaries (within word range)
        boundaries = [b for b in boundary_data.boundary_indices if 0 < b < len(words)]

        # Sort boundaries and remove duplicates
        boundaries = sorted(set(boundaries))

        # Create segments
        segments = self._create_segments(words, boundaries)

        return segments

    def _create_segments(self, words: list[str], boundaries: list[int]) -> list[Segment]:
        """Create Segment objects from words and boundaries.

        Args:
            words: Original word list.
            boundaries: Sorted boundary positions.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        # Add 0 at start and len(words) at end for easier iteration
        split_points = [0] + boundaries + [len(words)]

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]

            segment_words = words[start:end]
            text = self.tokenizer.detokenize(segment_words)

            segments.append(
                Segment(
                    text=text,
                    start_word=start,
                    end_word=end,
                    word_count=end - start,
                )
            )

        return segments
