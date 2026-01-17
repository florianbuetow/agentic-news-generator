"""Segment assembler for the topic segmentation pipeline.

Converts boundary positions back into text segments,
enforcing minimum segment size constraints.

Note: "token" in this module refers to word-level tokens (words and punctuation),
not BPE/subword tokens.
"""

from src.topic_detection.segmentation.data_types import BoundaryData, Segment
from src.topic_detection.segmentation.tokenizer import Tokenizer


class SegmentAssembler:
    """Assembles final text segments from tokens and boundary positions.

    Takes the detected boundaries and original tokens, splits the text
    at boundary positions, and reconstructs readable text segments.
    Enforces minimum segment length by merging too-short segments.

    Args:
        tokenizer: Tokenizer instance for converting tokens back to text.
        min_segment_tokens: Minimum number of tokens per segment.
                           Segments smaller than this are merged with neighbors.

    Example:
        >>> assembler = SegmentAssembler(tokenizer, min_segment_tokens=100)
        >>> segments = assembler.assemble(tokens, boundary_data)
        >>> for seg in segments:
        ...     print(f"Segment ({seg.token_count} tokens): {seg.text[:50]}...")
    """

    def __init__(self, tokenizer: Tokenizer, min_segment_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.min_segment_tokens = min_segment_tokens

    def assemble(self, tokens: list[str], boundary_data: BoundaryData) -> list[Segment]:
        """Assemble segments from tokens and boundaries.

        Args:
            tokens: Original token list.
            boundary_data: BoundaryData from the BoundaryDetector.

        Returns:
            List of Segment objects in document order.
        """
        if not tokens:
            return []

        # Get valid boundaries (within token range)
        boundaries = [b for b in boundary_data.boundary_indices if 0 < b < len(tokens)]

        # Sort boundaries
        boundaries = sorted(set(boundaries))

        # Filter boundaries to enforce minimum segment size
        boundaries = self._enforce_minimum_size(boundaries, len(tokens))

        # Create segments
        segments = self._create_segments(tokens, boundaries)

        return segments

    def _enforce_minimum_size(self, boundaries: list[int], total_tokens: int) -> list[int]:
        """Remove boundaries that would create too-small segments.

        Iteratively removes boundaries until all segments meet
        the minimum size requirement.

        Args:
            boundaries: Sorted list of boundary positions.
            total_tokens: Total number of tokens.

        Returns:
            Filtered list of boundary positions.
        """
        if not boundaries:
            return []

        # Create segment sizes from boundaries
        # Segments are: [0, b1), [b1, b2), ..., [bn, total)
        result = boundaries.copy()

        changed = True
        while changed:
            changed = False

            # Calculate segment sizes
            sizes = self._calculate_segment_sizes(result, total_tokens)

            # Find first segment that's too small
            for i, size in enumerate(sizes):
                if size < self.min_segment_tokens:
                    # Merge with smaller neighbor by removing a boundary
                    if i == 0:
                        # First segment too small - remove first boundary
                        if result:
                            result.pop(0)
                            changed = True
                            break
                    elif i == len(sizes) - 1:
                        # Last segment too small - remove last boundary
                        if result:
                            result.pop()
                            changed = True
                            break
                    else:
                        # Middle segment - merge with smaller neighbor
                        # Remove the boundary that creates the smaller merged segment
                        left_merge_size = sizes[i - 1] + sizes[i]
                        right_merge_size = sizes[i] + sizes[i + 1]

                        if left_merge_size <= right_merge_size:
                            # Remove boundary at index i-1 (merge with left)
                            result.pop(i - 1)
                        else:
                            # Remove boundary at index i (merge with right)
                            result.pop(i)

                        changed = True
                        break

        return result

    def _calculate_segment_sizes(self, boundaries: list[int], total_tokens: int) -> list[int]:
        """Calculate the size of each segment given boundary positions.

        Args:
            boundaries: Sorted list of boundary positions.
            total_tokens: Total number of tokens.

        Returns:
            List of segment sizes.
        """
        if not boundaries:
            return [total_tokens]

        sizes: list[int] = []

        # First segment: from 0 to first boundary
        sizes.append(boundaries[0])

        # Middle segments: between consecutive boundaries
        sizes.extend(boundaries[i] - boundaries[i - 1] for i in range(1, len(boundaries)))

        # Last segment: from last boundary to end
        sizes.append(total_tokens - boundaries[-1])

        return sizes

    def _create_segments(self, tokens: list[str], boundaries: list[int]) -> list[Segment]:
        """Create Segment objects from tokens and boundaries.

        Args:
            tokens: Original token list.
            boundaries: Sorted, filtered boundary positions.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        # Add 0 at start and len(tokens) at end for easier iteration
        split_points = [0] + boundaries + [len(tokens)]

        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]

            segment_tokens = tokens[start:end]
            text = self.tokenizer.detokenize(segment_tokens)

            segments.append(
                Segment(
                    text=text,
                    start_token=start,
                    end_token=end,
                    token_count=end - start,
                )
            )

        return segments
