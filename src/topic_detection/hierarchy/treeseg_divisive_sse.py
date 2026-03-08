"""TreeSeg-style deterministic hierarchical segmentation via divisive SSE splitting.

This is an unsupervised, deterministic algorithm that builds a binary tree over a
time-ordered sequence of embedding vectors. Each split chooses the boundary that
maximizes the reduction in within-segment sum of squared errors (SSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

from src.topic_detection.hierarchy.data_types import TopicTreeNode


@dataclass(frozen=True)
class SplitCandidate:
    """Best split candidate for a segment."""

    split_idx: int
    gain: float


class TreeSegDivisiveSSETopicSegmenter:
    """Build a deterministic hierarchical topic tree from a 1D embedding sequence.

    Args:
        max_depth: Maximum tree depth (root depth is 0).
        min_leaf_entries: Minimum number of entries in each leaf node.
        min_leaf_seconds: Minimum duration (seconds) in each leaf node.
        min_gain: Minimum SSE gain required to accept a split.
    """

    def __init__(
        self,
        *,
        max_depth: int,
        min_leaf_entries: int,
        min_leaf_seconds: float,
        min_gain: float,
    ) -> None:
        self._max_depth = max_depth
        self._min_leaf_entries = min_leaf_entries
        self._min_leaf_seconds = min_leaf_seconds
        self._min_gain = min_gain

    def build_tree(
        self,
        *,
        embeddings: NDArray[np.float32],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> TopicTreeNode:
        """Build a hierarchical topic tree.

        Args:
            embeddings: Array of shape [n_entries, embedding_dim].
            entry_start_seconds: Start time (seconds) per entry index.
            entry_end_seconds: End time (seconds) per entry index.

        Returns:
            Root TopicTreeNode.

        Raises:
            ValueError: If inputs are inconsistent.
        """
        n_entries = self._validate_inputs(
            embeddings=embeddings,
            entry_start_seconds=entry_start_seconds,
            entry_end_seconds=entry_end_seconds,
        )
        prefix_sum, prefix_sq = self._build_prefix_sums(embeddings)

        root = TopicTreeNode(
            start_entry_idx=0,
            end_entry_idx_exclusive=n_entries,
            depth=0,
            parent=None,
        )

        heap, counter = self._initialize_heap(
            root=root,
            prefix_sum=prefix_sum,
            prefix_sq=prefix_sq,
            entry_start_seconds=entry_start_seconds,
            entry_end_seconds=entry_end_seconds,
        )
        self._apply_splits(
            heap=heap,
            counter=counter,
            prefix_sum=prefix_sum,
            prefix_sq=prefix_sq,
            entry_start_seconds=entry_start_seconds,
            entry_end_seconds=entry_end_seconds,
        )

        return root

    def _validate_inputs(
        self,
        *,
        embeddings: NDArray[np.float32],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> int:
        """Validate inputs and return number of entries."""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape: {embeddings.shape}")

        n_entries = embeddings.shape[0]
        if n_entries == 0:
            raise ValueError("Cannot build a topic tree with 0 entries")

        if len(entry_start_seconds) != n_entries or len(entry_end_seconds) != n_entries:
            raise ValueError(
                "entry_start_seconds and entry_end_seconds must match embeddings length "
                f"(expected {n_entries}, got {len(entry_start_seconds)} and {len(entry_end_seconds)})"
            )

        if self._max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self._min_leaf_entries <= 0:
            raise ValueError("min_leaf_entries must be > 0")
        if self._min_leaf_seconds < 0:
            raise ValueError("min_leaf_seconds must be >= 0")
        if self._min_gain < 0:
            raise ValueError("min_gain must be >= 0")

        return n_entries

    def _initialize_heap(
        self,
        *,
        root: TopicTreeNode,
        prefix_sum: NDArray[np.float64],
        prefix_sq: NDArray[np.float64],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> tuple[list[tuple[float, int, int, int, int, int, TopicTreeNode]], int]:
        """Initialize heap with the best split candidate for the root."""
        heap: list[tuple[float, int, int, int, int, int, TopicTreeNode]] = []
        counter = 0
        root_candidate = self._best_split_candidate(
            node=root,
            prefix_sum=prefix_sum,
            prefix_sq=prefix_sq,
            entry_start_seconds=entry_start_seconds,
            entry_end_seconds=entry_end_seconds,
        )
        if root_candidate is not None and root_candidate.gain >= self._min_gain and root.depth < self._max_depth:
            heappush(
                heap,
                (
                    -root_candidate.gain,
                    root.depth,
                    root.start_entry_idx,
                    root.end_entry_idx_exclusive,
                    root_candidate.split_idx,
                    counter,
                    root,
                ),
            )
            counter += 1
        return heap, counter

    def _apply_splits(
        self,
        *,
        heap: list[tuple[float, int, int, int, int, int, TopicTreeNode]],
        counter: int,
        prefix_sum: NDArray[np.float64],
        prefix_sq: NDArray[np.float64],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> None:
        """Pop split candidates and apply them until no valid split remains."""
        while True:
            next_item = self._pop_valid_candidate(heap=heap)
            if next_item is None:
                break

            gain, split_idx, node = next_item
            if not self._can_split(node=node, gain=gain):
                continue

            left, right = self._split_node(node=node, split_idx=split_idx)
            counter = self._enqueue_child(
                child=left,
                heap=heap,
                counter=counter,
                prefix_sum=prefix_sum,
                prefix_sq=prefix_sq,
                entry_start_seconds=entry_start_seconds,
                entry_end_seconds=entry_end_seconds,
            )
            counter = self._enqueue_child(
                child=right,
                heap=heap,
                counter=counter,
                prefix_sum=prefix_sum,
                prefix_sq=prefix_sq,
                entry_start_seconds=entry_start_seconds,
                entry_end_seconds=entry_end_seconds,
            )

    def _pop_valid_candidate(
        self,
        *,
        heap: list[tuple[float, int, int, int, int, int, TopicTreeNode]],
    ) -> tuple[float, int, TopicTreeNode] | None:
        """Pop next candidate or return None if heap is empty."""
        if not heap:
            return None
        neg_gain, _, _, _, split_idx, _, node = heappop(heap)
        return -neg_gain, split_idx, node

    def _can_split(self, *, node: TopicTreeNode, gain: float) -> bool:
        """Check whether a node can be split."""
        if not node.is_leaf():
            return False
        if node.depth >= self._max_depth:
            return False
        return gain >= self._min_gain

    def _split_node(self, *, node: TopicTreeNode, split_idx: int) -> tuple[TopicTreeNode, TopicTreeNode]:
        """Split a node into left/right children."""
        left = TopicTreeNode(
            start_entry_idx=node.start_entry_idx,
            end_entry_idx_exclusive=split_idx,
            depth=node.depth + 1,
            parent=node,
        )
        right = TopicTreeNode(
            start_entry_idx=split_idx,
            end_entry_idx_exclusive=node.end_entry_idx_exclusive,
            depth=node.depth + 1,
            parent=node,
        )
        node.left = left
        node.right = right
        return left, right

    def _enqueue_child(
        self,
        *,
        child: TopicTreeNode,
        heap: list[tuple[float, int, int, int, int, int, TopicTreeNode]],
        counter: int,
        prefix_sum: NDArray[np.float64],
        prefix_sq: NDArray[np.float64],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> int:
        """Compute best split for a child and push onto heap if eligible."""
        if child.depth >= self._max_depth:
            return counter

        candidate = self._best_split_candidate(
            node=child,
            prefix_sum=prefix_sum,
            prefix_sq=prefix_sq,
            entry_start_seconds=entry_start_seconds,
            entry_end_seconds=entry_end_seconds,
        )
        if candidate is None or candidate.gain < self._min_gain:
            return counter

        heappush(
            heap,
            (
                -candidate.gain,
                child.depth,
                child.start_entry_idx,
                child.end_entry_idx_exclusive,
                candidate.split_idx,
                counter,
                child,
            ),
        )
        return counter + 1

    def _build_prefix_sums(
        self,
        embeddings: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build prefix sums for fast SSE queries."""
        emb64 = embeddings.astype(np.float64, copy=False)
        n_entries, emb_dim = emb64.shape

        prefix_sum = np.zeros((n_entries + 1, emb_dim), dtype=np.float64)
        prefix_sum[1:] = np.cumsum(emb64, axis=0)

        norms_sq = np.sum(emb64 * emb64, axis=1)
        prefix_sq = np.zeros(n_entries + 1, dtype=np.float64)
        prefix_sq[1:] = np.cumsum(norms_sq, axis=0)

        return prefix_sum, prefix_sq

    def _sse(
        self,
        *,
        start: int,
        end: int,
        prefix_sum: NDArray[np.float64],
        prefix_sq: NDArray[np.float64],
    ) -> float:
        """Compute within-segment sum of squared errors (SSE) for [start, end)."""
        if end <= start:
            raise ValueError("SSE segment must have end > start")

        seg_len = end - start
        sum_vec = prefix_sum[end] - prefix_sum[start]
        sum_sq = float(prefix_sq[end] - prefix_sq[start])
        mean_sq = float(np.dot(sum_vec, sum_vec)) / float(seg_len)
        return sum_sq - mean_sq

    def _segment_duration_seconds(
        self,
        *,
        start: int,
        end: int,
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> float:
        """Compute segment duration in seconds for [start, end)."""
        if end <= start:
            raise ValueError("Duration segment must have end > start")
        return float(entry_end_seconds[end - 1] - entry_start_seconds[start])

    def _best_split_candidate(
        self,
        *,
        node: TopicTreeNode,
        prefix_sum: NDArray[np.float64],
        prefix_sq: NDArray[np.float64],
        entry_start_seconds: list[float],
        entry_end_seconds: list[float],
    ) -> SplitCandidate | None:
        """Find the best split for a node that satisfies constraints."""
        start = node.start_entry_idx
        end = node.end_entry_idx_exclusive
        seg_len = end - start

        if seg_len < 2 * self._min_leaf_entries:
            return None

        if (
            self._segment_duration_seconds(
                start=start,
                end=end,
                entry_start_seconds=entry_start_seconds,
                entry_end_seconds=entry_end_seconds,
            )
            < self._min_leaf_seconds
        ):
            return None

        parent_sse = self._sse(start=start, end=end, prefix_sum=prefix_sum, prefix_sq=prefix_sq)

        best_child_sse = float("inf")
        best_split: int | None = None

        split_start = start + self._min_leaf_entries
        split_end_inclusive = end - self._min_leaf_entries

        for k in range(split_start, split_end_inclusive + 1):
            left_dur = float(entry_end_seconds[k - 1] - entry_start_seconds[start])
            right_dur = float(entry_end_seconds[end - 1] - entry_start_seconds[k])

            if left_dur < self._min_leaf_seconds or right_dur < self._min_leaf_seconds:
                continue

            child_sse = self._sse(start=start, end=k, prefix_sum=prefix_sum, prefix_sq=prefix_sq) + self._sse(
                start=k,
                end=end,
                prefix_sum=prefix_sum,
                prefix_sq=prefix_sq,
            )

            if child_sse < best_child_sse:
                best_child_sse = child_sse
                best_split = k

        if best_split is None:
            return None

        gain = parent_sse - best_child_sse
        return SplitCandidate(split_idx=best_split, gain=gain)
