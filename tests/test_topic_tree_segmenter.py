"""Tests for deterministic TreeSeg-style hierarchical segmentation."""

import json

import numpy as np

from src.topic_detection.hierarchy.data_types import TopicTreeNode
from src.topic_detection.hierarchy.treeseg_divisive_sse import TreeSegDivisiveSSETopicSegmenter


def preorder(node: TopicTreeNode) -> list[TopicTreeNode]:
    """Collect nodes in pre-order (root, left, right) without depending on implementation details."""
    nodes: list[TopicTreeNode] = []

    def visit(n: TopicTreeNode) -> None:
        nodes.append(n)
        left = n.left
        right = n.right
        if left is not None and right is not None:
            visit(left)
            visit(right)

    visit(node)
    return nodes


def node_repr(node: TopicTreeNode) -> dict[str, int | str]:
    """Create a stable representation of a node for determinism checks."""
    return {
        "node_id": node.node_id(),
        "start": node.start_entry_idx,
        "end": node.end_entry_idx_exclusive,
        "depth": node.depth,
    }


def test_treeseg_builds_expected_tree_and_is_deterministic() -> None:
    """Build a simple 3-cluster sequence and assert exact split points + determinism."""
    # 0..3: zeros, 4..7: ones, 8..11: tens.
    values = [0.0] * 4 + [1.0] * 4 + [10.0] * 4
    embeddings = np.array(values, dtype=np.float32).reshape(-1, 1)

    entry_start_seconds = [float(i * 10) for i in range(len(values))]
    entry_end_seconds = [float((i + 1) * 10) for i in range(len(values))]

    segmenter = TreeSegDivisiveSSETopicSegmenter(
        max_depth=2,
        min_leaf_entries=4,
        min_leaf_seconds=0.0,
        min_gain=0.0,
    )

    root1 = segmenter.build_tree(
        embeddings=embeddings,
        entry_start_seconds=entry_start_seconds,
        entry_end_seconds=entry_end_seconds,
    )
    root2 = segmenter.build_tree(
        embeddings=embeddings,
        entry_start_seconds=entry_start_seconds,
        entry_end_seconds=entry_end_seconds,
    )

    rep1 = [node_repr(n) for n in preorder(root1)]
    rep2 = [node_repr(n) for n in preorder(root2)]

    assert rep1 == rep2

    expected_ids = [
        "0:12:d0",  # root
        "0:8:d1",  # left
        "0:4:d2",  # left-left
        "4:8:d2",  # left-right
        "8:12:d1",  # right
    ]
    assert [r["node_id"] for r in rep1] == expected_ids

    # Byte-identical JSON serialization for the stable representation.
    json1 = json.dumps(rep1, sort_keys=True, separators=(",", ":"))
    json2 = json.dumps(rep2, sort_keys=True, separators=(",", ":"))
    assert json1 == json2
