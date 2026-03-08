"""Internal data types for hierarchical topic trees."""

from __future__ import annotations

from dataclasses import dataclass


def make_node_id(start_entry_idx: int, end_entry_idx_exclusive: int, depth: int) -> str:
    """Build a deterministic node id."""
    return f"{start_entry_idx}:{end_entry_idx_exclusive}:d{depth}"


@dataclass
class TopicTreeNode:
    """A node in a binary hierarchical topic tree (internal representation)."""

    start_entry_idx: int
    end_entry_idx_exclusive: int
    depth: int
    parent: TopicTreeNode | None
    left: TopicTreeNode | None = None
    right: TopicTreeNode | None = None

    def is_leaf(self) -> bool:
        """Return True if the node has no children."""
        return self.left is None and self.right is None

    def node_id(self) -> str:
        """Return the deterministic node id."""
        return make_node_id(self.start_entry_idx, self.end_entry_idx_exclusive, self.depth)
