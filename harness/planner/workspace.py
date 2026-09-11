# harness/planner/workspace.py — WorkspaceManager (V2.2)
"""
Manages file ownership during parallel execution to prevent conflicts.
Acquire/release pattern for write access; read access is not restricted.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import fnmatch
import threading


class WorkspaceManagerError(Exception):
    """Raised on workspace conflicts."""


@dataclass
class Conflict:
    """A workspace conflict between two nodes."""
    node_a: str
    node_b: str
    file_pattern: str


class WorkspaceManager:
    """Manages file ownership during parallel execution."""

    def __init__(self):
        self._locks: Dict[str, str] = {}  # file_pattern -> node_id
        self._lock_obj = threading.Lock()
        self._conflicts: List[Conflict] = []

    def acquire(self, node_id: str, files: List[str]) -> bool:
        """
        Try to acquire write access to files.

        Args:
            node_id: The node requesting access.
            files: List of file patterns (glob-style) to write to.

        Returns:
            True if all files acquired, False if conflict.
        """
        with self._lock_obj:
            for pattern in files:
                # Check for conflicts with existing locks
                for locked_pattern, locked_node in self._locks.items():
                    if locked_node != node_id and self._patterns_overlap(pattern, locked_pattern):
                        self._conflicts.append(Conflict(
                            node_a=node_id,
                            node_b=locked_node,
                            file_pattern=pattern,
                        ))
                        # Don't return False yet, record all conflicts
                        # Then fail for the first one
                        return False

            # Acquire locks
            for pattern in files:
                self._locks[pattern] = node_id
            return True

    def release(self, node_id: str):
        """Release all files owned by a node."""
        with self._lock_obj:
            to_remove = [
                pattern for pattern, owner in self._locks.items()
                if owner == node_id
            ]
            for pattern in to_remove:
                del self._locks[pattern]

    def get_conflicts(self) -> List[Conflict]:
        """Return all current workspace conflicts."""
        with self._lock_obj:
            return list(self._conflicts)

    def is_owned(self, node_id: str, file_path: str) -> bool:
        """Check if a file is owned by a specific node."""
        with self._lock_obj:
            for pattern, owner in self._locks.items():
                if owner == node_id and fnmatch.fnmatch(file_path, pattern):
                    return True
            return False

    def get_owner(self, file_path: str) -> Optional[str]:
        """Get the node that owns a file pattern matching the given path."""
        with self._lock_obj:
            for pattern, owner in self._locks.items():
                if fnmatch.fnmatch(file_path, pattern):
                    return owner
            return None

    def clear(self):
        """Release all locks."""
        with self._lock_obj:
            self._locks.clear()
            self._conflicts.clear()

    def _patterns_overlap(self, pattern_a: str, pattern_b: str) -> bool:
        """Check if two glob patterns could match overlapping files."""
        # Simple check: if patterns are equal or one is a subset
        if pattern_a == pattern_b:
            return True

        # If both have **, check if their non-wildcard prefixes overlap
        if "**" in pattern_a and "**" in pattern_b:
            prefix_a = pattern_a.split("*")[0].rstrip("/")
            prefix_b = pattern_b.split("*")[0].rstrip("/")
            # If either prefix is empty, they match everything - potential conflict
            if not prefix_a or not prefix_b:
                return True
            # Check if one prefix is a prefix of the other
            if prefix_a.startswith(prefix_b) or prefix_b.startswith(prefix_a):
                return True
            # If they have different prefixes, no overlap
            return False

        # Check if one pattern is a prefix of the other
        base_a = pattern_a.split("*")[0].rstrip("/")
        base_b = pattern_b.split("*")[0].rstrip("/")
        if base_a and base_b:
            if base_a.startswith(base_b) or base_b.startswith(base_a):
                return True

        return False
