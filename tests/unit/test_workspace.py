# Tests for WorkspaceManager
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.workspace import WorkspaceManager, Conflict, WorkspaceManagerError


def test_acquire_non_conflicting():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**", "tests/**"]) is True
    assert wm.is_owned("node1", "src/main.py") is True


def test_acquire_conflicting():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**"]) is True
    # Same pattern by different node
    assert wm.acquire("node2", ["src/**"]) is False


def test_acquire_overlapping_pattern():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**"]) is True
    # Overlapping pattern (both have **)
    assert wm.acquire("node2", ["src/**/*.py"]) is False


def test_release_frees():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**", "tests/**"]) is True
    wm.release("node1")
    # After release, another node can acquire
    assert wm.acquire("node2", ["src/**"]) is True
    assert wm.is_owned("node2", "src/main.py") is True


def test_get_conflicts():
    wm = WorkspaceManager()
    wm.acquire("node1", ["src/**"])
    wm.acquire("node2", ["src/**"])
    conflicts = wm.get_conflicts()
    assert len(conflicts) >= 1
    conflict = conflicts[0]
    assert conflict.node_a == "node2"
    assert conflict.node_b == "node1"
    assert conflict.file_pattern == "src/**"


def test_get_owner():
    wm = WorkspaceManager()
    wm.acquire("node1", ["src/**"])
    assert wm.get_owner("src/main.py") == "node1"
    assert wm.get_owner("tests/main.py") is None


def test_clear():
    wm = WorkspaceManager()
    wm.acquire("node1", ["src/**"])
    wm.acquire("node2", ["tests/**"])
    wm.clear()
    assert wm.get_conflicts() == []
    assert wm.get_owner("src/main.py") is None


def test_read_access_not_restricted():
    wm = WorkspaceManager()
    # By design, read access is not restricted — only write conflicts matter
    # This test verifies the design assumption
    assert wm.acquire("node1", ["src/**"]) is True
    # Reading is always allowed — no read tracking
    assert wm.get_owner("tests/main.py") is None  # Not owned for writes


def test_different_patterns_no_conflict():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**"]) is True
    assert wm.acquire("node2", ["docs/**"]) is True
    assert wm.acquire("node3", ["tests/**"]) is True


def test_patterns_overlap():
    wm = WorkspaceManager()
    assert wm._patterns_overlap("src/**", "src/**") is True
    assert wm._patterns_overlap("src/**", "docs/**") is False
    assert wm._patterns_overlap("src/**", "src/**/*.py") is True
    assert wm._patterns_overlap("**/*", "**/*") is True
    assert wm._patterns_overlap("src/app/**", "src/**") is True


def test_same_node_no_conflict():
    wm = WorkspaceManager()
    assert wm.acquire("node1", ["src/**"]) is True
    # Same node re-acquiring same pattern
    assert wm.acquire("node1", ["tests/**"]) is True


if __name__ == "__main__":
    test_acquire_non_conflicting()
    test_acquire_conflicting()
    test_acquire_overlapping_pattern()
    test_release_frees()
    test_get_conflicts()
    test_get_owner()
    test_clear()
    test_read_access_not_restricted()
    test_different_patterns_no_conflict()
    test_patterns_overlap()
    test_same_node_no_conflict()
    print("All workspace tests passed!")
