# Tests for harness/state
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.state import StateManager, StateError, TaskState


def test_valid_transition():
    state = TaskState(task_id="TASK-1")
    state.transition("ANALYZING")
    assert state.current_state == "ANALYZING"
    assert len(state.history) == 1
    assert state.history[0].from_state == "RECEIVED"
    assert state.history[0].to_state == "ANALYZING"


def test_invalid_transition():
    state = TaskState(task_id="TASK-2")
    try:
        state.transition("COMPLETED")
        assert False, "Should have raised StateError"
    except StateError:
        pass


def test_full_lifecycle():
    state = TaskState(task_id="TASK-3")
    transitions = ["ANALYZING", "PLANNED", "IMPLEMENTING", "VERIFYING", "EVALUATING", "REVIEWING", "COMPLETED"]
    for t in transitions:
        state.transition(t)
    assert state.current_state == "COMPLETED"
    assert state.is_terminal()


def test_persist_and_restore():
    import tempfile
    mgr = StateManager()
    state = mgr.register("TASK-4")
    state.transition("ANALYZING")

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        mgr.persist("TASK-4", f.name)
        f.flush()

        mgr2 = StateManager()
        restored = mgr2.restore("TASK-4", f.name)
        assert restored.current_state == "ANALYZING"
        assert len(restored.history) == 1
        os.unlink(f.name)


if __name__ == "__main__":
    test_valid_transition()
    test_invalid_transition()
    test_full_lifecycle()
    test_persist_and_restore()
    print("All state tests passed!")
