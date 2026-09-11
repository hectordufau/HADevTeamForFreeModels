# Tests for harness/task
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.task import TaskManager, TaskContract, TaskError


def test_create_valid_task():
    mgr = TaskManager()
    contract = mgr.create({
        "apiVersion": "harness/v1",
        "kind": "Task",
        "metadata": {"id": "TASK-0001", "title": "Test task"},
        "spec": {
            "objective": "Implement something",
            "requirements": ["req1"],
            "acceptance_criteria": ["test passes"],
            "allowed_changes": ["src/**"],
            "verification": {"required": ["unit_tests"]},
            "autonomy": {"maximum": 3},
        },
    })
    assert contract.is_valid()
    assert contract.metadata.id == "TASK-0001"
    assert contract.spec.objective == "Implement something"


def test_create_task_without_id():
    mgr = TaskManager()
    contract = mgr.create({
        "apiVersion": "harness/v1",
        "kind": "Task",
        "spec": {
            "objective": "Implement something",
            "acceptance_criteria": ["test passes"],
        },
    })
    assert contract.metadata.id.startswith("TASK-")
    assert contract.is_valid()


def test_invalid_task_missing_objective():
    mgr = TaskManager()
    contract = mgr.create({
        "apiVersion": "harness/v1",
        "kind": "Task",
        "metadata": {"id": "TASK-X"},
        "spec": {
            "acceptance_criteria": ["test passes"],
        },
    })
    assert not contract.is_valid()
    errors = contract.validate()
    assert any("objective" in e.lower() for e in errors)


def test_save_and_load_task():
    mgr = TaskManager()
    contract = mgr.create({
        "apiVersion": "harness/v1",
        "kind": "Task",
        "metadata": {"id": "TASK-SAVE", "title": "Save test"},
        "spec": {
            "objective": "Test save/load",
            "acceptance_criteria": ["loads correctly"],
            "autonomy": {"maximum": 2},
        },
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        mgr.save(contract, f.name)
        f.flush()
        loaded = mgr.load(f.name)
        assert loaded.metadata.id == "TASK-SAVE"
        assert loaded.spec.objective == "Test save/load"
        assert loaded.spec.autonomy["maximum"] == 2
        os.unlink(f.name)


if __name__ == "__main__":
    test_create_valid_task()
    test_create_task_without_id()
    test_invalid_task_missing_objective()
    test_save_and_load_task()
    print("All task tests passed!")
