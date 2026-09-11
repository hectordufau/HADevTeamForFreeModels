# Tests for BranchResolver
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.branching import BranchResolver, BranchingError
from harness.planner.workflow_planner import WorkflowNode


def test_pass_to_next():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="next", on_fail="block")
    result = type('obj', (object,), {"status": "completed"})
    target = resolver.resolve(node, result)
    assert target == "next"


def test_fail_to_block():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="next", on_fail="block")
    result = type('obj', (object,), {"status": "failed"})
    target = resolver.resolve(node, result)
    assert target == "block"


def test_pass_to_declared_target():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="review", on_fail="debug")
    result = type('obj', (object,), {"status": "completed"})
    target = resolver.resolve(node, result)
    assert target == "review"


def test_fail_to_declared_target():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="review", on_fail="debug")
    result = type('obj', (object,), {"status": "failed"})
    target = resolver.resolve(node, result)
    assert target == "debug"


def test_pass_to_complete():
    resolver = BranchResolver()
    node = WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free",
                       on_pass="complete", on_fail="block")
    result = type('obj', (object,), {"status": "completed"})
    target = resolver.resolve(node, result)
    assert target == "complete"


def test_skip_to_block():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="next", on_fail="block")
    result = type('obj', (object,), {"status": "skipped"})
    target = resolver.resolve(node, result)
    assert target == "block"


def test_no_branching_defaults_to_next():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free")
    result = type('obj', (object,), {"status": "completed"})
    target = resolver.resolve(node, result)
    assert target == "next"


def test_dict_node():
    resolver = BranchResolver()
    node = {"id": "test", "capability": "testing", "agent": "tester",
            "model_id": "mock:free", "on_pass": "next", "on_fail": "block"}
    result = type('obj', (object,), {"status": "failed"})
    target = resolver.resolve(node, result)
    assert target == "block"


def test_dict_result():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="next", on_fail="block")
    result = {"status": "failed", "error": "something broke"}
    target = resolver.resolve(node, result)
    assert target == "block"


def test_empty_target_node_id_fails():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="", on_fail="")
    result = type('obj', (object,), {"status": "completed"})
    target = resolver.resolve(node, result)
    assert target == "next"


def test_retry_and_feedback():
    resolver = BranchResolver()
    node = WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free",
                       on_pass="next", on_fail="retry")
    result = type('obj', (object,), {"status": "failed"})
    target = resolver.resolve(node, result)
    assert target == "retry"

    node2 = WorkflowNode(id="test2", capability="testing", agent="tester", model_id="mock:free",
                        on_pass="next", on_fail="feedback")
    result2 = type('obj', (object,), {"status": "failed"})
    target2 = resolver.resolve(node2, result2)
    assert target2 == "feedback"


if __name__ == "__main__":
    test_pass_to_next()
    test_fail_to_block()
    test_pass_to_declared_target()
    test_fail_to_declared_target()
    test_pass_to_complete()
    test_skip_to_block()
    test_no_branching_defaults_to_next()
    test_dict_node()
    test_dict_result()
    test_empty_target_node_id_fails()
    test_retry_and_feedback()
    print("All branching tests passed!")
