# Tests for Replanner
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.replanner import Replanner, ReplanReport, ReplannerError
from harness.planner.workflow_planner import WorkflowNode, ExecutionWorkflow
from harness.planner.failure_analyzer import FailureReport


class MockTask:
    metadata = type('obj', (object,), {"id": "T-001"})
    spec = type('obj', (object,), {"objective": "test"})


def test_replan_produces_new_workflow():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(
        node_id="b",
        capability="impl",
        category="implementation",
        message="TypeError occurred",
        missing_capabilities=["debugging"],
    )

    replanner = Replanner(max_plans=3)
    new_workflow = replanner.replan(MockTask(), workflow, ["a"], "b", report)
    assert new_workflow is not None
    # Completed nodes preserved
    new_node_ids = {n.id for n in new_workflow.nodes}
    assert "a" in new_node_ids  # completed
    assert "b" in new_node_ids  # failed but still in workflow


def test_completed_nodes_frozen():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(
        node_id="c", capability="test", category="test",
        message="test failure",
    )

    replanner = Replanner()
    new_workflow = replanner.replan(MockTask(), workflow, ["a", "b"], "c", report)
    # a and b are completed — they should still exist
    assert "a" in {n.id for n in new_workflow.nodes}
    assert "b" in {n.id for n in new_workflow.nodes}


def test_max_plans_enforced():
    report = FailureReport(
        node_id="b", capability="impl", category="implementation",
        message="error",
    )

    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    replanner = Replanner(max_plans=2)
    # First two should succeed
    replanner.replan(MockTask(), workflow, ["a"], "b", report)
    replanner.replan(MockTask(), workflow, ["a"], "b", report)
    # Third should fail
    try:
        replanner.replan(MockTask(), workflow, ["a"], "b", report)
        assert False, "Should have raised"
    except ReplannerError as e:
        assert "Max replans" in str(e)


def test_replan_report_generated():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(
        node_id="b", capability="impl", category="implementation",
        message="failed",
    )

    replanner = Replanner()
    new_workflow = replanner.replan(MockTask(), workflow, ["a"], "b", report)
    assert hasattr(new_workflow, '_replan_report')
    assert new_workflow._replan_report.failed_node_id == "b"
    assert new_workflow._replan_report.category == "implementation"


def test_generate_replan_report_dict():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(
        node_id="b", capability="impl", category="implementation",
        message="failed",
    )

    replanner = Replanner()
    new_workflow = replanner.replan(MockTask(), workflow, ["a"], "b", report)
    r = replanner.generate_replan_report(workflow, new_workflow, "test failure")
    assert "original_workflow" in r
    assert "new_workflow" in r
    assert "reason" in r
    assert "failed_node_id" in r


def test_replan_unknown_completed_node_raises():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
        ],
    )

    report = FailureReport(
        node_id="a", capability="arch", category="implementation",
        message="error",
    )

    replanner = Replanner()
    try:
        replanner.replan(MockTask(), workflow, ["unknown"], "a", report)
        assert False, "Should have raised"
    except ReplannerError:
        pass


def test_get_and_reset_plan_count():
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["c"]),
        ],
    )

    report = FailureReport(
        node_id="b", capability="impl", category="implementation",
        message="error",
    )

    replanner = Replanner()
    assert replanner.get_plan_count("T-001") == 0
    replanner.replan(MockTask(), workflow, ["a"], "b", report)
    assert replanner.get_plan_count("T-001") == 1
    replanner.reset_plan_count("T-001")
    assert replanner.get_plan_count("T-001") == 0


if __name__ == "__main__":
    test_replan_produces_new_workflow()
    test_completed_nodes_frozen()
    test_max_plans_enforced()
    test_replan_report_generated()
    test_generate_replan_report_dict()
    test_replan_unknown_completed_node_raises()
    test_get_and_reset_plan_count()
    print("All replanner tests passed!")
